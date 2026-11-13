# 装甲板定位：相机标定 → 识别 → 角点 → PnP → 卡尔曼滤波

猎隼战队算法组第五次作业的完整实现。整条链路：

```
MVS 相机采集 → YOLO-pose 装甲板识别 → 四角点排序 → solvePnP 位姿解算
            → 三维位置(X/Y/Z)与距离 → 卡尔曼滤波 → 运动状态估计与短时预测
```

主程序会把 **原始测量 / 滤波后位置 / 预测位置** 同时画在画面上，并把逐帧数据写成 CSV，
方便做「滤波前后对比」的作业分析。

---

## 1. 目录结构

| 文件 | 作用 |
|---|---|
| `camera.py` | 取帧：MVS 工业相机（GigE/USB3）+ 视频文件/USB 摄像头兜底 |
| `calibrate.py` | 棋盘格标定 → `calib.yaml`（内参 K、畸变系数、RMS） |
| `armor.py` | YOLO 识别、四角点几何排序、PnP（IPPE + LM）、坐标轴绘制 |
| `kf.py` | 卡尔曼滤波（CV 模型，6 状态）与纯函数预测 |
| `pipeline.py` | 实时主程序：可视化、录制、CSV/JSON 导出、快捷键 |
| `plot_kf.py` | 生成对比图（实测 CSV 或仿真真值两种） |
| `selftest.py` | 无相机自检：合成棋盘格标定 + 已知位姿 PnP 反解 |
| `run.sh` | 统一入口 |

---

## 2. 快速开始

```bash
cd ~/桌面/rm-armor-pnp

./run.sh --tool list          # 1) 看能不能枚举到相机
./run.sh --tool calibrate     # 2) 标定（棋盘格 9x6，边长按实物填）
./run.sh                      # 3) 实时运行
./run.sh --tool selftest      # 无需相机的数学自检
./run.sh --tool plot --synthetic   # 生成滤波对比图（仿真真值）
```

没有相机时可以先跑通流程：

```bash
./run.sh --source ~/桌面/noname/armor-inference/videos/d414.mp4
```

桌面图标 `装甲板PnP定位.desktop` 等价于 `./run.sh`。

---

## 3. 相机标定（作业第 1 项）

```bash
./run.sh --tool calibrate --cols 9 --rows 6 --square 25
```

- `--cols/--rows` 填**内角点数**（9x6 表示横 9 个、竖 6 个交点），`--square` 填方格边长（mm）。
- 把棋盘格放在画面里，切换不同位置/角度，检测到角点时按 **空格** 存一张，至少 8 张（建议 15+），
  按 **回车** 开始标定，**ESC** 放弃。
- 输出 `calib.yaml`：`camera_matrix`、`distortion_coefficients`、`rms`、标定分辨率。

判读标准：`RMS < 0.5 px` 很好，`< 1.0 px` 可接受。同时会打印 fx/fy/cx/cy、视场角和畸变系数。

也可以用已经拍好的图片：

```bash
./run.sh --tool calibrate --images assets/calib_frames
```

> ⚠️ 标定分辨率必须和实际运行分辨率一致，否则内参失效（程序会打印 warning 提醒）。

---

## 4. 识别与角点（作业第 2 项）

- 模型：`best.pt`（YOLO pose，6 类 B1/B2/B3/R1/R2/R3，每个目标 4 个关键点，`kpt_shape=[4,2]`）。
- 关键点顺序在该权重里并不稳定，因此 `armor.py:order_corners()` 不依赖模型输出顺序，
  而是按几何位置排序成 **左上、右上、右下、左下**：
  1. 按绕质心的极角排成一圈；
  2. 从 `x+y` 最小的点（左上）开始；
  3. 检查第二点是否在第四点右侧，否则整体反向。
- 还会做凸性、面积、最小边长检查，过滤退化的四边形。

---

## 5. PnP 位姿解算（作业第 3 项）

**关键点跨度 = 两根灯条围成的矩形**（实测结论，不是整块板的外框）：

```
--armor-w = 灯条外沿之间的横向距离（本板 224 mm）
--armor-h = 灯条长度（本板 54 mm）
```

物体点（米）：

```
TL(-w/2, -h/2, 0)  TR(+w/2, -h/2, 0)  BR(+w/2, +h/2, 0)  BL(-w/2, +h/2, 0)
```

判据：关键点像素跨度的宽高比应接近 `w/h`（本板 224/54 ≈ 4.15）。用整板尺寸 224×126
代入会得到 1.78 的比例，与实测 4.2 明显不符，PnP 会因重投影误差超限而失败。

- `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)`：平面靶标的解析解，会给出**两组**解；
- 两组解都用 `solvePnPRefineLM` 精化，按重投影误差择优；误差接近（<1.5 px）时选与上一帧更连续的那组，
  避免平面靶标的翻转二义性造成距离跳变；
- 重投影误差 > 6 px 的解直接丢弃；
- 输出：`tvec`（X/Y/Z，相机系：x 右、y 下、z 前）、距离 `‖t‖`、yaw/pitch/roll。

画面上会画出四角点（绿）、编号、以及 RGB 三色坐标轴（X 红 / Y 绿 / Z 蓝）。

---

## 6. 卡尔曼滤波与预测（作业第 4 项）

状态 `x = [X, Y, Z, Vx, Vy, Vz]`，观测 `z = [X, Y, Z]`（PnP 直接测量）：

```
x(k+1) = F(dt)·x(k) + w ,  F = [[I, dt·I], [0, I]]
z(k)   = H·x(k) + v     ,  H = [I, 0]
```

- 过程噪声用加速度白噪声模型 `Q = G·q·Gᵀ`，`G = [dt²/2·I; dt·I]`；
- `dt` 用真实时间戳，帧率波动时自动适配；
- 门限：`|测量 - 预测| > --kf-gate`（默认 0.6 m）判定为跳变/误检 → 重置滤波器；
- 超过 0.15 s 无检测、或目标类别切换 → 重置；
- 预测用**纯函数** `predict_at(dt) = p + v·dt`，不走 `KalmanFilter.predict()`，
  避免多推一帧把状态推进两次（这是很容易踩的坑，`kf.py` 顶部有注释）。

参数：`--kf-pos-std`（观测噪声，默认 0.04 m）、`--kf-acc-std`（过程噪声，默认 3.0 m/s²）。
默认值是用仿真轨迹扫参选出来的：静止/慢速时把 RMSE 从 19 mm 降到约 10 mm，快速运动约 13 mm。

---

## 7. 结果与证据（docs/ + captures/）

**实机标定**（MVS MV-CA016-10UC，1440×1080，26 mm 棋盘格 9×6）：

```
RMS 重投影误差 0.749 px      38 个有效样本
fx 2405.93  fy 2407.05      cx 713.58  cy 518.62
视场角 33.3° × 25.3°        畸变 [-0.030, 0.057, -0.005, -0.001, -0.545]
```

**两个距离档的实机结果**（同一标定、同一权重）：

| 档位 | 有效帧 | PnP 中位距离 | 稳定性 std | 卷尺 | 偏差 |
|---|---|---|---|---|---|
| 70 cm | 262 / 450 | **0.649 m** | **±3.5 mm** | 0.70 m | −7.2% |
| 120 cm | 450 / 450 | **1.021 m** | ±26.7 mm | 1.20 m | −14.9% |

**关键点跨度实测**（这一步决定了 PnP 能不能解出来）：

```
关键点像素跨度 619.5 × 147.1 px   比例 4.211
224 / 54  = 4.148   ← 吻合：四角点 = 两根灯条围成的矩形
224 / 126 = 1.778   ← 错误假设，PnP 直接因重投影超限失败
```

| 文件 | 内容 |
|---|---|
| `docs/dist_70cm.mp4` / `docs/dist_120cm.mp4` | 两个距离档的完整六步运行录像（带 HUD/角点/坐标轴/pred） |
| `docs/dist_70cm.csv` / `docs/dist_120cm.csv` | 逐帧数据：`raw_*` / `kf_*` / `pred_*` / `vel` / `dist` |
| `docs/distance_compare.png` | 两档对比：距离时间曲线、X-Z 轨迹、卷尺 vs PnP 柱状图 |
| `docs/kf_synthetic.png` | 仿真真值滤波对比：RMSE 20.3 → 11.9 mm（**改善 42%**），1 步预测 15.6 mm |
| `docs/kf_video.png` | 样品视频（演示内参）的 raw/filtered/predicted 曲线 |
| `docs/demo_result.mp4` / `docs/demo_frame.jpg` | 样品视频结果与单帧截图 |
| `captures/live_pose.jpg` | 实机单帧：B1、四角点、RGB 轴、X/Y/Z、欧拉角 |
| `captures/undistort_check.jpg` | 标定畸变校正前后对比 |

> 演示内参（`docs/kf_video.png`、`demo_result.mp4`）只用于流程演示；实机数据以 `dist_*.csv` 为准。

## 8. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--source` | 空（MVS 相机） | 视频文件/摄像头序号，调试用 |
| `--camera-index` / `--exposure` / `--gain` | 0 / 0 / 0 | 相机选择、固定曝光(us)、增益(dB) |
| `--weights` | 装甲板 `best.pt` | 换模型只改这里 |
| `--armor-w` / `--armor-h` | 0.224 / 0.054 | 装甲板实际尺寸（米），**必须和关键点对应的物理尺寸一致** |
| `--device` | auto | auto / cpu / 0（GPU） |
| `--imgsz` / `--conf` / `--iou` | 640 / 0.35 / 0.7 | 推理尺寸与阈值 |
| `--kf-pos-std` / `--kf-acc-std` / `--kf-gate` | 0.04 / 3.0 / 0.6 | 滤波噪声与跳变门限 |
| `--no-kf` | 关 | 关掉滤波只看原始 PnP |
| `--record` / `--csv` / `--report` | 空 | 录制视频 / 逐帧数据 / 统计 JSON |
| `--headless` / `--max-frames` | 关 / 0 | 无窗口压测 / 跑 N 帧退出 |

窗口快捷键：`q` 退出 · `s` 存图 · `r` 录制 · `p` 暂停 · `h` HUD · `k` 轨迹。

---

## 9. 自检（无需相机）

```bash
./run.sh --tool selftest
```

- 合成 12 张棋盘格视图 → 标定 → 与真值比较：fx/fy 误差 < 1%，PASS；
- 已知位姿投影 4 角点 → PnP 反解：位置误差 < 0.01 mm、角度误差 0°，PASS。

这就是「代码数学是对的，剩下的精度取决于你的标定和实际尺寸」的证据。

---

## 10. 交作业前

```bash
# 1) 标定
./run.sh --tool calibrate --cols 9 --rows 6 --square 25
# 2) 跑一段带滤波的录屏（改 armor 尺寸/曝光按需）
./run.sh --record docs/final_demo.mp4 --csv docs/final.csv --report docs/final.json
# 3) 生成对比图
./run.sh --tool plot --csv docs/final.csv --out docs/kf_compare.png
# 4) 截图要求：识别结果 + 坐标轴 + X/Y/Z/距离 + 滤波/预测数值
# 5) 推送
git add -A && git commit -m "feat: 装甲板 PnP 定位 + 卡尔曼滤波"
git remote add origin <你的仓库> && git push -u origin main
```

**作业注意点对照**

| 要求 | 实现位置 |
|---|---|
| 标定参数与实际相机/分辨率对应 | `calibrate.py` 保存分辨率；`pipeline.py` 检测不一致并告警 |
| 2D 角点与 3D 点一一对应 | `armor.py:order_corners()` 固定 TL/TR/BR/BL；`object_points` 同序 |
| 卡尔曼必须用连续帧实测数据 | 全程用 PnP 的连续输出，`kf.py` 按真实时间戳更新 |
| 固定相机做运动估计 | 默认 MVS 固定安装，输出相机系下的 X/Y/Z 与速度 |
| 展示原始/滤波/预测 | 画面 HUD + `docs/*.png` + CSV 三路数据 |

---

## 11. 常见问题

**Q: 重投影误差偏大（>5 px）、距离跳变？**
A: 多半是 `--armor-w/--armor-h` 与关键点对应的物理尺寸不一致。用卡尺量装甲板外框（含灯条外沿），
或先按 135×125 试，再微调到重投影误差最小。

**Q: 标定 RMS 很大（>2 px）？**
A: 棋盘格拍得太少/太正/有运动模糊；确保 `--cols/--rows` 与板子一致、方格边长填对、拍摄时板子平整。

**Q: 相机打不开（ret=0x80000203）？**
A: 相机独占，先关掉 MVS 客户端或另一个实例：`pkill -f pipeline.py`。

**Q: 画面很暗、帧率只有 1-2 FPS？**
A: 自动曝光在暗环境下被拉到几百毫秒。用 `--exposure 8000 --gain 12` 固定短曝光，并补光。

**Q: 没有 GPU 会怎样？**
A: 自动回退 CPU（`--device cpu`），1080p/imgsz640 约 13-17 FPS。
