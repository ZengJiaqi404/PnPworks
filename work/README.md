# work/ —— 第五次作业交付目录

```
work/
├── README.md                  ← 本文件（目录说明）
├── MANIFEST.txt               ← 打包清单 + SHA256 校验
├── rm-armor-pnp-src.zip       ← 源码包 1.6 MB（32 文件，传 GitHub 用）
├── rm-armor-pnp-full.zip      ← 完整包 30 MB（源码 + 3 段演示视频 + calib.yaml）
└── rm-armor-pnp/              ← 工程本体（可直接运行）
    ├── camera.py              取帧：MVS 相机 / 视频
    ├── calibrate.py           棋盘格标定（图片 / 录像抽帧）
    ├── armor.py               识别 + 角点排序 + PnP
    ├── kf.py                  卡尔曼滤波 + 预测
    ├── pipeline.py            实时主程序（可视化/录制/CSV）
    ├── record.py focus.py     录原始画面 / 对焦辅助
    ├── selftest.py            无相机数学自检
    ├── plot_kf.py compare_distances.py  出图工具
    ├── run.sh                 统一入口
    ├── calib.yaml             本机标定结果（1440×1080，RMS 0.749 px）
    ├── docs/                  演示视频、CSV、对比图、六步手册
    ├── captures/              运行截图与调试帧
    └── .git/                  单提交，1.8 MB
```

## 快速验证

```bash
cd ~/桌面/work/rm-armor-pnp
./run.sh --tool selftest        # 数学自检，应全 PASS
./run.sh --tool compare         # 复现两距离档对比（0.649 / 1.021 m）
./run.sh --exposure 9000 --gain 18   # 实机实时六步
```

桌面图标「装甲板 PnP 定位」已指向 `work/rm-armor-pnp/run.sh`。

## 上传 GitHub

```bash
cd ~/桌面/work/rm-armor-pnp
git remote add origin git@github.com:<用户名>/<仓库>.git
git push -u origin main
```

`weights/best.pt` 不入库：克隆后放到 `weights/`，或用 `--weights` / `ARMOR_WEIGHTS` 指定。
