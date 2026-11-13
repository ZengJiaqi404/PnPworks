#!/usr/bin/env bash
# 装甲板 PnP + 卡尔曼滤波 —— 统一入口
#   ./run.sh                       实时运行（MVS 相机）
#   ./run.sh --source xx.mp4       用视频跑
#   ./run.sh --tool calibrate      相机标定（棋盘格，图片/视频都可）
#   ./run.sh --tool record         录一段原始画面给标定用
#   ./run.sh --tool focus          对焦辅助（看 sharp 数值拧对焦环）
#   ./run.sh --tool list           列出 MVS 相机
#   ./run.sh --tool selftest       标定/PnP 数学自检（无需相机）
#   ./run.sh --tool plot           由 CSV 生成滤波对比图
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 解释器：优先用环境变量 PYTHON，其次常见虚拟环境，最后系统 python3
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in /home/zjq/pytorch_env/bin/python "$HOME/miniconda3/envs/torch/bin/python" "$(command -v python3)"; do
        [ -x "$cand" ] && PY="$cand" && break
    done
fi

export LD_LIBRARY_PATH="/opt/MVS/lib/64:${LD_LIBRARY_PATH:-}"
export YOLO_CONFIG_DIR="$HERE/.ultralytics"
export MPLCONFIGDIR="$HERE/.mplcache"
mkdir -p "$YOLO_CONFIG_DIR" "$HERE/captures" "$MPLCONFIGDIR" 2>/dev/null

if [ ! -x "$PY" ]; then
    echo "找不到 Python 环境：$PY"; read -rp "按回车关闭..." _; exit 1
fi
cd "$HERE" || exit 1

TOOL="run"
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tool)   TOOL="${2:-run}"; shift 2 ;;
        --tool=*) TOOL="${1#*=}"; shift ;;
        *)        ARGS+=("$1"); shift ;;
    esac
done

case "$TOOL" in
    calibrate|calib) "$PY" "$HERE/calibrate.py" ${ARGS[@]+"${ARGS[@]}"} ;;
    list)            "$PY" "$HERE/pipeline.py" --list-devices ;;
    selftest|test)   "$PY" "$HERE/selftest.py" ;;
    record)          "$PY" "$HERE/record.py" ${ARGS[@]+"${ARGS[@]}"} ;;
    focus)           "$PY" "$HERE/focus.py" ${ARGS[@]+"${ARGS[@]}"} ;;
    plot)            "$PY" "$HERE/plot_kf.py" ${ARGS[@]+"${ARGS[@]}"} ;;
    compare)         "$PY" "$HERE/compare_distances.py" ${ARGS[@]+"${ARGS[@]}"} ;;
    run|*)           "$PY" "$HERE/pipeline.py" ${ARGS[@]+"${ARGS[@]}"} ;;
esac
status=$?

if [ "$status" -ne 0 ] && [ -t 0 ]; then
    echo; read -rp "程序退出（状态码 $status），按回车关闭..." _
fi
exit "$status"
