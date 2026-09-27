#!/usr/bin/env bash
# =============================================================================
# NInfer · Bonsai-2-27B 启动脚本（OpenAI 兼容接口）— v2（2026-09-27 新引擎版）
#
# 用法:
#   ./start-ninfer.sh [profile] [port]     启动（默认 profile=work 端口=8088）
#   ./start-ninfer.sh stop    [profile]    停止
#   ./start-ninfer.sh status  [profile]    状态
#
# profile（4070 Ti 12GB + 核显显示实测调优）:
#   work   默认。新官方引擎 + PQ2 制品，64k fp8，MTP K=3，--no-prefix-reuse。
#          prefill 1823 t/s（旧版 +79%）、TTFT 190ms、decode ~76-84 t/s。富余 1.85GB。
#   long   同引擎 112k fp8（实测上限；131k 差 370MB，262k 需 9.2GB 放不下）。
#   fast   同引擎 8k fp8 —— 最省显存最稳（边办公边跑）。
#
# 已知事项:
#   · 新引擎必须加 --no-prefix-reuse，否则第二个请求会 500
#     （candidate token ledger does not match prompt length —— 上游 bug，等修复）。
#   · MTP 上限 K=5（内核 T≤6 硬限，源码已打 K7 补丁但内核不可越）；
#     dflash2 需要制品含 DFlash2 bundle（现制品都没有）。
#   · 262k fp8 需要 ~9.2GB 运行时预留，12GB 卡放不下，long 档封顶 112k。
# =============================================================================
set -uo pipefail

LOGDIR="$HOME/ninfer-logs"
SANYUAN="$HOME/pyprojects/sanyuan"
ART_PQ2="$SANYUAN/artifacts-pq2.ninfer"

ENG_NEW="$SANYUAN/ninfer-4090w-ternary"           # 沈三殊官方线（2026-09-26 树，唯一保留）

PROFILE="${1:-work}"
ACTION="start"
if [[ "$1" == "stop" || "$1" == "status" ]]; then
    ACTION="$1"; PROFILE="${2:-work}"
    PORT="${3:-8088}"
else
    PORT="${2:-8088}"
fi

# 思考模式: 默认开启；NINFER_THINKING=off ./start-ninfer.sh 可整体关闭
THINKING_FLAG=""
[[ "${NINFER_THINKING:-on}" == "off" ]] && THINKING_FLAG="--no-thinking"

PIDFILE="$LOGDIR/ninfer-$PROFILE.pid"
LOGFILE="$LOGDIR/ninfer-$PROFILE.log"
mkdir -p "$LOGDIR"

profile_spec() {  # 输出: 引擎目录|制品|引擎参数
    case "$PROFILE" in
        work)   echo "$ENG_NEW|$ART_PQ2|--max-context 65536 --kv-capacity 65536 --kv-dtype fp8 --spec mtp --draft-tokens 3 --no-prefix-reuse" ;;
        long)   echo "$ENG_NEW|$ART_PQ2|--max-context 114688 --kv-capacity 114688 --kv-dtype fp8 --spec mtp --draft-tokens 3 --no-prefix-reuse" ;;
        fast)   echo "$ENG_NEW|$ART_PQ2|--max-context 8192 --kv-capacity 8192 --kv-dtype fp8 --spec mtp --draft-tokens 3 --no-prefix-reuse" ;;
        *)      return 1 ;;
    esac
}

is_running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

do_start() {
    local spec EDIR ART ARGS
    spec="$(profile_spec)" || { echo "未知 profile: $PROFILE (work|long|fast)"; exit 1; }
    IFS='|' read -r EDIR ART ARGS <<< "$spec"
    [[ -f "$ART" ]] || { echo "错误: 制品不存在: $ART"; exit 1; }
    if is_running; then
        echo "[$PROFILE] 已在运行 (pid $(cat "$PIDFILE")), http://127.0.0.1:$PORT/v1"; exit 0
    fi
    echo "[$PROFILE] 引擎: $EDIR"
    echo "  制品: $ART"
    echo "  参数: $ARGS --port $PORT"
    ( cd "$EDIR" && export TMPDIR="$EDIR/tmp-nvcc" && mkdir -p "$TMPDIR" \
        && nohup ./build/apps/ninfer-serve "$ART" \
            --host 127.0.0.1 --port "$PORT" --model-id qwen3.8-27b \
            $ARGS $THINKING_FLAG --max-concurrency 1 \
            > "$LOGFILE" 2>&1 & echo $! > "$PIDFILE" )
    echo -n "  等待就绪"
    for _ in $(seq 1 90); do
        if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo " 就绪"
            echo "  OpenAI 接口: http://127.0.0.1:$PORT/v1  (模型 id: qwen3.8-27b)"
            echo "  日志: $LOGFILE | 停止: $0 stop $PROFILE"
            exit 0
        fi
        if ! pgrep -x ninfer-serve >/dev/null 2>&1; then
            echo " 启动失败:"; tail -4 "$LOGFILE"; rm -f "$PIDFILE"; exit 1
        fi
        echo -n "."; sleep 1
    done
    echo " 超时(90s), 日志: $LOGFILE"; exit 1
}

do_stop() {
    if is_running; then
        kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE"
        echo "[$PROFILE] 已停止"
    else
        rm -f "$PIDFILE"; echo "[$PROFILE] pidfile 无效/未在运行"
    fi
    local pids
    pids=$(pgrep -x ninfer-serve 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null
        for _ in $(seq 1 15); do
            sleep 1
            pids=$(pgrep -x ninfer-serve 2>/dev/null || true)
            [[ -z "$pids" ]] && break
        done
        echo "  显存已释放"
    fi
}

do_status() {
    if is_running; then
        echo "[$PROFILE] 运行中 (pid $(cat "$PIDFILE"))"
        curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 200; echo
    else
        echo "[$PROFILE] 未在运行"
    fi
}

case "$ACTION" in
    start)  do_start ;;
    stop)   do_stop ;;
    status) do_status ;;
    *) echo "用法: $0 [work|long|fast] [port] | stop|status [profile]"; exit 1 ;;
esac
