#!/usr/bin/env bash
# =============================================================================
# NInfer · Bonsai-2-27B 启动脚本（OpenAI 兼容接口）— v3（2026-09-27 深夜 A3 升级版）
#
# 用法:
#   ./start-ninfer.sh [profile] [port]     启动（默认 profile=work 端口=8088）
#   ./start-ninfer.sh stop    [profile]    停止
#   ./start-ninfer.sh status  [profile]    状态
#
# profile（4070 Ti 12GB + 核显显示实测调优）:
#   work   默认。A3 引擎（pkg2 r12 树+P1/P2/A3 补丁+SM_COUNT=60）+ PQ2 制品，
#          64k fp8，MTP K=3，前缀复用开（r12 已修账本 bug）。
#          prefill 2108-2209 t/s（旧版 1694-1837，+24.6% 经 pkg5 口径 A/B + sha 逐位门）、
#          TTFT 179ms、decode ~75 t/s、接受率 ~35%。显存 ~10.2GB。
#   long   同引擎 112k fp8（实测上限；131k 差 370MB，262k 需 9.2GB 放不下）。
#   fast   同引擎 8k fp8 —— 最省显存最稳（边办公边跑）。
#
# 已知事项:
#   · A3 收益开关 NINFER_TERNARY_TOKEN_GRID 默认开（pkg5 补丁，逐位无损已验证）；
#     必须配 NINFER_TERNARY_SMALL_T_ROWS=16：sched3 缺省 auto 会走到 32 行档，
#     该档有上游已知数值缺陷（"PTQ1_0 + 32 rows" PPL 算错）⇒ 本脚本显式钉 16。
#   · 前缀复用已开：旧树的账本 bug（第二个请求 500 candidate token ledger…）在本树
#     已被 pkg2 r12 修复（实测同 prompt 第二请求 200 且 prefill 86 t/s 缓存命中）。
#   · decode 想跑 150+ t/s：取决于任务可预测性——数数/模板/照抄类任务接受率 88-99%，
#     K=3 实测 145 t/s、K=5 实测 173 t/s；而说明文/对话类接受率仅 28-40%，K=3 ~80 t/s
#     才是正常水平。K 是启动参数（--draft-tokens，上限 5），按任务换档需重启。
#   · MTP 上限 K=5（内核 T≤6 硬限，源码已打 K7 补丁但内核不可越）；
#     dflash2 需要制品含 DFlash2 bundle（现制品都没有）。
#   · 回滚：把 ENG_NEW 改回 $SANYUAN/ninfer-4090w-ternary 并在参数里加回 --no-prefix-reuse
#     （旧树保留未动）。
# =============================================================================
set -uo pipefail

LOGDIR="$HOME/ninfer-logs"
SANYUAN="$HOME/pyprojects/sanyuan"
ART_PQ2="$SANYUAN/artifacts-pq2.ninfer"

ENG_NEW="$SANYUAN/ninfer-4090w-ternary-a3"        # A3 引擎（pkg2 r12 树 + P1/P2/A3 + SM_COUNT=60）
ENG_OLD="$SANYUAN/ninfer-4090w-ternary"           # 旧引擎（2026-09-26 树，回滚用，保留未动）

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

# A3 数值安全: rows 钉 16（绕开上游 32 行档已知缺陷），token grid 保持补丁缺省开
export NINFER_TERNARY_SMALL_T_ROWS=16

PIDFILE="$LOGDIR/ninfer-$PROFILE.pid"
LOGFILE="$LOGDIR/ninfer-$PROFILE.log"
mkdir -p "$LOGDIR"

profile_spec() {  # 输出: 引擎目录|制品|引擎参数
    case "$PROFILE" in
        work)   echo "$ENG_NEW|$ART_PQ2|--max-context 65536 --kv-capacity 65536 --kv-dtype fp8 --spec mtp --draft-tokens 3" ;;
        long)   echo "$ENG_NEW|$ART_PQ2|--max-context 114688 --kv-capacity 114688 --kv-dtype fp8 --spec mtp --draft-tokens 3" ;;
        fast)   echo "$ENG_NEW|$ART_PQ2|--max-context 8192 --kv-capacity 8192 --kv-dtype fp8 --spec mtp --draft-tokens 3" ;;
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
