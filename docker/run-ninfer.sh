#!/usr/bin/env bash
# =============================================================================
# 以容器方式启动 NInfer OpenAI 兼容服务（等价宿主机 start-ninfer.sh 的三档）
#
# 用法:
#   ./docker/run-ninfer.sh                  # 默认 work 档 64k 上下文，宿主端口 8089
#   ./docker/run-ninfer.sh -c 114688        # long 档（12GB 卡 fp8 上限 ~112k）
#   ./docker/run-ninfer.sh -p 8088          # 自定宿主端口（默认 8089 避让宿主服务）
#   ./docker/run-ninfer.sh -i <镜像tag>     # 指定镜像（50 系卡用 -arch120a tag；CUDA 版本后缀与 build-image.sh -b 一致）
#   ./docker/run-ninfer.sh -a <制品路径>    # 指定 .ninfer 制品（默认 sanyuan 下）
#   ./docker/run-ninfer.sh stop | logs | status
#
# 说明:
#   · 制品只读挂载不进镜像；容器内服务监听 8088，-p 映射到宿主
#   · 思考默认开启；NINFER_THINKING=off ./docker/run-ninfer.sh 整体关闭（与宿主一致）
#   · 显存预算与宿主直跑相同: 64k 需空闲 ~10GB，启动前确认 nvidia-smi < 2GB
# =============================================================================
set -euo pipefail

ART="$HOME/pyprojects/sanyuan/artifacts-pq2.ninfer"
CTX="65536"
PORT="8089"
IMAGE="ninfer-bonsai:cuda13.3-arch89"
NAME="ninfer-serve"

ACTION="run"
if [[ "${1:-}" == "stop" || "${1:-}" == "logs" || "${1:-}" == "status" ]]; then
    ACTION="$1"; shift
fi
while getopts "a:c:p:i:" o; do case "$o" in
    a) ART="$OPTARG" ;;
    c) CTX="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    i) IMAGE="$OPTARG" ;;
    *) grep '^#' "$0" | sed -n '2,18p'; exit 1 ;;
esac; done

DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

case "$ACTION" in
    stop)   $DOCKER rm -f "$NAME" && echo "[$NAME] 已停止（显存随容器释放）" ;;
    logs)   exec $DOCKER logs -f "$NAME" ;;
    status) $DOCKER ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
            curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" || echo "(接口未就绪)" ;;
    run)
        [[ -f "$ART" ]] || { echo "错误: 制品不存在: $ART" >&2; exit 1; }
        $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
        THINKING_FLAG=""
        if [[ "${NINFER_THINKING:-on}" == "off" ]]; then THINKING_FLAG="--no-thinking"; fi
        echo ">> 启动容器: 镜像 $IMAGE | 上下文 $CTX | 宿主端口 $PORT"
        $DOCKER run -d --name "$NAME" --gpus all \
            -p "$PORT:8088" \
            -v "$ART:/models/model.ninfer:ro" \
            "$IMAGE" \
            ninfer-serve /models/model.ninfer \
                --host 0.0.0.0 --port 8088 --model-id qwen3.8-27b \
                --max-context "$CTX" --kv-capacity "$CTX" --kv-dtype fp8 \
                --spec mtp --draft-tokens 3 --no-prefix-reuse \
                --max-concurrency 1 $THINKING_FLAG
        echo -n ">> 等待就绪"
        for _ in $(seq 1 120); do
            if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
                echo " 就绪"
                echo "  OpenAI 接口: http://127.0.0.1:$PORT/v1  (模型 id: qwen3.8-27b)"
                echo "  日志: $0 logs | 停止: $0 stop"
                exit 0
            fi
            if [[ -z "$($DOCKER ps --filter "name=$NAME" -q)" ]]; then
                echo " 容器退出，日志尾部:"; $DOCKER logs --tail 20 "$NAME"; exit 1
            fi
            echo -n "."; sleep 1
        done
        echo " 超时(120s)，日志: $0 logs"; exit 1 ;;
esac
