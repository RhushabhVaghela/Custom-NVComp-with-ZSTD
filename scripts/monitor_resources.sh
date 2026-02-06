#!/bin/bash
# Resource Monitor for NVComp with ZSTD
# Prevents OOM by checking RAM and VRAM before operations
# Safety margins: 2GB RAM, 2GB VRAM

RAM_SAFETY_MB=2048
VRAM_SAFETY_MB=2048
CHECK_INTERVAL=2
LOG_FILE="/tmp/nvcomp_resource_monitor.log"

get_ram_free_mb() {
    free -m | awk '/^Mem:/ {print $7}'
}

get_vram_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

get_ram_used_mb() {
    free -m | awk '/^Mem:/ {print $3}'
}

get_vram_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

get_gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

check_safe() {
    local ram_free=$(get_ram_free_mb)
    local vram_free=$(get_vram_free_mb)
    if [ "$ram_free" -lt "$RAM_SAFETY_MB" ]; then
        echo "[DANGER] RAM critically low: ${ram_free}MB free (need ${RAM_SAFETY_MB}MB safety)"
        return 1
    fi
    if [ "$vram_free" -lt "$VRAM_SAFETY_MB" ]; then
        echo "[DANGER] VRAM critically low: ${vram_free}MB free (need ${VRAM_SAFETY_MB}MB safety)"
        return 1
    fi
    return 0
}

print_status() {
    local ram_free=$(get_ram_free_mb)
    local ram_used=$(get_ram_used_mb)
    local vram_free=$(get_vram_free_mb)
    local vram_used=$(get_vram_used_mb)
    local gpu_util=$(get_gpu_util)
    printf "[%s] RAM: %5dMB used / %5dMB free | VRAM: %5dMB used / %5dMB free | GPU: %3s%%\n" \
        "$(date +%H:%M:%S)" "$ram_used" "$ram_free" "$vram_used" "$vram_free" "$gpu_util"
}

if [ "$1" = "--watch" ]; then
    echo "=== NVComp Resource Monitor ==="
    echo "Safety: RAM=${RAM_SAFETY_MB}MB, VRAM=${VRAM_SAFETY_MB}MB"
    while true; do
        print_status
        sleep "$CHECK_INTERVAL"
    done
fi

if [ "$1" = "--check" ] || [ -z "$1" ]; then
    print_status
    check_safe
    exit $?
fi

if [ "$1" = "--safe-run" ]; then
    shift
    if ! check_safe; then
        echo "[ABORT] Insufficient resources. Waiting..."
        while ! check_safe; do sleep 5; done
        echo "[OK] Resources available, proceeding."
    fi
    exec "$@"
fi
