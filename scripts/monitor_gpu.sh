#!/usr/bin/env bash
# AMD MI300X GPU telemetry sampler.
#
# Polls `rocm-smi` every 200ms and writes a CSV row with:
#   timestamp,gpu_temp_c,gpu_util_pct,vram_mb,power_w,sclk_mhz
#
# On SIGINT/SIGTERM, prints summary stats (mean/max util, peak temp, peak
# VRAM, peak power) to stderr.
#
# Usage:
#   ./scripts/monitor_gpu.sh                          # default /tmp/gpu_telemetry.csv
#   ./scripts/monitor_gpu.sh /path/to/out.csv

set -u

OUT_FILE="${1:-/tmp/gpu_telemetry.csv}"
INTERVAL_S="0.2"

if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "ERROR: rocm-smi not found on PATH" >&2
    exit 1
fi

echo "timestamp,gpu_temp_c,gpu_util_pct,vram_mb,power_w,sclk_mhz" > "$OUT_FILE"

# Aggregates for the summary on exit.
N=0
SUM_UTIL=0
MAX_UTIL=0
MAX_TEMP=0
MAX_VRAM=0
MAX_POWER=0

cleanup() {
    if [ "$N" -gt 0 ]; then
        # Use awk for float division; bash arithmetic is integer-only.
        MEAN_UTIL=$(awk -v s="$SUM_UTIL" -v n="$N" 'BEGIN{printf "%.2f", s/n}')
    else
        MEAN_UTIL="0.00"
    fi
    {
        echo "--- gpu telemetry summary ---"
        echo "samples       : $N"
        echo "mean util %   : $MEAN_UTIL"
        echo "max  util %   : $MAX_UTIL"
        echo "peak temp C   : $MAX_TEMP"
        echo "peak vram MB  : $MAX_VRAM"
        echo "peak power W  : $MAX_POWER"
        echo "csv file      : $OUT_FILE"
    } >&2
    exit 0
}
trap cleanup INT TERM

# Extract the first numeric value (int or float) from a chunk of rocm-smi output
# matching a label substring. Returns 0 if not found.
extract() {
    local label="$1" blob="$2"
    echo "$blob" \
        | grep -i "$label" \
        | head -n 1 \
        | grep -oE '[0-9]+(\.[0-9]+)?' \
        | head -n 1
}

while true; do
    TS=$(date +%s.%N)
    BLOB=$(rocm-smi --csv --showuse --showtemp --showmemuse --showpower \
                    --showclocks --showmeminfo vram 2>/dev/null || true)

    TEMP=$(extract "Temperature"      "$BLOB"); TEMP=${TEMP:-0}
    UTIL=$(extract "GPU use"          "$BLOB"); UTIL=${UTIL:-0}
    VRAM=$(extract "VRAM Total Used"  "$BLOB"); VRAM=${VRAM:-0}
    PWR=$(extract  "Power"            "$BLOB"); PWR=${PWR:-0}
    SCLK=$(extract "sclk"             "$BLOB"); SCLK=${SCLK:-0}

    echo "$TS,$TEMP,$UTIL,$VRAM,$PWR,$SCLK" >> "$OUT_FILE"

    N=$((N + 1))
    SUM_UTIL=$(awk -v a="$SUM_UTIL" -v b="$UTIL" 'BEGIN{printf "%.4f", a+b}')
    awk -v cur="$UTIL"  -v max="$MAX_UTIL"  'BEGIN{exit !(cur>max)}' && MAX_UTIL="$UTIL"
    awk -v cur="$TEMP"  -v max="$MAX_TEMP"  'BEGIN{exit !(cur>max)}' && MAX_TEMP="$TEMP"
    awk -v cur="$VRAM"  -v max="$MAX_VRAM"  'BEGIN{exit !(cur>max)}' && MAX_VRAM="$VRAM"
    awk -v cur="$PWR"   -v max="$MAX_POWER" 'BEGIN{exit !(cur>max)}' && MAX_POWER="$PWR"

    sleep "$INTERVAL_S"
done
