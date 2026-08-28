#!/usr/bin/env bash
# Supervisor for stress_test.py: on this machine (confirmed this session: 7.3GB total RAM,
# ~2.5GB free with the pretrained model stack loaded), the process can die at the native
# level -- a SIGSEGV was observed shortly after model load, and separately a MemoryError
# during a large HTTP retry -- neither of which Python's try/except can catch, since the
# whole process is killed. A per-recording try/except inside stress_test.py (already there)
# only protects against exceptions Python itself can catch; it cannot protect against the
# interpreter being killed outright.
#
# This script tracks a fixed wall-clock DEADLINE (not a per-attempt budget) and keeps
# relaunching `stress_test.py --resume` -- which skips recordings already logged in
# results.jsonl/failures.jsonl -- until either the deadline passes or a run completes
# cleanly (summary.json's "finished": true, written unconditionally at the end of a normal
# run() -- whether it stopped because the time budget ran out, --max-recordings was hit, or
# the dataset was exhausted). Aborts early only if several consecutive attempts each die
# within seconds of starting, since that pattern means a persistent bug rather than a
# transient crash, and continuing would just burn the deadline in a crash loop.
#
# Usage:
#   HF_TOKEN=... ./scripts/run_stress_test_supervised.sh [hours] [output_dir]
# Defaults: hours=3.0, output_dir=outputs/stress_test

set -uo pipefail
cd "$(dirname "$0")/.."

HOURS="${1:-3.0}"
OUT_DIR="${2:-outputs/stress_test}"
PYTHON="./.venv/Scripts/python.exe"

END_EPOCH=$(( $(date +%s) + $(awk "BEGIN{printf \"%d\", $HOURS*3600}") ))
ATTEMPT=0
CONSECUTIVE_FAST_FAILURES=0
MAX_CONSECUTIVE_FAST_FAILURES=5
FAST_FAILURE_THRESHOLD_SECONDS=30

echo "Supervisor starting. Deadline: $(date -d "@$END_EPOCH" 2>/dev/null || date -r "$END_EPOCH")"

while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
    NOW=$(date +%s)
    REMAINING_SECONDS=$(( END_EPOCH - NOW ))
    REMAINING_HOURS=$(awk "BEGIN{printf \"%.4f\", $REMAINING_SECONDS/3600}")
    ATTEMPT=$((ATTEMPT + 1))

    echo ""
    echo "=== supervisor attempt $ATTEMPT -- ${REMAINING_HOURS}h remaining ==="
    ATTEMPT_START=$(date +%s)

    "$PYTHON" -u scripts/stress_test.py \
        --time-budget-hours "$REMAINING_HOURS" \
        --output-dir "$OUT_DIR" \
        --resume
    EXIT_CODE=$?

    ATTEMPT_DURATION=$(( $(date +%s) - ATTEMPT_START ))
    echo "=== attempt $ATTEMPT exited with code $EXIT_CODE after ${ATTEMPT_DURATION}s ==="

    if grep -q '"finished": true' "$OUT_DIR/summary.json" 2>/dev/null; then
        echo "=== run completed normally (summary.json finished=true). Supervisor stopping. ==="
        break
    fi

    if [ "$ATTEMPT_DURATION" -lt "$FAST_FAILURE_THRESHOLD_SECONDS" ]; then
        CONSECUTIVE_FAST_FAILURES=$((CONSECUTIVE_FAST_FAILURES + 1))
        echo "=== fast failure #$CONSECUTIVE_FAST_FAILURES (attempt died within ${FAST_FAILURE_THRESHOLD_SECONDS}s) ==="
        if [ "$CONSECUTIVE_FAST_FAILURES" -ge "$MAX_CONSECUTIVE_FAST_FAILURES" ]; then
            echo "=== $MAX_CONSECUTIVE_FAST_FAILURES consecutive fast failures -- this looks like a persistent bug, not a transient crash. Aborting supervisor. ==="
            exit 1
        fi
    else
        CONSECUTIVE_FAST_FAILURES=0
    fi

    sleep 5
done

echo ""
echo "=== supervisor loop ended (deadline reached or run finished) ==="
