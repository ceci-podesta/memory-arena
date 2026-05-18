#!/bin/bash
LRU_PID="${1:-}"
if [ -z "$LRU_PID" ]; then
    echo "Uso: $0 <PID-del-d_lru-actual>"
    exit 1
fi

LOG="results/d_continuation_a_mem_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "=== Esperando a que termine D.LRU (PID=$LRU_PID): $(date) ===" | tee -a "$LOG"
while kill -0 "$LRU_PID" 2>/dev/null; do
    sleep 60
done

echo "" | tee -a "$LOG"
echo "=== D.LRU terminó. Arrancando CR -> TTL -> AR: $(date) ===" | tee -a "$LOG"

for split in cr ttl ar; do
    echo "" | tee -a "$LOG"
    echo "###############################################" | tee -a "$LOG"
    echo "### D.$split — INICIO $(date)" | tee -a "$LOG"
    echo "###############################################" | tee -a "$LOG"
    uv run python "scripts/run_d_${split}_a_mem.py" 2>&1 | tee -a "$LOG"
    code=$?
    echo "" | tee -a "$LOG"
    echo "### D.$split — FIN $(date)  exit_code=$code" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== TODOS LOS SPLITS TERMINARON: $(date) ===" | tee -a "$LOG"
