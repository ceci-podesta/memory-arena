#!/bin/bash
echo "==================================================="
echo "STATUS D.AR — $(date)"
echo "==================================================="
echo ""
echo "--- Ultimas 20 lineas del log mas reciente ---"
LATEST_LOG=$(ls -t results/d_ar_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Log: $LATEST_LOG"
    tail -20 "$LATEST_LOG"
else
    echo "  (no se encontro log de d_ar)"
fi
echo ""
echo "--- Responses de Accurate_Retrieval (recientes primero) ---"
ls -lt results/responses/*Accurate_Retrieval*.jsonl 2>/dev/null | head -8
echo ""
echo "--- Lineas por archivo (= records escritos) ---"
wc -l results/responses/*Accurate_Retrieval*.jsonl 2>/dev/null
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "--- Proceso D.AR vivo ---"
ps aux | grep run_d_ar_a_mem | grep -v grep
