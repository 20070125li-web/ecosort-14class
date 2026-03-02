#!/bin/bash
# EcoSort GPU Training Script - Complete Logging
# Automated GPU training with comprehensive logging and monitoring

set -e

PROJECT_ROOT="/public/home/zhw/cptac/projects/ecosort"
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🚀 EcoSort GPU Training - Full Logging               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configure environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecosort

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
ERROR_LOG="$LOG_DIR/training_${TIMESTAMP}_error.log"

echo "📁 Logs will be saved to: $LOG_FILE"
echo "📁 Error log: $ERROR_LOG"
echo ""

# Validate GPU availability
echo "🔍 Checking GPU Status..."
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA not available - falling back to CPU')
    exit(1)
" || {
    echo "❌ GPU unavailable - please install CUDA-enabled PyTorch first"
    exit 1
}

echo ""
echo "✓ GPU Validation Complete"
echo ""

# Training configuration
CONFIG_FILE="$PROJECT_ROOT/configs/baseline_resnet50.yaml"
DATA_ROOT="$PROJECT_ROOT/data/raw"
EXPERIMENT_NAME="gpu_training"

# Launch training process
echo "═══════════════════════════════════════════════════════════"
echo "🚀 Starting GPU-Accelerated Training"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Run training in background with nohup and full logging
nohup python experiments/train_baseline.py \
    --config "$CONFIG_FILE" \
    --data-root "$DATA_ROOT" \
    --exp-name "$EXPERIMENT_NAME" \
    --no-wandb \
    > "$LOG_FILE" 2> "$ERROR_LOG" &

TRAIN_PID=$!

echo "✓ Training Process Started"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📊 Training Information"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Process PID: $TRAIN_PID"
echo "Config File: $CONFIG_FILE"
echo "Data Directory: $DATA_ROOT"
echo "Experiment Name: $EXPERIMENT_NAME"
echo ""
echo "📁 Log Files:"
echo "  Standard Output: $LOG_FILE"
echo "  Error Output: $ERROR_LOG"
echo ""
echo "📁 Model Output:"
echo "  Output Directory: checkpoints/$EXPERIMENT_NAME/"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📈 Real-Time Monitoring Commands"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. View training logs (real-time):"
echo "   tail -f $LOG_FILE"
echo ""
echo "2. View error logs:"
echo "   tail -f $ERROR_LOG"
echo ""
echo "3. View latest output (last 50 lines):"
echo "   tail -50 $LOG_FILE"
echo ""
echo "4. Check process status:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "5. Monitor GPU utilization:"
echo "   nvidia-smi"
echo ""
echo "6. Track generated files:"
echo "   watch -n 5 'ls -lht checkpoints/$EXPERIMENT_NAME/'"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "⏱️  Estimated Training Time (GPU-Accelerated)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "With NVIDIA L20 GPUs (3 devices):"
echo "  • Per Epoch: ~1-2 minutes (vs 6-8 minutes on CPU)"
echo "  • Full Training (20 epochs): ~20-40 minutes"
echo "  • Estimated Completion: $(date '+%H:%M' -d '+30 minutes')"
echo ""
echo "⚡ Speedup Factor: ~5-10x faster than CPU"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Save PID for process management
echo $TRAIN_PID > "$LOG_DIR/training.pid"
echo "PID saved to: $LOG_DIR/training.pid"

# Wait for initial output generation
echo "Waiting 5 seconds for initial output..."
sleep 5

if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "========== Initial Training Output =========="
    head -50 "$LOG_FILE"
    echo ""
fi

echo ""
echo "✅ Training running in background!"
echo ""
echo "💡 Tip: Use 'tail -f $LOG_FILE' to monitor training progress in real-time"
