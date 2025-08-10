#!/bin/bash
# Docker Container Startup Script with Learning Initialization

echo "=========================================="
echo "  Trilingual AI - Container Startup"
echo "=========================================="

# Set environment variables
export PYTHONPATH=/app
export LEARNING_DATA_DIR=/app/learning_data
export ANALYTICS_DATA_DIR=/app/analytics_data

# Create necessary directories
mkdir -p $LEARNING_DATA_DIR
mkdir -p $ANALYTICS_DATA_DIR

# Check if this is the first run
FIRST_RUN_FLAG="$LEARNING_DATA_DIR/.first_run_complete"

if [ ! -f "$FIRST_RUN_FLAG" ]; then
    echo "🎯 First run detected - initializing learning features..."
    
    # Run essential Luo learning in background
    echo "📚 Learning essential Luo vocabulary from Glosbe..."
    python learn_luo_dictionary.py quick &
    
    # Create first run flag
    touch "$FIRST_RUN_FLAG"
    echo "✅ Learning initialization started"
else
    echo "✅ Learning features already initialized"
fi

# Start the main application
echo "🚀 Starting Enhanced Trilingual AI..."
exec "$@"
