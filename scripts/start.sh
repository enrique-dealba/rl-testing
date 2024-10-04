#!/bin/bash
set -e

# Function to check if NVIDIA GPU support is available
has_nvidia_support() {
    [ -e /dev/nvidia0 ] || [ -e /proc/driver/nvidia/version ]
}

# Check NVIDIA support
if has_nvidia_support; then
    echo "SUCCESS: NVIDIA GPU support detected."
else
    echo "WARNING: No NVIDIA GPU support detected."
fi

# Add the current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/app"

# Determine the command to run
case "$1" in
    train)
        shift
        python /app/src/main.py "$@"
        ;;
    diagnostics)
        python /app/scripts/run_diagnostics.py
        ;;
    streamlit)
        shift
        streamlit run /app/streamlit_app/streamlit_app.py --server.port 8888 --server.address 0.0.0.0
        ;;
    *)
        echo "Usage: ./scripts/start.sh [train|diagnostics|streamlit] [options]"
        exit 1
        ;;
esac
