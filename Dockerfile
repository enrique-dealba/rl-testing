FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch and torchvision with CUDA 11.8 support
RUN pip uninstall torch -y
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install pettingzoo[mpe]==1.24.3

# Copy the source code into the container
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY streamlit_app/ /app/streamlit_app/

# Make start.sh executable
RUN chmod +x /app/scripts/start.sh

# Expose port 8888 (if needed for Jupyter or other services)
EXPOSE 8888

# Ensure log directory exists
RUN mkdir -p /app/logs

# Set the start.sh script as the entrypoint
ENTRYPOINT ["/app/scripts/start.sh"]

# Default to running diagnostics
CMD ["diagnostics"]
