FROM python:3.12-slim

WORKDIR /app

# Install PyTorch (CUDA for GPU)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
RUN pip install --no-cache-dir \
    numpy \
    pyyaml \
    tensorboard

# Copy project code
COPY pacman/ /app/pacman/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/

# Create directories for checkpoints and data
RUN mkdir -p /app/runs
