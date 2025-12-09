# Energy-Constrained Tipping Cascades Research Environment
# Base image with CUDA support for GPU-accelerated simulations
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="Jason Holt"
LABEL project="tipping-cascades-research"
LABEL description="PyCascades research environment with GPU support"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libhdf5-dev \
    libnetcdf-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge default, includes mamba, no Anaconda ToS issues)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment with Python 3.9
RUN mamba create -n tipping-research python=3.9 -y && \
    mamba clean -afy

# Activate environment by default
ENV CONDA_DEFAULT_ENV=tipping-research
ENV PATH=$CONDA_DIR/envs/tipping-research/bin:$PATH

# Install scientific computing stack via mamba
# Pin dask/distributed versions for compatibility with workers
RUN mamba install -n tipping-research -c conda-forge \
    numpy=1.26.3 \
    scipy \
    matplotlib \
    cartopy \
    seaborn \
    netCDF4 \
    h5py \
    networkx \
    numba \
    pandas=2.1.4 \
    xarray \
    dask=2024.1.0 \
    distributed=2024.1.0 \
    cloudpickle=3.0.0 \
    lz4=4.3.3 \
    msgpack-python=1.0.7 \
    toolz=0.12.0 \
    tornado=6.3.3 \
    bokeh \
    ipykernel \
    jupyterlab \
    jupyter-server-proxy \
    ipywidgets \
    nodejs \
    -y && \
    conda clean -afy

# Install GPU-accelerated libraries (optional, graceful fallback)
RUN mamba install -n tipping-research -c conda-forge \
    cupy \
    -y || echo "CuPy installation skipped (no GPU or incompatible CUDA)"

# Install PyCascades-specific dependencies via pip
RUN pip install --no-cache-dir \
    sdeint \
    PyPDF2 \
    pyDOE \
    SALib \
    mlflow \
    redis \
    python-dotenv \
    tqdm \
    pytest \
    pytest-cov \
    black \
    isort \
    graphviz

# Install Dask extensions
RUN pip install --no-cache-dir \
    dask-kubernetes \
    dask-jobqueue

# Clone and install PyCascades and related packages
WORKDIR /opt/research
RUN git clone https://github.com/pik-copan/pycascades.git && \
    cd pycascades && pip install -e . && cd ..

RUN git clone https://github.com/pik-copan/pyunicorn.git && \
    cd pyunicorn && pip install -e . && cd .. || echo "pyunicorn install failed - may need additional deps"

# Note: pycopancore has complex dependencies, install separately if needed
# RUN git clone https://github.com/pik-copan/pycopancore.git && \
#     cd pycopancore && pip install -e . && cd ..

# Clone cryosphere analysis repository for reference
RUN git clone https://github.com/JonathanRosser/Cryosphere-tipping-elements.git || \
    echo "Cryosphere repo clone failed - continuing"

# Create working directories
RUN mkdir -p /workspace/notebooks /workspace/data /workspace/checkpoints /workspace/src

# Set up Jupyter configuration
RUN mkdir -p /root/.jupyter
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_server_config.py

# Register the kernel
RUN python -m ipykernel install --user --name tipping-research --display-name "Tipping Cascades (Python 3.9)"

# Environment variables for runtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Default working directory
WORKDIR /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pycascades; print('healthy')" || exit 1

# Default command (JupyterLab)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
