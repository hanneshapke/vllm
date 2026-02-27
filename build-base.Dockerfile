FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /root/dataiku
RUN uv venv --python 3.13.12 .venv
RUN . .venv/bin/activate && python -m ensurepip && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN git clone --branch extract-activations --single-branch https://github.com/hanneshapke/vllm.git
RUN . .venv/bin/activate && python -m pip install -r vllm/requirements/build.txt -r vllm/requirements/cuda.txt
# Build the image
ENV PATH="/root/dataiku/.venv/bin:$PATH"
CMD ["bash"]
