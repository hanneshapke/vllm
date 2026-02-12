FROM nvcr.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /root/dataiku
RUN uv venv --python 3.13.12 .venv 
RUN . .venv/bin/activate && python -m ensurepip && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN curl  -LO https://storage.googleapis.com/dioscuri-pollux/vllm-0.1.dev13816%2Bgdcc9c93c3.cu130-cp313-cp313-linux_x86_64.whl && . .venv/bin/activate && python -m pip install ./vllm-0.1.dev13816%2Bgdcc9c93c3.cu130-cp313-cp313-linux_x86_64.whl && rm vllm-0.1.dev13816%2Bgdcc9c93c3.cu130-cp313-cp313-linux_x86_64.whl
RUN curl -LO https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
ENV PATH="/root/dataiku/.venv/bin:$PATH"
CMD ["vllm", "serve", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "4096", \
     "--trust-remote-code", \
     "--extract-activation-layers", "20", \
     "--enable-auto-tool-choice", \
     "--tool-call-parser", "qwen3_coder", \
     "--reasoning-parser-plugin", "nano_v3_reasoning_parser.py", \
     "--reasoning-parser", "nano_v3"]
