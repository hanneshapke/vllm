FROM nvcr.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.13.12
ARG CUDA_VARIANT=cu130
ARG VLLM_WHEEL=https://storage.googleapis.com/dioscuri-pollux/vllm-0.1.dev13816%2Bgdcc9c93c3.cu130-cp313-cp313-linux_x86_64.whl

WORKDIR /root/dataiku
RUN uv venv --python ${PYTHON_VERSION} .venv
RUN . .venv/bin/activate && python -m ensurepip && \
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VARIANT}
RUN curl -LO ${VLLM_WHEEL} && \
    . .venv/bin/activate && \
    python -m pip install ./$(basename "${VLLM_WHEEL}") && \
    rm -f ./$(basename "${VLLM_WHEEL}")

ARG MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
RUN curl -LO https://huggingface.co/${MODEL}/resolve/main/nano_v3_reasoning_parser.py

ENV PATH="/root/dataiku/.venv/bin:$PATH"
ENV VLLM_MODEL=${MODEL}
ENV VLLM_TP_SIZE=1
ENV VLLM_MAX_MODEL_LEN=4096
ENV VLLM_EXTRACT_ACTIVATION_LAYERS=20

CMD vllm serve ${VLLM_MODEL} \
    --tensor-parallel-size ${VLLM_TP_SIZE} \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --trust-remote-code \
    --extract-activation-layers ${VLLM_EXTRACT_ACTIVATION_LAYERS} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3
