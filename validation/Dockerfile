# syntax=docker/dockerfile:1

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

ARG TORCH_VERSION=2.13.0
ARG TORCH_CUDA_VARIANT=cu130
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu130

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Runtime downloads, model caches, temporary files, and generated evaluation
# artifacts all belong to the mounted /data volume, never to the image layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOME=/data/home \
    TMPDIR=/data/tmp \
    XDG_CACHE_HOME=/data/cache/xdg \
    HF_HUB_CACHE=/data/cache/huggingface/hub \
    HF_DATASETS_CACHE=/data/cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/data/cache/huggingface/transformers \
    TORCH_HOME=/data/cache/torch \
    UV_CACHE_DIR=/data/cache/uv \
    PIP_CACHE_DIR=/data/cache/pip \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH="/app/model-tools/.venv/bin:/usr/local/bin:${PATH}"

WORKDIR /app/model-tools
COPY model-tools/pyproject.toml model-tools/uv.lock ./
COPY model-tools/src ./src
# HACK: The host/dev uv lock intentionally remains CPU-only; the evaluator
# container installs the same Torch release's CUDA build to avoid burdening
# non-evaluator environments with CUDA dependencies.
RUN uv sync --locked --no-dev \
    && uv pip uninstall --python /app/model-tools/.venv/bin/python torch \
    && uv pip install \
        --python /app/model-tools/.venv/bin/python \
        --index-url "${TORCH_CUDA_INDEX_URL}" \
        "torch==${TORCH_VERSION}+${TORCH_CUDA_VARIANT}" \
    && rm -rf /root/.cache/uv /data/cache/uv /data/home /data/tmp

# Checked-in code and model metadata are image inputs. Runtime acquisition
# writes the corpus, source manifests, annotations, scores, and model bundle
# only under /data.
COPY model-tools/model-sources ./model-sources

WORKDIR /app
ENTRYPOINT ["model-tools-evaluation"]
CMD ["all"]
