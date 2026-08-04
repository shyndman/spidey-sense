# syntax=docker/dockerfile:1

# Keep JavaScript tooling in a separate build stage so the runtime image can
# combine the pinned Node 22 runtime with the Python 3.13 base image.
FROM node:22-bookworm-slim AS extension-dependencies

WORKDIR /build/extension
RUN npm install --global pnpm@10.34.1
COPY extension/package.json extension/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile --ignore-scripts

FROM python:3.13-slim

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
    COREPACK_HOME=/data/cache/corepack \
    NODE_COMPILE_CACHE=/data/cache/node \
    NPM_CONFIG_CACHE=/data/cache/npm \
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
    CUDA_VISIBLE_DEVICES="" \
    PATH="/app/model-tools/.venv/bin:/usr/local/bin:${PATH}"

# The two official images share the Debian/glibc ABI; copying /usr/local
# supplies Node 22, npm, and pnpm without replacing Python 3.13.
COPY --from=extension-dependencies /usr/local /usr/local
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

WORKDIR /app/model-tools
COPY model-tools/pyproject.toml model-tools/uv.lock ./
COPY model-tools/src ./src
RUN uv sync --locked --no-dev \
    && rm -rf /root/.cache/uv /data/cache/uv /data/home /data/tmp

# Checked-in code and model metadata are image inputs. Runtime acquisition
# writes the corpus, source manifests, annotations, scores, and model bundle
# only under /data.
COPY model-tools/model-source.toml ./
COPY extension/package.json extension/pnpm-lock.yaml /app/extension/
COPY --from=extension-dependencies /build/extension/node_modules /app/extension/node_modules
COPY extension/src /app/extension/src
COPY extension/scripts /app/extension/scripts

WORKDIR /app
ENTRYPOINT ["model-tools-evaluation"]
CMD ["all"]
