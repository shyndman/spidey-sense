FROM python:3.13-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/
WORKDIR /app

COPY model-tools/pyproject.toml model-tools/uv.lock ./
COPY model-tools/src ./src
RUN uv sync --locked --no-dev --no-editable \
    && rm -rf /root/.cache/uv

USER 65532:65532
ENTRYPOINT ["/app/.venv/bin/model-tools"]
CMD ["--version"]
