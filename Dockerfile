# AgentReliabilityLab — everything runs offline against mock tools and the
# reference agents; the container makes no external calls at runtime.
#
#   docker build -t arl .
#   docker run --rm arl demo                      # weak vs guarded + comparison
#   docker run --rm arl eval suites/ci.yaml       # CI eval gate
#   docker run --rm arl pytest -q                 # platform self-tests (CPU only)
#   docker run --rm -p 8600:8600 arl serve --host 0.0.0.0
#
# Dev dependencies are installed so the self-test suite runs inside the image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependency layer (cached across builds).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Project source plus the scenario/suite data the CLI resolves relative to /app.
COPY src ./src
COPY scenarios ./scenarios
COPY suites ./suites
COPY examples ./examples
COPY tests ./tests
COPY README.md LICENSE PLAN.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["arl"]
CMD ["--help"]
