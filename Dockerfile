# Standalone build: clones talkie-core for sdk + modules.api, overlays this repo as modules/rag.
# Image matches CLI/compose: ghcr.io/talkie-assistant/talkie-module-rag, CMD python -m modules.rag.server
FROM alpine/git as talkie-core
ARG TALKIE_CORE_REF=main
WORKDIR /src
RUN git clone --depth 1 --branch "${TALKIE_CORE_REF}" https://github.com/talkie-assistant/talkie-core.git .

FROM python:3.11-slim
WORKDIR /app
COPY --from=talkie-core /src/sdk ./sdk
COPY --from=talkie-core /src/modules/__init__.py /src/modules/discovery.py ./modules/
COPY --from=talkie-core /src/modules/api ./modules/api
COPY . ./modules/rag
RUN pip install --no-cache-dir -r modules/rag/requirements.txt
RUN mkdir -p data

CMD ["python", "-m", "modules.rag.server", "--host", "0.0.0.0", "--port", "8002"]
