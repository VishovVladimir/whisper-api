# ============ Stage 1: build whisper.cpp ============
FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS whisper_builder
ARG MODEL=tiny          # tiny, tiny.en, base, base.en, small, small.en
ARG QTYPE=q5_1          # q5_1 (default), q5_0, q4_1, f16
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates pkg-config \
    libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git .
# Enable fast CPU paths (reduces wall-time while keeping 1 CPU thread during inference)
RUN cmake -S . -B build \
      -DWHISPER_FFMPEG=ON \
      -DGGML_NATIVE=ON \
      -DGGML_CPU_F16=ON \
      -DGGML_CPU_DOTPROD=ON
RUN cmake --build build -j"$(nproc)"
RUN cmake --install build --prefix /out/install

# Model: tiny multilingual quantized by default
RUN set -eux; mkdir -p /out/models; \
    MF="ggml-${MODEL}-${QTYPE}.bin"; \
    if ! curl -fsSL -o "/out/models/${MF}" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MF}"; then \
        echo "Requested quantized model not found, falling back to f16"; \
        MF="ggml-${MODEL}.bin"; \
        curl -fsSL -o "/out/models/${MF}" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MF}"; \
    fi; \
    printf "%s" "$MF" > /out/models/MODEL_NAME; \
    ln -sf "/out/models/${MF}" /out/models/model.bin

# ============ Stage 2: build our C++ server ============
FROM debian:bookworm-slim AS server_builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make curl ca-certificates pkg-config \
    libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
ADD https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h /build/httplib.h
COPY server.cpp /build/server.cpp

# whisper headers/libs
COPY --from=whisper_builder /out/install/include /usr/local/include
COPY --from=whisper_builder /out/install/lib /usr/local/lib
ENV LIBRARY_PATH=/usr/local/lib

RUN g++ -O3 -DNDEBUG -std=c++17 -fno-omit-frame-pointer -I/build -I/usr/local/include \
        server.cpp -o server \
        -L/usr/local/lib -lwhisper -lggml-base -lggml \
        -lavformat -lavcodec -lavutil -lswresample -lpthread -lm \
    && strip /build/server

# ============ Stage 3: runtime ============
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat59 libavcodec59 libavutil57 libswresample4 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# whisper shared libs
COPY --from=whisper_builder /out/install/lib/ /usr/local/lib/
RUN ldconfig
# model
COPY --from=whisper_builder /out/models /app/models
RUN set -eux; MF="$(cat /app/models/MODEL_NAME)"; ln -sf "/app/models/${MF}" /app/models/model.bin
# server
COPY --from=server_builder /build/server /usr/local/bin/server

ENV WHISPER_MODEL=/app/models/model.bin \
    WHISPER_THREADS=1 \
    MAX_AUDIO_SEC=120 \
    OMP_NUM_THREADS=1
EXPOSE 8081

# Non-root
RUN useradd -r -s /usr/sbin/nologin whisper && chown -R whisper:whisper /app
USER whisper

# Keep container responsive under host load
ENTRYPOINT ["nice","-n","10","/usr/local/bin/server"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -fsS http://127.0.0.1:8081/healthz || exit 1
