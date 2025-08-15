# ============ Stage 1: build whisper.cpp ============
FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS whisper_builder
ARG MODEL=tiny  # tiny or tiny.en
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates pkg-config \
    libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git .
RUN cmake -S . -B build \
      -DWHISPER_FFMPEG=ON \
      -DGGML_NATIVE=OFF \
      -DGGML_CPU_F16=OFF \
      -DGGML_CPU_DOTPROD=OFF
RUN cmake --build build -j"$(nproc)"
RUN cmake --install build --prefix /out/install

# модель
RUN set -eux; mkdir -p /out/models; \
    MF=$([ "$MODEL" = "tiny.en" ] && echo ggml-tiny.en.bin || echo ggml-tiny.bin); \
    curl -L -o "/out/models/${MF}" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MF}"; \
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

# заголовки/библиотеки whisper
COPY --from=whisper_builder /out/install/include /usr/local/include
COPY --from=whisper_builder /out/install/lib /usr/local/lib
ENV LIBRARY_PATH=/usr/local/lib

RUN g++ -O3 -std=c++17 -I/build -I/usr/local/include \
        server.cpp -o server \
        -L/usr/local/lib -lwhisper -lggml-base -lggml \
        -lavformat -lavcodec -lavutil -lswresample -lpthread -lm

# ============ Stage 3: runtime ============
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat59 libavcodec59 libavutil57 libswresample4 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# whisper shared libs
COPY --from=whisper_builder /out/install/lib/ /usr/local/lib/
RUN ldconfig
# модель
COPY --from=whisper_builder /out/models /app/models
RUN set -eux; MF="$(cat /app/models/MODEL_NAME)"; ln -sf "/app/models/${MF}" /app/models/model.bin
# сервер
COPY --from=server_builder /build/server /usr/local/bin/server

ENV WHISPER_MODEL=/app/models/model.bin \
    WHISPER_THREADS=1
EXPOSE 8081

RUN useradd -r -s /usr/sbin/nologin whisper && chown -R whisper:whisper /app
USER whisper

ENTRYPOINT ["/usr/local/bin/server"]
