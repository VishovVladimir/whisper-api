// server.cpp
#include <httplib.h>     // https://github.com/yhirose/cpp-httplib (header-only)
#include <whisper.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}

#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>

static std::mutex g_mutex;
static whisper_context* g_ctx = nullptr;

static std::string MODEL_PATH  = "/app/models/model.bin";
static int         N_THREADS   = []{
    if (const char* e = std::getenv("WHISPER_THREADS")) return std::max(1, std::atoi(e));
    return 1;
}();

static inline std::string json_escape(const std::string& s) {
    std::string o; o.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"':  o += "\\\""; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:   o += c;      break;
        }
    }
    return o;
}

// ---------- Decode from memory to mono 16k float ----------
static bool decode_to_pcm16k_f32(const uint8_t* data, size_t size, std::vector<float>& pcmf32) {
    AVFormatContext* fmt = avformat_alloc_context();
    if (!fmt) return false;

    struct MemCtx { const uint8_t* base; const uint8_t* p; size_t left; };
    MemCtx* mc = (MemCtx*)av_mallocz(sizeof(MemCtx));
    if (!mc) { avformat_free_context(fmt); return false; }

    mc->base = (const uint8_t*)av_malloc(size);
    if (!mc->base) { av_free(mc); avformat_free_context(fmt); return false; }
    memcpy((void*)mc->base, data, size);
    mc->p = mc->base;
    mc->left = size;

    const int IO_BUF_SZ = 4096;
    uint8_t* iobuf = (uint8_t*)av_malloc(IO_BUF_SZ);
    if (!iobuf) { av_free((void*)mc->base); av_free(mc); avformat_free_context(fmt); return false; }

    auto read_packet = [](void* opaque, uint8_t* dst, int dst_size)->int {
        auto* m = (MemCtx*)opaque;
        int n = std::min<int>(dst_size, (int)m->left);
        if (n <= 0) return AVERROR_EOF;
        memcpy(dst, m->p, n);
        m->p   += n;
        m->left -= n;
        return n;
    };

    AVIOContext* avio = avio_alloc_context(iobuf, IO_BUF_SZ, 0, mc, read_packet, nullptr, nullptr);
    if (!avio) {
        av_free(iobuf);
        av_free((void*)mc->base);
        av_free(mc);
        avformat_free_context(fmt);
        return false;
    }
    fmt->pb = avio;
    fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

    bool ok = false;

    if (avformat_open_input(&fmt, nullptr, nullptr, nullptr) == 0 &&
        avformat_find_stream_info(fmt, nullptr) == 0) {

        int astream = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (astream >= 0) {
            AVStream* st = fmt->streams[astream];
            const AVCodec* dec = avcodec_find_decoder(st->codecpar->codec_id);
            if (dec) {
                AVCodecContext* codec = avcodec_alloc_context3(dec);
                if (codec && avcodec_parameters_to_context(codec, st->codecpar) >= 0 &&
                    avcodec_open2(codec, dec, nullptr) >= 0) {

                    int in_rate = codec->sample_rate > 0 ? codec->sample_rate : 48000;
                    int in_ch   = codec->channels     > 0 ? codec->channels     : 2;
                    int64_t in_layout = codec->channel_layout ?
                        codec->channel_layout : av_get_default_channel_layout(in_ch);

                    SwrContext* swr = swr_alloc_set_opts(
                        nullptr,
                        av_get_default_channel_layout(1), AV_SAMPLE_FMT_FLT, 16000,
                        in_layout, codec->sample_fmt, in_rate,
                        0, nullptr);

                    if (swr && swr_init(swr) >= 0) {
                        AVPacket* pkt = av_packet_alloc();
                        AVFrame*  frm = av_frame_alloc();
                        if (pkt && frm) {
                            pcmf32.clear();

                            while (av_read_frame(fmt, pkt) >= 0) {
                                if (pkt->stream_index != astream) { av_packet_unref(pkt); continue; }
                                if (avcodec_send_packet(codec, pkt) < 0) { av_packet_unref(pkt); continue; }
                                av_packet_unref(pkt);

                                while (true) {
                                    int r = avcodec_receive_frame(codec, frm);
                                    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
                                    if (r < 0) break;

                                    const int out_needed = av_rescale_rnd(
                                        swr_get_delay(swr, in_rate) + frm->nb_samples,
                                        16000, in_rate, AV_ROUND_UP);

                                    std::vector<float> out((size_t)out_needed);
                                    uint8_t* out_data = (uint8_t*)out.data();

                                    int got = swr_convert(swr, &out_data, out_needed,
                                                          (const uint8_t**)frm->data, frm->nb_samples);
                                    if (got > 0) {
                                        out.resize((size_t)got);
                                        pcmf32.insert(pcmf32.end(), out.begin(), out.end());
                                    }
                                    av_frame_unref(frm);
                                }
                            }

                            ok = !pcmf32.empty();
                        }
                        if (pkt) av_packet_free(&pkt);
                        if (frm) av_frame_free(&frm);
                        swr_free(&swr);
                    }

                    avcodec_free_context(&codec);
                } else if (codec) {
                    avcodec_free_context(&codec);
                }
            }
        }
    }

    // Порядок освобождения: сначала формат, затем AVIO, потом наш буфер и контекст
    avformat_close_input(&fmt);
    avio_context_free(&avio);        // освободит iobuf
    av_free((void*)mc->base);        // освобождаем ИМЕННО base, а не p!
    av_free(mc);

    return ok;
}

int main() {
    if (const char* e = std::getenv("WHISPER_MODEL")) MODEL_PATH = e;

    // Инициализируем whisper-модель один раз
    {
        whisper_context_params cparams = whisper_context_default_params();
        g_ctx = whisper_init_from_file_with_params(MODEL_PATH.c_str(), cparams);
        if (!g_ctx) {
            fprintf(stderr, "ERR: whisper_init failed (model=%s)\n", MODEL_PATH.c_str());
            return 1;
        }
    }

    httplib::Server app;
    app.set_read_timeout  (600, 0);
    app.set_write_timeout (600, 0);
    app.set_idle_interval (0, 500000);
    app.set_payload_max_length(50 * 1024 * 1024);

    app.Get("/healthz", [](const httplib::Request&, httplib::Response& res){
        res.set_content(R"({"ok":true})", "application/json");
    });

    app.Post("/inference", [](const httplib::Request& req, httplib::Response& res){
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"empty body"})", "application/json");
            return;
        }

        std::string language = req.has_param("language") ? req.get_param_value("language") : "ru";

        std::vector<float> pcmf32;
        if (!decode_to_pcm16k_f32((const uint8_t*)req.body.data(), req.body.size(), pcmf32)) {
            res.status = 400;
            res.set_content(R"({"error":"decode failed"})", "application/json");
            return;
        }

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_progress   = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.translate        = false;
        wparams.no_timestamps    = true;
        wparams.n_threads        = N_THREADS;
        wparams.language         = language.c_str();
        wparams.token_timestamps = false;
        wparams.temperature      = 0.0f;
        wparams.max_initial_ts   = 0.0f;

        int rc = 0;
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            rc = whisper_full(g_ctx, wparams, pcmf32.data(), (int)pcmf32.size());
        }
        if (rc != 0) {
            res.status = 500;
            res.set_content(R"({"error":"whisper_full failed"})", "application/json");
            return;
        }

        std::ostringstream oss;
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            int n = whisper_full_n_segments(g_ctx);
            for (int i = 0; i < n; ++i) {
                oss << whisper_full_get_segment_text(g_ctx, i);
            }
        }
        std::string text = oss.str();

        std::ostringstream js;
        js << R"({"text":")" << json_escape(text) << R"("})";
        res.set_content(js.str(), "application/json");
        res.status = 200;
    });

    const char* host = "0.0.0.0";
    int port = 8081;
    fprintf(stderr, "Listening on %s:%d, threads=%d, model=%s\n",
            host, port, N_THREADS, MODEL_PATH.c_str());
    app.listen(host, port);

    if (g_ctx) { whisper_free(g_ctx); g_ctx = nullptr; }
    return 0;
}
