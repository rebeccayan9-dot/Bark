/**
 * BarkSense — on-device dog vocalization classifier
 * XIAO ESP32S3 Sense | DS-CNN α=1.0 INT8 | TFLite Micro
 *
 * Pipeline (time-multiplexed, single-threaded):
 *   PDM mic (I2S0) → 1-s buffer → on-chip log-mel(40) + Δ + ΔΔ = 120×61 int8
 *   → TFLite Micro → argmax → per-class response
 *
 * Class responses:
 *   bark    → ntfy.sh POST + LED blink   (60 s cooldown)
 *   growl   → play /owner_voice.wav      (60 s cooldown) + LED solid
 *   grunt   → append to /events.log
 *   ambient → no-op
 *
 * Build flags:
 *   -DFEATURE_TEST   replaces the loop with a one-shot fixture check against
 *                    the Python-generated tensor in test_data.h.
 *
 * Stubs (search "STUB:"):
 *   - play_wav()   logs only; real impl needs RIFF parse + i2s_write loop
 *   - led_color()  single-LED placeholder for the RGB-LED responses in spec
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <LittleFS.h>
#include <driver/i2s.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "config.h"
#include "secrets.h"
#include "model_data.h"
#include "norm_params.h"
#include "mel_filterbank.h"
#include "delta_kernels.h"

#ifdef FEATURE_TEST
#include "test_data.h"
#endif

// ═════════════════════════════════════════════════════════════════════════════
// Buffers
// ═════════════════════════════════════════════════════════════════════════════

// Tensor arena in PSRAM; SRAM stays free for stack/WiFi/DMA.
// Heap-allocated because EXT_RAM_ATTR doesn't reliably place BSS arrays in
// PSRAM on Arduino-ESP32 v2.x.
static uint8_t* s_tensor_arena = nullptr;

// 1-second mic capture (32 KB in SRAM).
static int16_t s_audio[kClipSamples];

// Feature work buffers — laid out so each row is contiguous in time.
// Total ≈ 30 KB SRAM, no heap.
static float s_logmel[kMelBins][kNFrames];
static float s_delta1[kMelBins][kNFrames];
static float s_delta2[kMelBins][kNFrames];

// FFT scratch (interleaved Re/Im) + one-sided power spectrum.
static float s_fft_buf[kNFft * 2];
static float s_power[kFftBins];

// Hann window precomputed once at boot.
static float s_hann[kNFft];

// TFLite Micro handles.
static const tflite::Model*      s_model       = nullptr;
static tflite::MicroInterpreter* s_interpreter = nullptr;
static TfLiteTensor*             s_input       = nullptr;
static TfLiteTensor*             s_output      = nullptr;

// Per-class cooldowns.
static uint32_t s_last_notify_ms = 0;
static uint32_t s_last_play_ms   = 0;

// ═════════════════════════════════════════════════════════════════════════════
// Forward declarations
// ═════════════════════════════════════════════════════════════════════════════

static void init_serial();
static void init_led();
static void init_littlefs();
static void init_wifi();
static void init_i2s_mic();
static void init_i2s_spk();
static void init_tflite();
static void init_hann();

static bool capture_audio();
static float clip_rms(const int16_t* x, int n);
static void fft_inplace(float* buf, int n);
static void extract_features(const int16_t* samples, int8_t* out);
static int  run_inference();

static void dispatch(int class_id);
static void handle_dog_sound(const char* class_name, bool notify);

static bool play_wav(const char* path);
static bool play_test_tone(uint32_t freq_hz, uint32_t duration_ms);
static bool post_ntfy(const char* title, const char* body);
static void log_event(const char* class_name);

static void led_set(bool on);
static void led_color(uint8_t r, uint8_t g, uint8_t b, uint32_t hold_ms);  // STUB
static void led_blink(int times, uint32_t period_ms);

#ifdef FEATURE_TEST
static void run_feature_test();
#endif

// ═════════════════════════════════════════════════════════════════════════════
// Arduino entry points
// ═════════════════════════════════════════════════════════════════════════════

void setup() {
    init_serial();
    Serial.println("[BarkSense] boot");

    init_led();
    init_littlefs();
    init_hann();
    init_tflite();        // before mic: fail fast if model won't load
    init_i2s_mic();
    init_i2s_spk();

#if defined(FEATURE_TEST)
    run_feature_test();
    Serial.println("[test] halt — power-cycle to rerun");
    for (;;) { delay(1000); }
#elif defined(AUDIO_TEST)
    Serial.printf("[audio_test] playing %s\n", kOwnerVoicePath);
    bool ok = play_wav(kOwnerVoicePath);
    if (!ok) {
        Serial.println("[audio_test] WAV failed; trying generated test tone");
        ok = play_test_tone(440, 1200);
    }
    Serial.printf("[audio_test] %s — halt, power-cycle to rerun\n",
                  ok ? "PASS" : "FAIL");
    for (;;) { delay(1000); }
#else
    init_wifi();          // last; non-fatal if it fails
    Serial.printf("[ready] arena %u / %u bytes\n",
                  s_interpreter->arena_used_bytes(), kTensorArenaBytes);
#endif
}

void loop() {
    if (!capture_audio()) { delay(100); return; }
    const int cls = run_inference();
    if (cls < 0) return;
    Serial.printf("[infer] %s\n", kClassNames[cls]);
    dispatch(cls);
}

// ═════════════════════════════════════════════════════════════════════════════
// Init
// ═════════════════════════════════════════════════════════════════════════════

static void init_serial() {
    Serial.begin(115200);
    // Native USB CDC blocks in write() when the TX buffer fills and the host
    // isn't draining — that stalls the whole loop (symptom: monitor prints a
    // line or two then freezes). Drop excess instead of blocking so the
    // inference loop always runs free.
    Serial.setTxTimeoutMs(0);
    for (uint32_t t0 = millis(); !Serial && millis() - t0 < 2000; ) {}
}

static void init_led() {
    pinMode(kLedPin, OUTPUT);
    led_set(false);
}

static void init_littlefs() {
    Serial.println(LittleFS.begin(/*formatOnFail=*/true)
                   ? "[fs] LittleFS mounted"
                   : "[fs] LittleFS mount FAILED");
}

static void init_wifi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(kWifiSsid, kWifiPass);
    Serial.print("[wifi] connecting");
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
        delay(250);
        Serial.print('.');
    }
    if (WiFi.status() == WL_CONNECTED)
        Serial.printf("\n[wifi] %s\n", WiFi.localIP().toString().c_str());
    else
        Serial.println("\n[wifi] not connected — bark notifications disabled");
}

static void init_i2s_mic() {
    i2s_config_t cfg = {};
    cfg.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
    cfg.sample_rate          = kSampleRate;
    cfg.bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT;
    cfg.channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_PCM_SHORT;
    cfg.intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count        = 8;
    cfg.dma_buf_len          = 512;
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr));

    i2s_pin_config_t pins = {};
    pins.bck_io_num   = I2S_PIN_NO_CHANGE;
    pins.ws_io_num    = kMicClkPin;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num  = kMicDataPin;
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
    i2s_zero_dma_buffer(I2S_NUM_0);
    Serial.println("[i2s0] PDM mic ready");
}

static void init_i2s_spk() {
    i2s_config_t cfg = {};
    cfg.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
    cfg.sample_rate          = kSampleRate;
    cfg.bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT;
    cfg.channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    cfg.intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count        = 8;
    cfg.dma_buf_len          = 512;
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_1, &cfg, 0, nullptr));

    i2s_pin_config_t pins = {};
    pins.bck_io_num   = kSpkBclkPin;
    pins.ws_io_num    = kSpkLrclkPin;
    pins.data_out_num = kSpkDinPin;
    pins.data_in_num  = I2S_PIN_NO_CHANGE;
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_1, &pins));
    i2s_zero_dma_buffer(I2S_NUM_1);
    Serial.println("[i2s1] MAX98357A speaker ready");
}

static void init_tflite() {
    s_tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_SPIRAM);
    if (!s_tensor_arena) {
        Serial.printf("[tflite] PSRAM alloc %d B failed — check BOARD_HAS_PSRAM\n",
                      kTensorArenaBytes);
        for (;;) { delay(1000); }
    }

    s_model = tflite::GetModel(g_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[tflite] schema mismatch: model=%u runtime=%u\n",
                      s_model->version(), TFLITE_SCHEMA_VERSION);
        for (;;) { delay(1000); }
    }

    // Ops actually used by the DS-CNN INT8 (same set across α — width only).
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();              // GlobalAveragePooling2D
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    // Older TFLM API (tanakamasayuki port) requires a real ErrorReporter;
    // SimpleMemoryAllocator::Create dereferences it unconditionally.
    static tflite::MicroErrorReporter reporter;
    static tflite::MicroInterpreter interp(
        s_model, resolver, s_tensor_arena, kTensorArenaBytes,
        &reporter, /*resources=*/nullptr, /*profiler=*/nullptr);
    s_interpreter = &interp;

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[tflite] AllocateTensors FAILED — raise kTensorArenaBytes");
        for (;;) { delay(1000); }
    }
    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    const TfLiteIntArray* d = s_input->dims;
    if (d->size != 4 || d->data[1] != kFeatDim || d->data[2] != kNFrames) {
        Serial.printf("[tflite] unexpected input shape (%d,%d,%d,%d) — expected (1,%d,%d,1)\n",
                      d->data[0], d->data[1], d->data[2], d->data[3], kFeatDim, kNFrames);
        for (;;) { delay(1000); }
    }
    Serial.printf("[tflite] input %dx%d int8 q=(scale=%.5f, zp=%d)\n",
                  kFeatDim, kNFrames, s_input->params.scale, s_input->params.zero_point);
}

static void init_hann() {
    for (int i = 0; i < kNFft; i++) {
        s_hann[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (kNFft - 1));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Audio capture
// ═════════════════════════════════════════════════════════════════════════════

static bool capture_audio() {
    // Drain any backlogged audio first. While the loop is blocked (~6 s owner-
    // voice playback + ~2 s ntfy POST per detection), the mic DMA keeps queuing
    // samples; without flushing, i2s_read would return that stale audio FIFO and
    // detections would fall progressively further behind real time (seconds →
    // minutes). Discard the backlog so each analyzed clip is current.
    size_t drained = 0;
    do {
        i2s_read(I2S_NUM_0, s_audio, sizeof(s_audio), &drained, 0);
    } while (drained == sizeof(s_audio));

    size_t got = 0;
    esp_err_t err = i2s_read(I2S_NUM_0, s_audio, sizeof(s_audio), &got, pdMS_TO_TICKS(2000));
    if (err != ESP_OK || got != sizeof(s_audio)) {
        Serial.printf("[i2s0] read err=%d got=%u\n", err, (unsigned)got);
        return false;
    }
    // Remove the PDM mic's large DC bias (~1300 here). Training WAVs are DC-free,
    // so a leftover offset dominates FFT bin 0 / power_to_db(ref=max) and makes
    // the model see a degenerate feature map (it then emits a near-constant class
    // on every live clip). Subtracting the mean realigns live audio with the
    // training distribution.
    int64_t sum = 0;
    for (int i = 0; i < kClipSamples; i++) sum += s_audio[i];
    const int32_t dc = (int32_t)(sum / kClipSamples);
    for (int i = 0; i < kClipSamples; i++) s_audio[i] = (int16_t)(s_audio[i] - dc);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// FFT — Cooley-Tukey radix-2 DIT, in-place, interleaved Re/Im
// 512-pt complex FFT runs in ~3 ms on ESP32-S3 @ 240 MHz. esp-dsp swap is a
// drop-in replacement here if/when the project moves to mixed Arduino+espidf.
// ═════════════════════════════════════════════════════════════════════════════

static void fft_inplace(float* buf, int n) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(buf[2*i],   buf[2*j]);
            std::swap(buf[2*i+1], buf[2*j+1]);
        }
    }
    // Butterfly passes
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * (float)M_PI / len;
        const float wRe = cosf(ang), wIm = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float uRe = 1.0f, uIm = 0.0f;
            for (int j = 0; j < (len >> 1); j++) {
                const int a = i + j, b = a + (len >> 1);
                const float tRe = uRe * buf[2*b]   - uIm * buf[2*b+1];
                const float tIm = uRe * buf[2*b+1] + uIm * buf[2*b];
                buf[2*b]    = buf[2*a]   - tRe;
                buf[2*b+1]  = buf[2*a+1] - tIm;
                buf[2*a]   += tRe;
                buf[2*a+1] += tIm;
                const float nu = uRe * wRe - uIm * wIm;
                uIm = uRe * wIm + uIm * wRe;
                uRe = nu;
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Feature extraction
//   1. Per frame: Hann window → FFT → power spectrum → mel filterbank → energy
//   2. power_to_db(ref=max, top_db=80)  — applied to the whole 40×61 grid
//   3. Δ = savgol(polyorder=1, deriv=1) along time axis, mode='interp'
//   4. ΔΔ = savgol(polyorder=2, deriv=2) along time axis, mode='interp'
//      (For polyorder == deriv the highest derivative is constant across the
//       polynomial fit, so boundary frames inherit the nearest interior value.)
//   5. Stack [logmel; Δ; ΔΔ] → 120 × 61
//   6. Z-score with kFeatMean[c] / kFeatStd[c]
//   7. Quantize int8: round(x / scale) + zero_point, clip [-128, 127]
//   8. Row-major: out[c * kNFrames + t]
// ═════════════════════════════════════════════════════════════════════════════

static void extract_features(const int16_t* samples, int8_t* out) {
    // (1) Per-frame STFT → mel energy. Store raw mel energy in s_logmel for now;
    // we need the whole grid before applying power_to_db with ref=max.
    for (int t = 0; t < kNFrames; t++) {
        const int start = t * kHopLength;
        for (int i = 0; i < kNFft; i++) {
            const float s = (float)samples[start + i] / 32768.0f;
            s_fft_buf[2 * i]     = s * s_hann[i];
            s_fft_buf[2 * i + 1] = 0.0f;
        }
        fft_inplace(s_fft_buf, kNFft);

        for (int k = 0; k < kFftBins; k++) {
            const float re = s_fft_buf[2 * k];
            const float im = s_fft_buf[2 * k + 1];
            s_power[k] = re * re + im * im;
        }
        for (int m = 0; m < kMelBins; m++) {
            float e = 0.0f;
            for (int k = 0; k < kFftBins; k++) e += kMelFb[m][k] * s_power[k];
            s_logmel[m][t] = e;
        }
    }

    // (2) librosa.power_to_db(ref=np.max, amin=1e-10, top_db=80):
    //     log_spec = 10*log10(max(amin, S)) − 10*log10(max(amin, max(S)))
    //     log_spec = max(log_spec, log_spec.max() − top_db)
    constexpr float kAmin  = 1e-10f;
    constexpr float kTopDb = 80.0f;

    float max_e = kAmin;
    for (int m = 0; m < kMelBins; m++)
        for (int t = 0; t < kNFrames; t++)
            if (s_logmel[m][t] > max_e) max_e = s_logmel[m][t];

    const float ref_db = 10.0f * log10f(max_e);   // max_e already ≥ kAmin
    float max_db = -1e30f;
    for (int m = 0; m < kMelBins; m++) {
        for (int t = 0; t < kNFrames; t++) {
            float e  = s_logmel[m][t];
            if (e < kAmin) e = kAmin;
            const float db = 10.0f * log10f(e) - ref_db;
            s_logmel[m][t] = db;
            if (db > max_db) max_db = db;
        }
    }
    const float floor_db = max_db - kTopDb;
    for (int m = 0; m < kMelBins; m++)
        for (int t = 0; t < kNFrames; t++)
            if (s_logmel[m][t] < floor_db) s_logmel[m][t] = floor_db;

    // (3) + (4) Savgol deltas along the time axis.
    auto savgol_along_t = [&](const float (*kernel),
                              const float (*src)[kNFrames],
                              float (*dst)[kNFrames]) {
        for (int m = 0; m < kMelBins; m++) {
            for (int t = kDeltaHalf; t < kNFrames - kDeltaHalf; t++) {
                float acc = 0.0f;
                for (int n = -kDeltaHalf; n <= kDeltaHalf; n++)
                    acc += kernel[n + kDeltaHalf] * src[m][t + n];
                dst[m][t] = acc;
            }
            for (int t = 0; t < kDeltaHalf; t++)
                dst[m][t] = dst[m][kDeltaHalf];
            for (int t = kNFrames - kDeltaHalf; t < kNFrames; t++)
                dst[m][t] = dst[m][kNFrames - kDeltaHalf - 1];
        }
    };
    savgol_along_t(kDelta1Kernel, s_logmel, s_delta1);
    savgol_along_t(kDelta2Kernel, s_logmel, s_delta2);

    // (5)-(8) Stack + z-score + quantize → row-major (c, t).
    const float scale = s_input->params.scale;
    const int   zp    = s_input->params.zero_point;

    auto put = [&](int c, int t, float v) {
        v = (v - kFeatMean[c]) / kFeatStd[c];
        int q = (int)roundf(v / scale) + zp;
        if (q < -128) q = -128;
        else if (q > 127) q = 127;
        out[c * kNFrames + t] = (int8_t)q;
    };

    for (int m = 0; m < kMelBins; m++) {
        for (int t = 0; t < kNFrames; t++) {
            put(m,                  t, s_logmel[m][t]);
            put(m + kMelBins,       t, s_delta1[m][t]);
            put(m + 2 * kMelBins,   t, s_delta2[m][t]);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Inference
// ═════════════════════════════════════════════════════════════════════════════

static float clip_rms(const int16_t* x, int n) {
    // DC-removed (AC) RMS. The PDM mic carries a large DC/bias component
    // (~1300-1500 here) that swamps the acoustic signal and barely moves with
    // sound; sqrt(mean(x^2)) would measure that bias, not loudness. Subtract the
    // mean first so this reflects actual acoustic energy.
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += (double)x[i];
    mean /= n;
    double acc = 0.0;
    for (int i = 0; i < n; i++) { const double d = (double)x[i] - mean; acc += d * d; }
    return (float)sqrt(acc / n);
}

static int run_inference() {
    // Loudness gate FIRST — skip inference entirely on quiet clips. The model
    // never saw true silence in training, and per-clip power_to_db(ref=max)
    // turns a near-silent buffer into a degenerate feature map; running Invoke on
    // that wastes cycles and can only produce a spurious class. rms is the
    // DC-removed AC level (quiet room ≈ 15-75 here, real events ≳ 200).
    const float rms = clip_rms(s_audio, kClipSamples);
    if (rms < kSilenceRms) {
        return kClassAmbient;
    }

    extract_features(s_audio, s_input->data.int8);
    if (s_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[infer] Invoke failed");
        return -1;
    }
    int best = 0;
    int second = 0;
    int8_t best_v = s_output->data.int8[0];
    int8_t second_v = INT8_MIN;
    for (int i = 1; i < kNumClasses; i++) {
        if (s_output->data.int8[i] > best_v) {
            second_v = best_v;
            second = best;
            best_v = s_output->data.int8[i];
            best   = i;
        } else if (s_output->data.int8[i] > second_v) {
            second_v = s_output->data.int8[i];
            second = i;
        }
    }

    const float out_scale = s_output->params.scale;
    const int out_zp = s_output->params.zero_point;
    float prob[kNumClasses];
    for (int i = 0; i < kNumClasses; i++) {
        prob[i] = ((int)s_output->data.int8[i] - out_zp) * out_scale;
        if (prob[i] < 0.0f) prob[i] = 0.0f;
        if (prob[i] > 1.0f) prob[i] = 1.0f;
    }

    const float confidence = prob[best];
    const float margin = prob[best] - prob[second];
    const float dog_total = prob[kClassBark] + prob[kClassGrowl] + prob[kClassGrunt];
    Serial.printf("[score] bark=%.2f growl=%.2f grunt=%.2f ambient=%.2f dog=%.2f best=%s margin=%.2f rms=%.0f\n",
                  prob[kClassBark], prob[kClassGrowl], prob[kClassGrunt],
                  prob[kClassAmbient], dog_total, kClassNames[best], margin, rms);

    // Loud enough to be a real event (quiet was gated before Invoke). Fire a dog
    // action when the model is
    // either confident+unambiguous, or clearly points at a dog overall. The
    // bark/growl split often holds every single class below kActionMinConfidence
    // even on an obvious bark (e.g. bark=0.59 growl=0.36), so dog_total catches
    // that. Speech/TV keep dog_total low and fall through to no-op. The silence
    // gate above is what keeps this dog_total bypass from firing on noise.
    const bool clear_dog = dog_total >= kDogEventMinTotal;
    const bool confident = confidence >= kActionMinConfidence && margin >= kActionMinMargin;
    const bool fire = (best != kClassAmbient) && (clear_dog || confident);
    if (fire) {
        return best;
    }
    if (best != kClassAmbient) {
        Serial.printf("[infer] %s ignored: confidence=%.2f margin=%.2f dog=%.2f\n",
                      kClassNames[best], confidence, margin, dog_total);
    }
    return kClassAmbient;
}

// ═════════════════════════════════════════════════════════════════════════════
// Dispatch
// ═════════════════════════════════════════════════════════════════════════════

static void dispatch(int class_id) {
    switch (class_id) {
        case kClassBark:    handle_dog_sound("bark",  true); break;
        case kClassGrowl:   handle_dog_sound("growl", true); break;
        case kClassGrunt:   handle_dog_sound("grunt", true); break;
        case kClassAmbient: /* no-op (includes human speech) */ break;
        default:                              break;
    }
}

static void handle_dog_sound(const char* class_name, bool notify) {
    const uint32_t now = millis();
    Serial.printf("[dog] detected as %s\n", class_name);

    if (notify && now - s_last_notify_ms >= kNotifyCooldownMs) {
        s_last_notify_ms = now;
        char body[48];
        snprintf(body, sizeof(body), "Dog detected: %s", class_name);
        post_ntfy("BarkSense", body);
    }

    if (now - s_last_play_ms >= kPlayCooldownMs) {
        s_last_play_ms = now;
        led_blink(/*times=*/3, /*period_ms=*/120);
        led_color(255, 0, 0, 5000);
        Serial.println("[dog] playing owner voice");
        const bool ok = play_wav(kOwnerVoicePath);
        Serial.println(ok ? "[dog] playback done" : "[dog] playback failed");
    } else {
        Serial.printf("[dog] playback cooldown %lu ms remaining\n",
                      (unsigned long)(kPlayCooldownMs - (now - s_last_play_ms)));
    }

    log_event(class_name);
}

// ═════════════════════════════════════════════════════════════════════════════
// WAV playback
//   Streams a RIFF/WAVE PCM file from LittleFS to I2S1 (MAX98357A). Strict on
//   format: 16 kHz / 16-bit / mono / PCM. The mono samples are duplicated into
//   both I2S slots so the amp works whether SD/MODE selects left, right, or
//   stereo-average output. Skips aux chunks (LIST/JUNK/etc.) on the way to
//   'data'. Driver is left installed; DMA is zeroed at the end so the speaker
//   goes quiet without a tear-down/re-init cycle.
// ═════════════════════════════════════════════════════════════════════════════

static bool play_wav(const char* path) {
    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("[wav] open %s failed\n", path);
        return false;
    }

    struct __attribute__((packed)) WavHdr {
        char     riff[4];           // "RIFF"
        uint32_t chunk_size;
        char     wave[4];           // "WAVE"
        char     fmt[4];            // "fmt "
        uint32_t fmt_size;
        uint16_t format;            // 1 = PCM
        uint16_t channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample;
    } hdr;

    if (f.read((uint8_t*)&hdr, sizeof(hdr)) != sizeof(hdr)) {
        Serial.println("[wav] short read on header");
        f.close();
        return false;
    }
    if (memcmp(hdr.riff, "RIFF", 4) || memcmp(hdr.wave, "WAVE", 4) ||
        memcmp(hdr.fmt,  "fmt ", 4)) {
        Serial.println("[wav] not a RIFF/WAVE/fmt file");
        f.close();
        return false;
    }
    if (hdr.format != 1 || hdr.channels != 1 ||
        hdr.sample_rate != (uint32_t)kSampleRate || hdr.bits_per_sample != 16) {
        Serial.printf("[wav] unsupported: fmt=%u ch=%u rate=%lu bits=%u "
                      "(need PCM/mono/%dHz/16-bit)\n",
                      hdr.format, hdr.channels,
                      (unsigned long)hdr.sample_rate, hdr.bits_per_sample,
                      kSampleRate);
        f.close();
        return false;
    }
    // fmt chunk may carry extra bytes (WAVE_FORMAT_EXTENSIBLE = 40 B).
    if (hdr.fmt_size > 16) f.seek(f.position() + (hdr.fmt_size - 16));

    // Walk chunks until we hit 'data'. Skip LIST/JUNK/etc. Pad odd-size chunks.
    uint32_t data_size = 0;
    for (;;) {
        char     id[4];
        uint32_t sz;
        if (f.read((uint8_t*)id, 4) != 4 || f.read((uint8_t*)&sz, 4) != 4) {
            Serial.println("[wav] EOF before 'data' chunk");
            f.close();
            return false;
        }
        if (!memcmp(id, "data", 4)) { data_size = sz; break; }
        f.seek(f.position() + sz + (sz & 1));
    }

    // Cap playback to ~1.5 s. The loop is single-threaded, so the full clip would
    // freeze detection for its whole duration; 1.5 s keeps the demo responsive.
    const uint32_t kMaxPlayBytes = 48000;   // 1.5 s @ 16 kHz / 16-bit / mono
    if (data_size > kMaxPlayBytes) data_size = kMaxPlayBytes;

    Serial.printf("[wav] PCM 16k/16/mono, %lu bytes (%.2f s)\n",
                  (unsigned long)data_size, data_size / 32000.0f);

    i2s_zero_dma_buffer(I2S_NUM_1);
    static int16_t mono[512];         // 1 KB = 16 ms at 16 kHz mono 16-bit
    static int16_t stereo[1024];      // LRLR, same sample in both slots
    uint32_t remaining = data_size;
    while (remaining > 0) {
        const size_t want = remaining > sizeof(mono) ? sizeof(mono) : remaining;
        const size_t got  = f.read((uint8_t*)mono, want);
        if (got == 0) break;
        const size_t samples = got / sizeof(int16_t);
        for (size_t i = 0; i < samples; i++) {
            stereo[2 * i]     = mono[i];
            stereo[2 * i + 1] = mono[i];
        }
        size_t written = 0;
        const size_t out_bytes = samples * 2 * sizeof(int16_t);
        const esp_err_t err = i2s_write(I2S_NUM_1, stereo, out_bytes, &written, portMAX_DELAY);
        if (err != ESP_OK || written != out_bytes) {
            Serial.printf("[wav] i2s_write err=%d wrote=%u/%u\n",
                          err, (unsigned)written, (unsigned)out_bytes);
            f.close();
            i2s_zero_dma_buffer(I2S_NUM_1);
            return false;
        }
        remaining -= got;
    }

    // Wait for the DMA queue to drain so we don't cut the tail of the clip,
    // then silence the line. (i2s_write returns when bytes are queued, not
    // when they've left the chip.)
    delay((sizeof(mono) / 2) / (kSampleRate / 1000) + 20);  // ~36 ms safety
    i2s_zero_dma_buffer(I2S_NUM_1);
    f.close();
    return true;
}

static bool play_test_tone(uint32_t freq_hz, uint32_t duration_ms) {
    Serial.printf("[tone] %lu Hz for %lu ms\n",
                  (unsigned long)freq_hz, (unsigned long)duration_ms);

    i2s_zero_dma_buffer(I2S_NUM_1);
    static int16_t stereo[512];       // 256 stereo frames
    const uint32_t total_frames = (kSampleRate * duration_ms) / 1000;
    uint32_t phase = 0;
    const uint32_t phase_inc = (uint32_t)(((uint64_t)freq_hz << 32) / kSampleRate);
    uint32_t frames_left = total_frames;

    while (frames_left > 0) {
        const size_t frames = frames_left > 256 ? 256 : frames_left;
        for (size_t i = 0; i < frames; i++) {
            const float x = sinf((phase / 4294967296.0f) * 2.0f * PI);
            const int16_t sample = (int16_t)(x * 12000.0f);
            stereo[2 * i]     = sample;
            stereo[2 * i + 1] = sample;
            phase += phase_inc;
        }

        size_t written = 0;
        const size_t bytes = frames * 2 * sizeof(int16_t);
        const esp_err_t err = i2s_write(I2S_NUM_1, stereo, bytes, &written, portMAX_DELAY);
        if (err != ESP_OK || written != bytes) {
            Serial.printf("[tone] i2s_write err=%d wrote=%u/%u\n",
                          err, (unsigned)written, (unsigned)bytes);
            i2s_zero_dma_buffer(I2S_NUM_1);
            return false;
        }
        frames_left -= frames;
    }

    delay(40);
    i2s_zero_dma_buffer(I2S_NUM_1);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Notification & logging
// ═════════════════════════════════════════════════════════════════════════════

static bool post_ntfy(const char* title, const char* body) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[ntfy] WiFi down — skipped");
        return false;
    }
    HTTPClient http;
    http.begin(kNtfyUrl);
    http.addHeader("Title", title);
    http.addHeader("Tags",  "dog");
    const int code = http.POST((uint8_t*)body, strlen(body));
    http.end();
    Serial.printf("[ntfy] HTTP %d\n", code);
    return code >= 200 && code < 300;
}

static void log_event(const char* class_name) {
    File f = LittleFS.open(kEventLogPath, "a");
    if (!f) {
        Serial.printf("[log] open %s failed\n", kEventLogPath);
        return;
    }
    f.printf("%lu,%s\n", (unsigned long)millis(), class_name);
    f.close();
}

// ═════════════════════════════════════════════════════════════════════════════
// LED — single-LED placeholder for the RGB-LED spec
// ═════════════════════════════════════════════════════════════════════════════

static void led_set(bool on) { digitalWrite(kLedPin, on ? LOW : HIGH); }

static void led_color(uint8_t /*r*/, uint8_t /*g*/, uint8_t /*b*/, uint32_t hold_ms) {
    if (hold_ms == 0) return;
    led_set(true);
    delay(hold_ms);
    led_set(false);
}

static void led_blink(int times, uint32_t period_ms) {
    for (int i = 0; i < times; i++) {
        led_set(true);
        delay(period_ms / 2);
        led_set(false);
        delay(period_ms / 2);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// FEATURE_TEST — verify extract_features() matches the Python reference
// ═════════════════════════════════════════════════════════════════════════════

#ifdef FEATURE_TEST
static void run_feature_test() {
    static int8_t buf[kFeatDim * kNFrames];

    Serial.println("\n[test] extract_features() vs Python reference");
    Serial.printf("[test] audio %d samples, expected feature %d values\n",
                  kTestAudioLen, kExpectedFeatureLen);

    const uint32_t t0 = micros();
    extract_features(kTestAudio, buf);
    const uint32_t dt = micros() - t0;
    Serial.printf("[test] extract_features took %lu us (%lu ms)\n", dt, dt / 1000);

    int diffs_gt1 = 0, diffs_gt2 = 0;
    int max_diff = 0;
    long sum_abs = 0;
    for (int i = 0; i < kFeatDim * kNFrames; i++) {
        int d = (int)buf[i] - (int)kExpectedFeature[i];
        int ad = d < 0 ? -d : d;
        if (ad > 1) diffs_gt1++;
        if (ad > 2) diffs_gt2++;
        if (ad > max_diff) max_diff = ad;
        sum_abs += ad;
    }
    const int N = kFeatDim * kNFrames;
    Serial.printf("[test] mean |Δ| = %.3f  max |Δ| = %d\n",
                  (float)sum_abs / N, max_diff);
    Serial.printf("[test] |Δ| > 1: %d / %d (%.1f%%)\n",
                  diffs_gt1, N, 100.0f * diffs_gt1 / N);
    Serial.printf("[test] |Δ| > 2: %d / %d (%.1f%%)\n",
                  diffs_gt2, N, 100.0f * diffs_gt2 / N);

    // Pass if every single element is within ±2 of the Python reference.
    Serial.println(max_diff <= 2 ? "[test] PASS (all |Δ| ≤ 2)"
                                 : "[test] FAIL — extraction drift exceeds tolerance");

    // Bonus: run the model on the MCU-extracted features and check class.
    for (int i = 0; i < N; i++) s_input->data.int8[i] = buf[i];
    if (s_interpreter->Invoke() == kTfLiteOk) {
        int best = 0;
        int8_t bv = s_output->data.int8[0];
        for (int i = 1; i < kNumClasses; i++) {
            if (s_output->data.int8[i] > bv) { bv = s_output->data.int8[i]; best = i; }
        }
        Serial.printf("[test] model on MCU features → %s  (expected %s)\n",
                      kClassNames[best], kClassNames[kTestExpectedClass]);
    }
}
#endif
