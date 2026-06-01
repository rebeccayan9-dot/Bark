#pragma once
#include <cstdint>

// ── DSP — must match scripts/preprocess.py exactly ───────────────────────────
constexpr int kSampleRate    = 16000;
constexpr int kClipSamples   = 16000;   // 1-second inference window
constexpr int kNFft          = 512;
constexpr int kHopLength     = 256;
// center=False: 1 + (kClipSamples - kNFft) / kHopLength = 61
constexpr int kNFrames       = 1 + (kClipSamples - kNFft) / kHopLength;
// kFeatDim / kMelBins / kFftBins live in norm_params.h + mel_filterbank.h
// (auto-generated). kDeltaWidth / kDeltaHalf live in delta_kernels.h. Don't
// redefine here or the constexpr collision breaks the build.

// ── Classifier ────────────────────────────────────────────────────────────────
enum ClassId : uint8_t {
    kClassBark    = 0,
    kClassGrowl   = 1,
    kClassGrunt   = 2,
    kClassAmbient = 3,   // includes human speech (folded into ambient at training time)
    kNumClasses   = 4,
};
constexpr const char* kClassNames[kNumClasses] = {
    "bark", "growl", "grunt", "ambient",
};

// ── Inference ─────────────────────────────────────────────────────────────────
// Arena lives in PSRAM (heap_caps_malloc MALLOC_CAP_SPIRAM in main.cpp) so SRAM
// stays free for stack, DMA, WiFi, and audio buffers. Sized for DS-CNN α=1.0
// INT8 (~1.8 MB estimated arena); 3 MB gives headroom on the 8 MB PSRAM. Check
// the boot log "[ready] arena X / Y bytes" for actual usage and trim if desired.
constexpr int      kTensorArenaBytes  = 3 * 1024 * 1024;
constexpr uint32_t kNotifyCooldownMs  = 60'000;  // ntfy POST cooldown
constexpr uint32_t kPlayCooldownMs    = 10'000;  // owner-voice playback cooldown

// Ignore low-confidence predictions. This prevents ordinary speech/noise from
// triggering bark/growl/grunt actions just because one class had the largest
// softmax score.
constexpr float    kActionMinConfidence = 0.75f;
constexpr float    kActionMinMargin     = 0.25f;

// A clip whose softmax clearly points at a dog overall, even if the bark/growl
// split keeps any single class below kActionMinConfidence. Paired with the
// silence gate below so it can't fire on near-silent noise.
constexpr float    kDogEventMinTotal = 0.85f;

// Absolute loudness gate (DC-removed RMS of the raw int16 clip — see clip_rms).
// The model was trained only on clips that contain real audio; on near-silence,
// per-clip power_to_db(ref=max) normalization amplifies the noise floor into a
// structured spectrogram and the model emits a confident (wrong) dog class.
// Clips quieter than this are treated as no-event regardless of the softmax.
// Measured on this XIAO: quiet room ≈ 15-40, claps/loud sound ≈ 400-700, so 150
// sits ~4× above the floor and well below any real acoustic event.
constexpr float    kSilenceRms = 150.0f;

// ── Hardware — XIAO ESP32S3 Sense ────────────────────────────────────────────
// PDM mic on I2S0 (board-integrated)
constexpr int kMicClkPin    = 42;
constexpr int kMicDataPin   = 41;

// MAX98357A speaker on I2S1.
// These are ESP32-S3 GPIO numbers. On the XIAO ESP32S3 edge connector:
//   GPIO1 = D0, GPIO2 = D1, GPIO3 = D2.
constexpr int kSpkBclkPin   = 1;  // XIAO D0 -> MAX98357A BCLK
constexpr int kSpkLrclkPin  = 2;  // XIAO D1 -> MAX98357A LRC / WS
constexpr int kSpkDinPin    = 3;  // XIAO D2 -> MAX98357A DIN

// Board user LED (single colour, active-low). Colour-coded responses in the
// spec assume an RGB LED — patch the LED helpers if you wire one up.
constexpr int kLedPin       = 21;

// ── LittleFS paths ───────────────────────────────────────────────────────────
constexpr const char* kOwnerVoicePath = "/owner_voice.wav";
constexpr const char* kEventLogPath   = "/events.log";
