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
    kClassAmbient = 3,
    kNumClasses   = 4,
};
constexpr const char* kClassNames[kNumClasses] = {
    "bark", "growl", "grunt", "ambient",
};

// ── Inference ─────────────────────────────────────────────────────────────────
// 460 KB arena lives in PSRAM (EXT_RAM_BSS_ATTR in main.cpp) so SRAM stays
// free for stack, DMA, WiFi, and audio buffers.
constexpr int      kTensorArenaBytes  = 480 * 1024;   // ~457 KB needed + allocator overhead
constexpr uint32_t kNotifyCooldownMs  = 60'000;  // ntfy POST cooldown
constexpr uint32_t kPlayCooldownMs    = 60'000;  // owner-voice playback cooldown

// ── Hardware — XIAO ESP32S3 Sense ────────────────────────────────────────────
// PDM mic on I2S0 (board-integrated)
constexpr int kMicClkPin    = 42;
constexpr int kMicDataPin   = 41;

// MAX98357A speaker on I2S1
constexpr int kSpkBclkPin   = 1;
constexpr int kSpkLrclkPin  = 2;
constexpr int kSpkDinPin    = 3;

// Board user LED (single colour, active-low). Colour-coded responses in the
// spec assume an RGB LED — patch the LED helpers if you wire one up.
constexpr int kLedPin       = 21;

// ── LittleFS paths ───────────────────────────────────────────────────────────
constexpr const char* kOwnerVoicePath = "/owner_voice.wav";
constexpr const char* kEventLogPath   = "/events.log";
