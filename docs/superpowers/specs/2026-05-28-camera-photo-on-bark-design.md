# BarkSense — Photo on Dog Detection

**Date:** 2026-05-28
**Status:** Design approved, ready for implementation plan
**Scope:** Single feature — capture a still photo when a dog is detected and deliver it to the owner's phone. No presence/away detection, no false-positive tuning (both deferred to later specs).

## Problem

BarkSense runs while the owner is away and notifies them when the dog vocalizes. Today the alert is text-only ("Dog sound detected"). The owner can't see what's actually happening at home. The XIAO ESP32S3 Sense has an onboard OV2640 camera that is currently unused.

## Goal

When a dog event triggers a notification, attach a still photo of the scene so the owner sees the situation in the same push notification.

### Non-goals (explicitly out of scope for this spec)

- Using the camera to decide whether anyone is home (presence/geofence) — deferred.
- Tightening the audio false-positive logic (`kActionMinConfidence` etc.) — deferred.
- Video / multi-frame clips — single still only.
- Long-term photo archival — photos are transient alert payloads.
- Night/low-light capture — accepted limitation (see below).

## Approach

Deliver the photo as an **ntfy attachment**. The firmware already POSTs text alerts to `ntfy.sh` (`post_ntfy` in `main.cpp`). ntfy supports file attachments: PUT the JPEG as the request body with a `Filename` header, and the ntfy mobile app renders the image inline in the notification.

This requires **no new backend, no custom app, and no extra storage**. An OV2640 VGA JPEG is ~10–50 KB, well under ntfy.sh's 15 MB attachment limit.

**Trade-off accepted:** attachments on ntfy.sh expire after a few hours. This is fine for a real-time alert; it is not an archive. If archival is wanted later, that becomes a separate spec (host image, send URL via ntfy's `Attach` header).

Rejected alternatives:
- External storage + image URL: re-introduces a backend; defeats the simplicity win.
- Dedicated IoT platform (Blynk/MQTT + image): overkill for one photo per event.

## Hardware feasibility (verified)

XIAO ESP32S3 Sense, 8 MB PSRAM (`qio_opi`), 8 MB flash (4 MB app + 4 MB LittleFS).

- **No peripheral conflict:** camera uses the LCD_CAM peripheral, PDM mic uses I2S0, speaker uses I2S1.
- **No pin conflict:** camera DVP pins are on the board's B2B connector (XCLK 10, SIOD 40, SIOC 39, VSYNC 38, HREF 47, PCLK 13, D0–D7 = 15/17/18/16/14/12/11/48). None overlap the mic (41/42), speaker (1/2/3), or LED (21).
- **Memory:** the 480 KB TFLite arena is in PSRAM; a single JPEG frame buffer (`fb_count = 1`, `fb_location = PSRAM`) adds tens of KB. Plenty of headroom in 8 MB.

## Components

### `init_camera()`
- Called in `setup()` after `init_tflite()` (so PSRAM is confirmed working first).
- Uses `esp_camera` (bundled with the Arduino-ESP32 framework; `#include "esp_camera.h"`).
- Config: `PIXFORMAT_JPEG`, `FRAMESIZE_VGA`, `jpeg_quality ≈ 12`, `fb_count = 1`, `fb_location = CAMERA_FB_IN_PSRAM`, `grab_mode = CAMERA_GRAB_LATEST`, `xclk_freq_hz = 20 MHz`.
- Pin map: XIAO_ESP32S3 standard camera pins (constants added to `config.h` or a small `camera_pins.h`).
- Non-fatal on failure: log it and set a `s_camera_ok` flag false. The audio pipeline must still run without the camera.

### `capture_and_post_photo(title, body)`
- `esp_camera_fb_get()` → JPEG buffer (`fb->buf`, `fb->len`).
- PUT to `kNtfyUrl` with headers: `Filename: bark.jpg`, `Title`, `Tags: dog`, `Message: <body>`.
- `esp_camera_fb_return(fb)` always, even on HTTP failure.
- Returns success/failure.

### `post_ntfy(title, body)` (existing)
- Retained as the text-only fallback path.

## Data flow

```
loop(): capture_audio → run_inference → dispatch
                                          │
                              handle_dog_sound("bark", notify=true)
                                          │
              notify && cooldown elapsed ─┤
                                          ▼
                          camera ok? ── yes ─▶ capture_and_post_photo()
                                  │                  │ fail
                                  └── no ────────────┴──▶ post_ntfy()  (text only)
                          WiFi down ─▶ skip (existing behavior)
```

- The photo is tied to the existing notification path (the `notify == true` / bark case) and reuses `kNotifyCooldownMs` (60 s), so no notification spam.
- `growl`/`grunt` keep their current behavior (no push). Extending photos to them is a future tweak, not part of this spec.

## Timing

Single-threaded, time-multiplexed loop is unchanged. On a dog event the device spends ~100–300 ms capturing plus WiFi-dependent upload time during which the mic is not sampling. This is acceptable: the device is already in a 60 s notify cooldown after an event, so missing a second of audio there has no practical cost.

## Failure handling

| Condition | Behavior |
|---|---|
| Camera init fails at boot | Log, `s_camera_ok = false`, continue; alerts fall back to text-only. |
| `esp_camera_fb_get()` returns null | Log, fall back to `post_ntfy()` text-only for this event. |
| WiFi down | Skip notification entirely (existing behavior). |
| ntfy HTTP non-2xx | Log the code; event still recorded via `log_event()`. |

## Testing

- **`-DCAMERA_TEST` build flag** (same pattern as `AUDIO_TEST` / `FEATURE_TEST`): at boot, capture one frame and PUT it to ntfy, print the HTTP result, then halt. Verifies the camera + upload chain independently of the audio model.
- Manual end-to-end: play a bark near the mic, confirm the phone receives a notification with an embedded photo.

## Risks / things to verify during implementation

- Confirm `esp_camera.h` is available from the Arduino framework on this PlatformIO platform without adding a lib dependency; if not, add the `espressif/esp32-camera` component.
- Confirm camera DMA line buffers (internal SRAM) coexist with I2S DMA + WiFi without exhausting SRAM. Check free heap at boot after `init_camera()`.
- Confirm `HTTPClient::PUT(uint8_t*, size_t)` sends the binary body intact to ntfy.

## Accepted limitations

- **Low light:** OV2640 has no IR illumination; night captures will be dark. The text notification still arrives. Fill light / IR is out of scope.
