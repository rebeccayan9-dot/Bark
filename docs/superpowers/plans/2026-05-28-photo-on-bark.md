# Photo on Dog Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When BarkSense detects a dog and sends a notification, attach a still JPEG from the OV2640 camera so the owner sees the scene in the push notification.

**Architecture:** Add `esp_camera` (OV2640) init at boot. On the existing bark-notify path in `handle_dog_sound()`, capture one JPEG and PUT it to ntfy as an attachment (ntfy renders it inline). Camera failure degrades gracefully to the current text-only `post_ntfy()`. A `-DCAMERA_TEST` build flag exercises camera+upload independently, mirroring the existing `FEATURE_TEST` / `AUDIO_TEST` harnesses.

**Tech Stack:** C++17, Arduino-ESP32, PlatformIO, `esp_camera` (bundled with the ESP32 Arduino framework), XIAO ESP32S3 Sense (OV2640), ntfy.sh attachments.

**Testing note:** This is firmware with no host unit-test framework. "Test" = compile (`pio run`) for each task, plus the on-device `CAMERA_TEST` harness (flash, watch serial, confirm the phone receives a photo). Build command throughout: `pio run -d barksense/firmware -e seeed_xiao_esp32s3`.

---

### Task 1: Camera pin map header

**Files:**
- Create: `barksense/firmware/include/camera_pins.h`

- [ ] **Step 1: Create the XIAO ESP32S3 Sense OV2640 pin map**

These are the fixed DVP pins on the XIAO ESP32S3 Sense B2B connector. They do not overlap the PDM mic (41/42), speaker (1/2/3), or LED (21).

```cpp
#pragma once

// OV2640 DVP pins — XIAO ESP32S3 Sense (fixed on the board's B2B connector).
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   10
#define SIOD_GPIO_NUM   40   // SCCB SDA
#define SIOC_GPIO_NUM   39   // SCCB SCL
#define Y9_GPIO_NUM     48
#define Y8_GPIO_NUM     11
#define Y7_GPIO_NUM     12
#define Y6_GPIO_NUM     14
#define Y5_GPIO_NUM     16
#define Y4_GPIO_NUM     18
#define Y3_GPIO_NUM     17
#define Y2_GPIO_NUM     15
#define VSYNC_GPIO_NUM  38
#define HREF_GPIO_NUM   47
#define PCLK_GPIO_NUM   13
```

- [ ] **Step 2: Commit**

```bash
git add barksense/firmware/include/camera_pins.h
git commit -m "feat(cam): add XIAO ESP32S3 OV2640 pin map"
```

---

### Task 2: Camera tunables in config.h

**Files:**
- Modify: `barksense/firmware/include/config.h`

- [ ] **Step 1: Add camera constants after the Inference section**

Insert immediately after the line `constexpr uint32_t kPlayCooldownMs    = 10'000;  // owner-voice playback cooldown` (config.h:32):

```cpp

// ── Camera — OV2640 on XIAO ESP32S3 Sense ────────────────────────────────────
// Frame size (FRAMESIZE_VGA) is set in init_camera(); kept out of config.h so
// this header doesn't need to pull in esp_camera.h everywhere it's included.
constexpr int kCameraXclkHz      = 20'000'000;
constexpr int kCameraJpegQuality = 12;   // 0–63, lower = better quality / bigger file
```

- [ ] **Step 2: Verify it compiles**

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3`
Expected: builds with no new errors (the constants are unused for now; `-Wno-unused-variable` is already set).

- [ ] **Step 3: Commit**

```bash
git add barksense/firmware/include/config.h
git commit -m "feat(cam): add camera tunables to config"
```

---

### Task 3: Wire the CAMERA_TEST build flag

**Files:**
- Modify: `barksense/firmware/platformio.ini:34-38`

- [ ] **Step 1: Add the commented CAMERA_TEST flag**

After the `; -DAUDIO_TEST` block (platformio.ini:38), add:

```ini
    ; Uncomment to capture one frame at boot, PUT it to ntfy, and halt.
    ; Verifies camera init + JPEG capture + ntfy attachment upload, independent
    ; of the audio pipeline. Needs WiFi (secrets.h). FEATURE_TEST / AUDIO_TEST
    ; take precedence if also set.
    ; -DCAMERA_TEST
```

- [ ] **Step 2: Verify it compiles (flag still commented)**

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3`
Expected: builds unchanged.

- [ ] **Step 3: Commit**

```bash
git add barksense/firmware/platformio.ini
git commit -m "chore(cam): add commented CAMERA_TEST build flag"
```

---

### Task 4: Camera init

**Files:**
- Modify: `barksense/firmware/src/main.cpp` (includes, forward decls, new function, setup() call)

- [ ] **Step 1: Add the esp_camera include and pin map**

After the `#include "delta_kernels.h"` line (main.cpp:43), add:

```cpp
#include "esp_camera.h"
#include "camera_pins.h"
```

- [ ] **Step 2: Add the camera-ok flag**

After `static uint32_t s_last_play_ms   = 0;` (main.cpp:82), add:

```cpp

// Set true once esp_camera_init() succeeds; gates photo capture so the audio
// pipeline still runs if the camera is missing or fails to init.
static bool s_camera_ok = false;
```

- [ ] **Step 3: Add the forward declaration**

After `static void init_hann();` (main.cpp:95), add:

```cpp
static void init_camera();
```

- [ ] **Step 4: Implement init_camera()**

After `init_hann()`'s definition (the closing brace at main.cpp:292), add:

```cpp

static void init_camera() {
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = kCameraXclkHz;
    config.frame_size   = FRAMESIZE_VGA;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = kCameraJpegQuality;
    config.fb_count     = 1;

    const esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[cam] init failed 0x%x — photos disabled\n", err);
        s_camera_ok = false;
        return;
    }
    s_camera_ok = true;
    Serial.println("[cam] OV2640 ready");
}
```

- [ ] **Step 5: Call init_camera() in setup()**

In `setup()`, after `init_i2s_spk();` (main.cpp:131), add:

```cpp
    init_camera();        // non-fatal: photos disabled if it fails
```

- [ ] **Step 6: Verify it compiles**

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3`
Expected: PASS. If the build cannot find `esp_camera.h`, add `espressif/esp32-camera` to `lib_deps` in platformio.ini, commit that as a separate change, and rebuild. (Normally it's bundled with the ESP32 Arduino framework and no dep is needed.)

- [ ] **Step 7: Commit**

```bash
git add barksense/firmware/src/main.cpp
git commit -m "feat(cam): initialize OV2640 at boot (non-fatal)"
```

---

### Task 5: Photo-attachment upload

**Files:**
- Modify: `barksense/firmware/src/main.cpp` (forward decl + new function)

- [ ] **Step 1: Add the forward declaration**

After `static bool post_ntfy(const char* title, const char* body);` (main.cpp:107), add:

```cpp
static bool post_ntfy_photo(const char* title, const char* body);
```

- [ ] **Step 2: Implement post_ntfy_photo()**

Immediately after the `post_ntfy()` function definition (after its closing brace at main.cpp:712), add. Note `len` is read into a local *before* `esp_camera_fb_return(fb)` so the printf never touches freed memory:

```cpp

// Capture one JPEG and PUT it to ntfy as an attachment. ntfy treats the PUT
// body as a file and renders images inline in the notification. Returns false
// (so the caller can fall back to text-only post_ntfy) if WiFi is down, the
// camera isn't ready, capture fails, or the HTTP status isn't 2xx.
static bool post_ntfy_photo(const char* title, const char* body) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[ntfy] WiFi down — photo skipped");
        return false;
    }
    if (!s_camera_ok) return false;

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[cam] capture failed");
        return false;
    }
    const size_t len = fb->len;

    HTTPClient http;
    http.begin(kNtfyUrl);
    http.addHeader("Filename", "bark.jpg");
    http.addHeader("Title", title);
    http.addHeader("Tags", "dog,camera");
    http.addHeader("Message", body);
    http.addHeader("Content-Type", "image/jpeg");
    const int code = http.PUT(fb->buf, len);
    http.end();

    esp_camera_fb_return(fb);
    Serial.printf("[ntfy] photo HTTP %d (%u bytes)\n", code, (unsigned)len);
    return code >= 200 && code < 300;
}
```

- [ ] **Step 3: Verify it compiles**

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3`
Expected: PASS (function is unused for now; `-Wno-unused-function` is set).

- [ ] **Step 4: Commit**

```bash
git add barksense/firmware/src/main.cpp
git commit -m "feat(cam): add ntfy photo-attachment upload"
```

---

### Task 6: CAMERA_TEST harness (on-device test)

**Files:**
- Modify: `barksense/firmware/src/main.cpp` (setup() test branches)

- [ ] **Step 1: Add the CAMERA_TEST branch in setup()**

In `setup()`, change the `#elif defined(AUDIO_TEST)` chain. Insert a new `#elif` branch *before* the final `#else` (between main.cpp:146 `for (;;) { delay(1000); }` of the AUDIO_TEST block and main.cpp:147 `#else`):

```cpp
#elif defined(CAMERA_TEST)
    init_wifi();          // needed to upload the test shot
    Serial.println("[camera_test] capturing one frame and posting to ntfy");
    const bool ok = post_ntfy_photo("BarkSense test", "camera test shot");
    Serial.printf("[camera_test] %s — halt, power-cycle to rerun\n",
                  ok ? "PASS" : "FAIL");
    for (;;) { delay(1000); }
```

- [ ] **Step 2: Verify the normal build still compiles**

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3`
Expected: PASS.

- [ ] **Step 3: Build and flash with CAMERA_TEST enabled**

Uncomment `-DCAMERA_TEST` in `barksense/firmware/platformio.ini`, then:

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3 -t upload && pio device monitor -e seeed_xiao_esp32s3`
Expected serial output:
```
[cam] OV2640 ready
[wifi] <ip>
[camera_test] capturing one frame and posting to ntfy
[ntfy] photo HTTP 200 (NNNNN bytes)
[camera_test] PASS — halt, power-cycle to rerun
```
And: the ntfy mobile app (subscribed to the topic in `secrets.h`) shows a notification **with an embedded photo**.

If `[ntfy] photo HTTP` is non-2xx: check `kNtfyUrl`/topic and WiFi. If `[cam] capture failed`: re-seat the camera ribbon and confirm `[cam] OV2640 ready` printed.

- [ ] **Step 4: Re-comment the flag**

Re-comment `-DCAMERA_TEST` in `barksense/firmware/platformio.ini` so normal builds run the real loop.

- [ ] **Step 5: Commit**

```bash
git add barksense/firmware/src/main.cpp barksense/firmware/platformio.ini
git commit -m "test(cam): add CAMERA_TEST boot harness"
```

---

### Task 7: Send the photo on a real bark

**Files:**
- Modify: `barksense/firmware/src/main.cpp:530-533` (the notify block in `handle_dog_sound`)

- [ ] **Step 1: Swap the notify call to try photo first, fall back to text**

Replace this block (main.cpp:530-533):

```cpp
    if (notify && now - s_last_notify_ms >= kNotifyCooldownMs) {
        s_last_notify_ms = now;
        post_ntfy("BarkSense", "Dog sound detected");
    }
```

with:

```cpp
    if (notify && now - s_last_notify_ms >= kNotifyCooldownMs) {
        s_last_notify_ms = now;
        // Photo + alert; fall back to text-only if the camera/capture fails.
        if (!post_ntfy_photo("BarkSense", "Dog sound detected"))
            post_ntfy("BarkSense", "Dog sound detected");
    }
```

- [ ] **Step 2: Verify it compiles**

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3`
Expected: PASS.

- [ ] **Step 3: Flash and verify end-to-end**

Run: `pio run -d barksense/firmware -e seeed_xiao_esp32s3 -t upload && pio device monitor -e seeed_xiao_esp32s3`
Then play a dog bark near the mic. Expected serial:
```
[infer] bark
[dog] detected as bark
[ntfy] photo HTTP 200 (NNNNN bytes)
```
And the phone notification includes the photo. Cover the camera and bark again within 60 s → no second notification (notify cooldown), confirming no spam.

- [ ] **Step 4: Commit**

```bash
git add barksense/firmware/src/main.cpp
git commit -m "feat(cam): attach photo to bark notification"
```
