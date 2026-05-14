# Speaking Detection: Implementation Guide & Research

This document summarizes the research-backed approach for visual-only voice activity detection (VAD) and the current implementation status in `main.cpp`.

---

## A) Research-Backed Approach

### 1. MediaPipe Blendshapes (Best Signal – If Available)
- **Source**: MediaPipe Face Landmarker can output blendshape coefficients (`jawOpen`, `mouthClose`, etc.)
- **Benefit**: Less noisy than raw landmarks; designed for facial actions
- **Usage**: If your pipeline exposes blendshapes (some only output mesh points), prefer:

  ```
  signal = 0.7 * jawOpen + 0.3 * mouthOpen
  ```

- **Note**: The current Face Landmark model outputs only 468 landmarks; blendshapes are not available on-device.

### 2. Landmark-Based Baseline (Current Implementation)
- **Source**: Standard practice for robust visual VAD
- **Approach**:
  - Mouth openness normalized by mouth width (scale-invariant)
  - Short-window motion energy (mean absolute velocity)
  - Adaptive thresholds learned during “not speaking”
  - Hangover / hysteresis for stable ON/OFF transitions

### 3. One Euro Filter
- **Source**: MediaPipe smoothing often uses One-Euro style
- **Benefit**: Less lag than heavy EMA when motion is real
- **Status**: Not implemented; current code uses EMA (MAR_SMOOTHING_ALPHA = 0.22).

### 4. Preprocessing as Root Cause
- **Source**: Embedded model quantization mismatch
- **Issue**: Using `int8 = uint8 - 128` assumes scale=1/255 and zero_point=-128. If the model expects different quantization, landmarks become wrong.
- **Fix**: Use `q = round(x / scale + zero_point)` with tensor scale/zero_point.
- **Status**: Implemented via `QuantizePixel()`.

---

## B) Current Implementation Status

| Step | Description | Status | Location |
|------|-------------|--------|----------|
| 1 | Fix INT8 preprocessing with scale/zero_point | Done | `QuantizePixel()`, lines ~217–230; both face detection and landmark |
| 2 | Bbox padding (10–20%) | Done | 15% padding, `BBOX_PADDING_PERCENT`, lines ~675–690 |
| 3 | 4-point mouth_open_norm (13, 14, 61, 291) | Done | `ComputeMouthOpenNorm()`, lines ~339–358 |
| 4 | Motion energy ring buffer | Done | `motionBuf[12]`, `ComputeMotionEnergyAndUpdate()` |
| 5 | Adaptive thresholds (noise_mean, noise_mad) | Done | `on_thresh = noise_mean + K_on*noise_mad` |
| 6 | Head-motion gating | Done | `headMoving` gates ON transition |
| 7 | Debug overlay | Done | `SPEAKING_DEBUG_OVERLAY`; set to 1 to enable |

### Tunable Constants (main.cpp)

```cpp
#define NORMALIZE_INPUT_TO_0_1  1   /* 0 if model expects raw [0,255] */
#define SPEAKING_DEBUG_OVERLAY  0   /* 1=draw op, me, on, off, hm */
#define OPEN_MIN_THRESHOLD      0.12f
#define CLOSE_MIN_THRESHOLD     0.08f
#define ADAPTIVE_K_ON           5.0f   /* Try 6.0 for less sensitivity */
#define ADAPTIVE_K_OFF          2.5f   /* Try 3.0 for less sensitivity */
#define HEAD_MOVE_THRESHOLD     0.12f
#define MOTION_ENERGY_WINDOW    12
```

---

## C) Cursor Prompt for Future Changes

Use this prompt when asking Cursor to modify the speaking detector:

```text
We have an embedded face landmark pipeline (Nuvoton M55, TFLite int8) and our speaking detector is unstable: flickers between speaking/not speaking, and head movement triggers "speaking". Landmarks also lag and do not track lips accurately. Please modify main.cpp to improve both landmark accuracy and speaking detection robustness.

GOALS
1) Fix any model input preprocessing mismatch (most important) so landmarks track mouth correctly.
2) Replace current MAR-based speaking detector with a more robust "visual VAD":
   - mouth openness normalized by mouth width (scale-invariant)
   - short-window motion energy (mean |velocity| over last N updates)
   - adaptive thresholds learned during "not speaking"
   - hangover / hysteresis with stable ON/OFF transitions
   - explicit head-motion gating (based on bbox center velocity)
3) Add on-screen debug overlays to tune thresholds on-device.

CONTEXT / CURRENT STATE
- We currently quantize model input by doing int8 = uint8 - 128 for both face detection and face landmark models.
- We already have access to input tensor quant params via GetTensorQuantParams(inputTensor).
- We already track bbox movement and have per-face tracks.
- We run speaking detection every SPEAKING_DETECT_EVERY_N_FRAMES frames.

STEP 1 — FIX INT8 PREPROCESSING (do this first)
For BOTH DetectFaceRegion() and DetectFaceLandmark_DrawResult():
- Replace the "signed_req_data[i] = req_data[i] - 128;" shortcut with correct quantization using input tensor quant params:
  - Convert uint8 pixel to float in [0,1] (or [0,255] depending on model expectation) then apply:
    q = round(x / scale + zero_point)
    clamp to int8 range
- Add a compile-time switch NORMALIZE_TO_0_1 that defaults ON; we can flip if needed.

STEP 2 — STABILIZE THE FACE CROP
- Keep using RAW bbox for crop (we tried smoothed bbox and it caused poor lip tracking).
- Add padding around bbox (e.g., 10–20%) before cropping, clamped to image boundaries.
- Optionally clamp bbox aspect ratio to square (centered) to reduce warp differences frame-to-frame.

STEP 3 — NEW MOUTH METRICS
Use MediaPipe-style mouth indices:
- Upper lip center: 13
- Lower lip center: 14
- Mouth left corner: 61
- Mouth right corner: 291
Compute:
- mouth_open = distance(13,14) in face-crop pixel coords
- mouth_width = distance(61,291)
- mouth_open_norm = mouth_open / (mouth_width + eps)
- jaw_open = distance(13,152) normalized by mouth_width (optional)

STEP 4 — WINDOWED MOTION ENERGY + ADAPTIVE THRESHOLDS + HANGOVER
- Ring buffer of last ~10–20 velocity samples
- motion_energy = mean(|velocity| over window)
- noise_mean, noise_mad = EMA when NOT speaking
- on_thresh = noise_mean + K_on * noise_mad (K_on=6)
- off_thresh = noise_mean + K_off * noise_mad (K_off=3)
- Turn ON: motion_energy > on_thresh for N_on consecutive updates AND mouth_open_norm > open_min
- Turn OFF: motion_energy < off_thresh for N_off consecutive updates OR mouth_open_norm < close_min

STEP 5 — HEAD MOTION GATING
- If head_move > threshold: do not allow NOT speaking -> speaking
- Do allow speaking -> speaking (don't force OFF) unless mouth_open_norm is very low

STEP 6 — DEBUG OVERLAY
Draw: mouth_open_norm, motion_energy, noise_mean, noise_mad, on_thresh, off_thresh, head_move, speaking state, confirm counters
```

---

## D) Optional Improvements Not Yet Implemented

1. **Jaw proxy (chin 152)**: Add `jaw_open = distance(13,152) / mouth_width` and blend with mouth_open_norm for a stronger jaw-opening cue.
2. **K_on=6, K_off=3**: Use `ADAPTIVE_K_ON=6.0f`, `ADAPTIVE_K_OFF=3.0f` in `main.cpp` (around lines 66–67) for tighter thresholds.
3. **Square aspect crop**: Constrain crop to square (centered) to reduce warp differences frame-to-frame.
4. **One Euro filter**: Replace EMA smoothing with One Euro for less lag when motion is fast.
5. **RGB vs BGR**: Verify `imlib_nvt_scale` outputs RGB and `PIXFORMAT_RGB888` matches the model.

---

## E) Quick “Why It Flickers” Diagnosis

- **Input quantization mismatch** → landmarks “soft” / delayed / wrong → mouth signal becomes noise.
- **Frame-to-frame deltas** similar to landmark noise → fragile thresholds.
- **Heavy smoothing + every-N-frames** → phase lag when mouth opens.
- **Face order changes** → state indexed by face index instead of stable trackId → fixed by using `trackId` for all state.

---

## F) References

- MediaPipe Face Landmarker: [ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- MediaPipe Face Mesh indices: [mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html](https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html)
- One Euro filter: [cristal.univ-lille.fr/~casiez/1euro/](https://cristal.univ-lille.fr/~casiez/1euro/)
- TFLite quantization: `real_value = scale * (quantized_value - zero_point)`, so `quantized = round(real_value / scale) + zero_point`
