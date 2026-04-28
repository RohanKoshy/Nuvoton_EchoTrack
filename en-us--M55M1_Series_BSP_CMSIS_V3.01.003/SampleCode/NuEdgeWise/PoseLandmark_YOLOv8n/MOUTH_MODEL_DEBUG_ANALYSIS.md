# Mouth Model Debug Analysis – Response to Max's Recommendations

This document addresses Max's debugging points and provides concrete checks and next steps.

---

## 1. Model Loading Verification

**Recommendation:** Load the other team's working model (YOLOv8n-pose or EXERCISE) to verify loading pipeline.

### How to test

1. **Replace the mouth model file on SD card** with a working model:
   - `YOLOv8n-pose.tflite` (from the working PoseLandmark sample)
   - `EXERCISE_int8_vela.tflite` (from the working sample)

2. **Update code** in `main.cpp`:
   - Change `MODEL_FILE` to `"0:\\YOLOv8n-pose.tflite"` or `"0:\\EXERCISE_int8_vela.tflite"`
   - Use `arm::app::YOLOv8nPoseModel` or `arm::app::ExerciseClassifierModel` (would require adding that model class and post-processing)
   - Or **simpler:** run the existing **PoseLandmark_YOLOv8n_workout_w_accel** project with its SD models – if that works, loading and arena setup are fine.

3. **Interpretation:**
   - If the **working project** runs pose + exercise from SD → loading and arena are OK; the mouth model is the likely problem.
   - Max reported the same failure when loading the mouth model in another group's code → strongly supports a model-specific issue.

---

## 2. Tensor Arena Size

**Current setting:** `ACTIVATION_BUF_SZ=0x80000` (512KB) in PoseLandmark.uvprojx.

**Max’s observation:** "We increased activation_buffer until it overlapped with UART and still had the error."

This strongly suggests the failure is **not** due to arena size. When the arena is oversized:
- The allocator should succeed.
- A different failure would occur (e.g. incorrect outputs, memory corruption, crashes during inference).

So the "tensor allocation failed" is probably caused by something other than arena capacity.

**Constraint:** The arena must stay **entirely in SRAM** (Ethos-U cannot use HyperRAM for activations). SRAM region: 0x81F00000–0x81FFFFFF (~1MB). With SRAM_NONCACHEABLE using part of it, the arena should be ≤512KB–1MB depending on layout.

---

## 3. Vela Compilation and NPU Configuration

**Recommendation:** Ensure the mouth model is compiled with Vela for Ethos-U55-256 and compatible settings.

### M55M1 NPU config

- **Hardware:** Ethos-U55 with **256 MACs** (from `ethosu_npu_init` logs: `MACs/cc: 256`).
- **Target:** `ethos-u55-256`.

### Required Vela parameters

The model must be Vela-optimized with matching config. Example:

```bash
vela yolo-fastest-1.1-int8.tflite \
  --accelerator-config=ethos-u55-256 \
  --optimise=Performance \
  --memory-mode=Shared_Sram \
  --system-config=Ethos_U55_High_End_Embedded \
  --config=<path-to-vela.ini> \
  --output-dir=./vela_output
```

**Checklist:**

| Item | Status |
|------|--------|
| `--accelerator-config=ethos-u55-256` | Must match M55M1 (256 MACs) |
| `--memory-mode=Shared_Sram` | Typical for this platform |
| `--system-config=Ethos_U55_High_End_Embedded` | Standard for Cortex-M55 + Ethos-U55 |

### Possible problems

1. **Config mismatch:** Model compiled with `ethos-u55-128` → incompatible with 256-MAC NPU.
2. **No Vela pass:** If the model was not run through Vela, it has no Ethos-U custom ops. TFLite Micro then falls back to CPU for all layers, and "tensor allocation failed" could come from different causes (e.g. memory layout, unsupported op).
3. **Wrong Vela environment:** Notebook or script uses different Vela version or parameters than expected.

### How to check if the model is Vela-optimized

- Inspect the model (e.g. Netron) for **Ethos-U custom operator** nodes.
- Or run:

```bash
vela --help
# Then try running Vela on the model and observe output
vela yolo-fastest-1.1-int8_vela.tflite --accelerator-config=ethos-u55-256 --show-cpu-operations 2>&1
```

- If Vela reports “already optimized” or processes it successfully, the model likely contains Ethos-U ops.
- `--show-cpu-operations` shows which layers remain on CPU.

---

## 4. Op Resolver and Ethos-U Registration

**Current setup:** `MouthDetectionModel.cpp` registers:

- `AddDepthwiseConv2D`, `AddConv2D`, `AddAdd`, `AddResizeNearestNeighbor`, `AddPad`, `AddMaxPool2D`, `AddConcatenation`, `AddTranspose`
- `AddEthosU()` (when `ARM_NPU` is defined)

If the model includes **any operator not in this list**, allocation or invocation can fail. Use TFLite tooling or the `generate_micro_mutable_op_resolver` flow to confirm the full op set used by the model.

---

## 5. Model Input/Output Matching

**Expected for mouth model (192×192 YOLO-Fastest):**

- **Input:** `[1, 192, 192, 3]` (int8, quantized)
- **Output:** YOLO detection tensors (e.g. stride 32: 6×6, stride 16: 12×12)

**Working models for reference:**

- **YOLOv8n-pose:** 192×192 input, different output layout (pose keypoints).
- **EXERCISE:** 96×96 or similar, classifier.

If the mouth model’s input shape or output layout does not match what the app expects, post-processing will be wrong, but allocation usually succeeds. Since Max sees “tensor allocation failed”, the more likely issue is:

- Vela/NPU config mismatch, or  
- Unsupported ops / memory layout.

---

## 6. Summary and Recommended Actions

| Priority | Action | Purpose |
|---------|--------|---------|
| 1 | Load `YOLOv8n-pose.tflite` or `EXERCISE_int8_vela.tflite` in the working project | Confirm SD loading and arena setup |
| 2 | Confirm Vela settings: `ethos-u55-256`, `Shared_Sram`, `Ethos_U55_High_End_Embedded` | Match NPU hardware and platform |
| 3 | Re-run mouth model through Vela with the correct config and replace on SD | Ensure Ethos-U ops and SRAM-compatible layout |
| 4 | Optionally try a different pretrained YOLO model and fine-tune for mouth | Reduce chance of model-specific incompatibility |
| 5 | Try retraining at lower input resolution (e.g. 128×128) | Lower activation memory, easier to fit in SRAM |

---

## 7. Vela Command Reference (GitHub / Colab)

If the model comes from a Colab `workspace/ColabTrain/vela/` path, ensure the Colab script uses:

```python
# Example for ethos-u55-256 (M55M1)
vela_model = vela.convert_model(
    input_model,
    accelerator_config="ethos-u55-256",
    system_config="Ethos_U55_High_End_Embedded",
    memory_mode="Shared_Sram",
)
```

Or the equivalent command-line arguments for your Vela version.

---

## 8. Files to Inspect

- **Model loading:** `main.cpp` → `PrepareModelToHyperRAM()`, `model.Init()`
- **Op resolver:** `Model/MouthDetectionModel.cpp` → `EnlistOperations()`
- **Arena size:** `KEIL/PoseLandmark.uvprojx` → `ACTIVATION_BUF_SZ=0x80000`
- **Scatter / memory:** `KEIL/M55M1.scatter` → SRAM01_HYPERRAM, activation_buf_sram
- **NPU init:** `NPU/ethosu_npu_init.c`
