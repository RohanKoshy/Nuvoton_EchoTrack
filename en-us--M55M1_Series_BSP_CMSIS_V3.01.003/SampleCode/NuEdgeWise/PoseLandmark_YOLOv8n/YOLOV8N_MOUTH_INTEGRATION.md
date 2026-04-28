# YOLOv8n Mouth Model Integration

Integration is complete for the new YOLOv8n ReLU6 mouth detection model (`best_full_integer_quant_vela.tflite`).

## Summary of Changes

### 1. MouthDetectionModel
- **Op resolver:** Switched from YOLO-Fastest ops to YOLOv8n: only `AddTranspose` + `AddEthosU`
- **Removed:** mouth_anchor1, mouth_anchor2 (YOLOv8 is anchor-free)

### 2. New MouthYOLOv8PostProcessing
- **6 output tensors:** box P3(0), box P4(1), box P5(2), cls P3(3), cls P4(4), cls P5(5)
- **DFL decoding:** reg_max=16, 64 values per box (4×16)
- **Grid sizes:** 24×24 (stride 8), 12×12 (stride 16), 6×6 (stride 32)
- **Confidence threshold:** 0.25
- **NMS IoU:** 0.45

### 3. main.cpp
- **Model file:** `0:\best_full_integer_quant_vela.tflite` (change `MODEL_FILE` if you rename)
- **Thresholds:** conf=0.25, iou=0.45

### 4. Keil Project
- Replaced `FaceDetectorPostProcessing.cpp` with `MouthYOLOv8PostProcessing.cpp`

## What You Need To Do

1. **Copy model to SD card root**  
   Place `best_full_integer_quant_vela.tflite` at the root of a FAT32 SD card.

2. **Rename model (optional)**  
   If you use a different filename, update `MODEL_FILE` in `main.cpp`:
   ```cpp
   #define MODEL_FILE "0:\\your_model_name.tflite"
   ```

3. **Rebuild**  
   Build the PoseLandmark project in Keil.

4. **Run**  
   Flash the firmware, insert the SD card, and power on.

## Model Spec (Reference)

| Property | Value |
|----------|-------|
| Input | 192×192 RGB, int8 |
| Outputs | 6 tensors (box P3/P4/P5, cls P3/P4/P5) |
| Classes | mouth closed (0), mouth open (1) |
| Vela | ethos-u55-256, Shared_Sram, Ethos_U55_High_End_Embedded |
