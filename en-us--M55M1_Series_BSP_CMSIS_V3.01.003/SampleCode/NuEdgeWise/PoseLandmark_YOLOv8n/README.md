# PoseLandmark_YOLOv8n (Mouth Detection)
A demonstration sample for mouth detection using YOLO-Fastest v1.1 model.
Detects mouth open (Speaking) vs mouth closed (Closed).

## Requirement
1. Keil uVision5

## Howto
1. Build by Keil
2. Copy `yolo-fastest-1.1-int8_vela.tflite` file to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Run

## Model
- **Input:** 224×224 RGB, int8 = uint8 - 128
- **Output:** 2 tensors (stride 32: 7×7, stride 16: 14×14)
- **Classes:** 0 = mouth closed, 1 = mouth open (Speaking)
- **Vela:** Ethos-U55-256