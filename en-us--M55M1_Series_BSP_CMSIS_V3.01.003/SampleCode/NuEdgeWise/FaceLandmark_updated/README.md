# FaceLandmark
A demonstration sample for face landmark with optional **speaking detection**.
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Run
## Speaking Detection (optional)
Uses **lip-relative** MAR (Mouth Aspect Ratio) and MAR velocity to detect who is speaking:
- **Lip landmarks**: 24 lip contour points + chin for jaw-opening cue
- **Detection**: Speaking when MAR velocity > threshold (mouth changing) AND MAR > threshold (mouth open)
- **Lip-relative**: MAR and MAR velocity are pure mouth geometry—head movement does not trigger false positives
- **Visual feedback**: Bounding box turns **green** when speaking; "Speaking" label appears
- **Hysteresis**: Speaking state persists for a few frames after movement stops (reduces flicker)
- **Tuning in main.cpp**: `SPEAKING_MAR_VELOCITY_THRESHOLD_ON/OFF`, `SPEAKING_MAR_THRESHOLD_ON/OFF`, `SPEAKING_RELEASE_FRAMES`
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|Yolo fastest|192x192x1|441|443|131.1|
|FaceLandmark|192x192x3|679|460|38.6|

Total frame rate: 13 fps


