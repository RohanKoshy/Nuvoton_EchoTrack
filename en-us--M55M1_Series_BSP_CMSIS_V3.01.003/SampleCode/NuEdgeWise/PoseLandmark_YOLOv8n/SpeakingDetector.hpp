/**************************************************************************//**
 * @file     SpeakingDetector.hpp
 * @brief    Modular speaking-detection API.
 *           Wraps face detection + YOLOv8n mouth model + per-face temporal
 *           smoothing into three simple calls: Init / RunFrame / Draw.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef SPEAKING_DETECTOR_HPP
#define SPEAKING_DETECTOR_HPP

#include <stdint.h>

#define MAX_TRACKED_FACES 4

struct SpeakingFaceResult {
    int  x, y, w, h;      /* face bounding box in frame coordinates */
    bool isSpeaking;       /* temporally-smoothed speaking state */
    bool rawMouthOpen;     /* raw (unsmoothed) mouth-open flag this frame */
};

struct SpeakingDetectorConfig {
    float mouthThreshold;  /* mouth detection confidence threshold (default 0.05) */
    float nmsThreshold;    /* mouth NMS IoU threshold              (default 0.45) */
    float faceThreshold;   /* face presence threshold              (default 0.4)  */
    int   hysteresisOn;    /* consecutive "open" frames to switch to Speaking     */
    int   hysteresisOff;   /* consecutive "closed" frames to switch to Not Speaking */
};

/**
 * Load the embedded face-detection model and the mouth model from SD/HyperRAM.
 * Sets up tensor arenas and post-processing.  The caller must configure MPU
 * regions (via InitPreDefMPURegion) before calling this function — use
 * SpeakingDetector_GetTensorArenas() to obtain the arena addresses.
 * @param cfg  Configuration (pass NULL for defaults).
 * @return 0 on success, negative on error.
 */
int SpeakingDetector_Init(const SpeakingDetectorConfig *cfg);

/**
 * Run face detection + per-face mouth inference on one RGB565 camera frame.
 * Populates results[] with up to maxResults faces and their speaking state.
 * @return Number of faces found (0..maxResults).
 */
int SpeakingDetector_RunFrame(
    uint8_t *rgb565Data, int frameW, int frameH,
    SpeakingFaceResult results[], int maxResults);

/**
 * Draw bounding boxes and "Speaking" / "Not Speaking" labels onto an RGB565
 * frame buffer.  This is optional — skip it if you only need the state for
 * hardware control.
 */
void SpeakingDetector_Draw(
    uint8_t *rgb565Data, int frameW, int frameH,
    const SpeakingFaceResult results[], int numResults);

/**
 * Retrieve the addresses and sizes of the internal tensor arenas so the
 * caller can include them in its MPU configuration.  Call BEFORE
 * SpeakingDetector_Init if you need to set up MPU in your own main().
 */
void SpeakingDetector_GetTensorArenas(
    void **faceArena, uint32_t *faceArenaSize,
    void **mouthArena, uint32_t *mouthArenaSize);

#endif /* SPEAKING_DETECTOR_HPP */
