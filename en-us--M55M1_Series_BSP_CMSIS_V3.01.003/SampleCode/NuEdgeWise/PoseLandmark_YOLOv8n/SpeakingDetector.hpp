/**************************************************************************//**
 * @file     SpeakingDetector.hpp
 * @brief    Modular speaking-detection API.
 *           Wraps face detection + face landmark keypoints + temporal MAR
 *           smoothing into three simple calls: Init / RunFrame / Draw.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef SPEAKING_DETECTOR_HPP
#define SPEAKING_DETECTOR_HPP

#include <stdint.h>

#define MAX_TRACKED_FACES 4
#define SPEAKING_MAX_LIP_LANDMARKS 24

struct SpeakingFaceResult {
    int  x, y, w, h;      /* face bounding box in frame coordinates */
    bool isSpeaking;       /* temporally-smoothed speaking state */
    bool rawMouthOpen;     /* raw mouth-open/motion signal this frame */
    int  lipCount;         /* number of valid lip landmarks for drawing */
    int  lipX[SPEAKING_MAX_LIP_LANDMARKS];
    int  lipY[SPEAKING_MAX_LIP_LANDMARKS];
    float mouthOpenNorm;   /* debug/tuning: dist(13,14) / dist(61,291) */
    float motionEnergy;    /* debug/tuning: short-window abs velocity */
    float onThreshold;     /* debug/tuning: active ON threshold */
    float offThreshold;    /* debug/tuning: active OFF threshold */
    float headMove;        /* debug/tuning: normalized bbox motion */
};

struct SpeakingDetectorConfig {
    float faceThreshold;       /* face detector confidence threshold (default 0.4) */
    float landmarkThreshold;   /* face landmark presence threshold (default 0.4) */
    float marVelocityOn;       /* motion-energy threshold to enter speaking */
    float marVelocityOff;      /* motion-energy threshold to release speaking */
    float marOn;               /* mouth-open threshold to enter speaking */
    float marOff;              /* mouth-open threshold to release speaking */
    int   confirmFrames;       /* consecutive above-threshold frames to switch on */
    int   releaseFrames;       /* consecutive below-threshold frames to switch off */
};

/**
 * Load the embedded face-detection and face-landmark models.
 * The caller must configure MPU regions (via InitPreDefMPURegion) before calling
 * this function; use SpeakingDetector_GetTensorArenas() to obtain the arena
 * addresses.
 * @param cfg  Configuration (pass NULL for defaults).
 * @return 0 on success, negative on error.
 */
int SpeakingDetector_Init(const SpeakingDetectorConfig *cfg);

/**
 * Run face detection + per-face landmark inference on one RGB565 camera frame.
 * Populates results[] with up to maxResults faces and their speaking state.
 * @return Number of faces found (0..maxResults).
 */
int SpeakingDetector_RunFrame(
    uint8_t *rgb565Data, int frameW, int frameH,
    SpeakingFaceResult results[], int maxResults);

/**
 * Draw bounding boxes, speaking labels, and lip keypoints onto an RGB565 frame.
 */
void SpeakingDetector_Draw(
    uint8_t *rgb565Data, int frameW, int frameH,
    const SpeakingFaceResult results[], int numResults);

/**
 * Retrieve the addresses and sizes of the internal tensor arenas so the caller
 * can include them in its MPU configuration. Call before SpeakingDetector_Init.
 */
void SpeakingDetector_GetTensorArenas(
    void **faceArena, uint32_t *faceArenaSize,
    void **landmarkArena, uint32_t *landmarkArenaSize);

#endif /* SPEAKING_DETECTOR_HPP */
