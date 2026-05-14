/**************************************************************************//**
 * @file     SpeakingDetector.cpp
 * @brief    Self-contained speaking-detection module.
 *           Runs embedded face detection + face landmark models, tracks faces
 *           across frames, and derives speaking state from lip motion.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "SpeakingDetector.hpp"

#include "BufAttributes.hpp"
#include "FaceDetectionModel.hpp"
#include "FaceDetectorPostProcessing.hpp"
#include "FaceDetectionResult.hpp"
#include "FaceLandmarkModel.hpp"
#include "FaceLandmarkPostProcessing.hpp"
#include "KeypointResult.hpp"
#include "log_macros.h"
#include "imlib.h"

#undef PI
#include "NuMicro.h"

#include <cmath>
#include <cstddef>
#include <stdio.h>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Internal constants                                                 */
/* ------------------------------------------------------------------ */
#define FACE_DETECTION_ACTIVATION_BUF_SZ  (460000)
#define FACE_LANDMARK_ACTIVATION_BUF_SZ   (460000)

#define DEFAULT_FACE_THRESHOLD            (0.4f)
#define DEFAULT_LANDMARK_THRESHOLD        (0.4f)

#define SPEAKING_DEBUG_OVERLAY               (0)

#define OPEN_MIN_THRESHOLD                   (0.11f)
#define CLOSE_MIN_THRESHOLD                  (0.08f)
#define MOTION_ENERGY_THRESHOLD_ON           (0.008f)
#define MOTION_ENERGY_THRESHOLD_OFF          (0.004f)
#define SPEAKING_SMOOTHING_FRAMES            (2)
#define SPEAKING_RELEASE_FRAMES              (4)
#define SPEAKING_MIN_DURATION_FRAMES         (5)
#define SPEAKING_DETECT_EVERY_N_FRAMES       (1)
#define DISPLAY_LANDMARK_SMOOTHING_ALPHA     (0.65f)
#define HEAD_MOVE_THRESHOLD                  (0.10f)
#define HEAD_SIZE_CHANGE_THRESHOLD           (0.12f)
#define MOTION_ENERGY_WINDOW                 (6)
#define ADAPTIVE_K_ON                        (5.0f)
#define ADAPTIVE_K_OFF                       (2.5f)
#define ADAPTIVE_THRESHOLD_MAX               (0.020f)
#define NOISE_EMA_ALPHA                      (0.08f)

#define LIP_OFFSET_X                         (4)
#define LIP_OFFSET_Y                         (3)

#define MOUTH_UPPER_INDEX                    (13)
#define MOUTH_LOWER_INDEX                    (14)
#define MOUTH_LEFT_INDEX                     (61)
#define MOUTH_RIGHT_INDEX                    (291)

static const int s_lipLandmarkIndices[SPEAKING_MAX_LIP_LANDMARKS] = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185,
    40, 39, 37, 0, 267, 269, 270, 409, 78, 308, 87, 14
};

/* ------------------------------------------------------------------ */
/*  Tensor arenas                                                      */
/* ------------------------------------------------------------------ */
ACTIVATION_BUF_ATTRIBUTE
static uint8_t s_arenaFace[FACE_DETECTION_ACTIVATION_BUF_SZ];

ACTIVATION_BUF_ATTRIBUTE
static uint8_t s_arenaLandmark[FACE_LANDMARK_ACTIVATION_BUF_SZ];

namespace {

struct PrevLipState {
    float lipX[SPEAKING_MAX_LIP_LANDMARKS];
    float lipY[SPEAKING_MAX_LIP_LANDMARKS];
    float prevMouthOpenNorm;
    int prevMouthValid;
    float mouthOpenWindow[MOTION_ENERGY_WINDOW];
    float motionWindow[MOTION_ENERGY_WINDOW];
    int windowIndex;
    int windowCount;
    float mouthOpenMean;
    float motionEnergy;
    float noiseMean;
    float noiseDev;
    int noiseValid;
    int x0, y0, w, h;
    int valid;
};

static arm::app::FaceDetectionModel s_faceModel;
static arm::app::FaceLandmarkModel s_landmarkModel;

static arm::app::FaceDetectorPostProcess *s_postFace = nullptr;
static arm::app::face_detection::PostProcessParams *s_fdParamsPtr = nullptr;
static arm::app::face_landmark::FaceLandmarkPostProcessing *s_postLandmark = nullptr;

static TfLiteTensor *s_faceInput = nullptr;
static TfLiteTensor *s_landmarkInput = nullptr;
static int s_faceCols = 0;
static int s_faceRows = 0;
static int s_landmarkCols = 0;
static int s_landmarkRows = 0;

static std::vector<arm::app::face_detection::DetectionResult> s_faceResults;
static std::vector<arm::app::face_landmark::KeypointResult> s_keypoints;

static PrevLipState s_prevLip[MAX_TRACKED_FACES];
static int s_speakingConfirmCount[MAX_TRACKED_FACES];
static int s_speakingReleaseCount[MAX_TRACKED_FACES];
static int s_speakingDurationCount[MAX_TRACKED_FACES];
static bool s_speaking[MAX_TRACKED_FACES];

static float s_displayRelX[SPEAKING_MAX_LIP_LANDMARKS];
static float s_displayRelY[SPEAKING_MAX_LIP_LANDMARKS];
static uint32_t s_speakingDetectFrameCount = 0;

static float s_faceThreshold = DEFAULT_FACE_THRESHOLD;
static float s_landmarkThreshold = DEFAULT_LANDMARK_THRESHOLD;
static float s_motionEnergyOn = MOTION_ENERGY_THRESHOLD_ON;
static float s_motionEnergyOff = MOTION_ENERGY_THRESHOLD_OFF;
static float s_mouthOpenOn = OPEN_MIN_THRESHOLD;
static float s_mouthOpenOff = CLOSE_MIN_THRESHOLD;
static int s_confirmFrames = SPEAKING_SMOOTHING_FRAMES;
static int s_releaseFrames = SPEAKING_RELEASE_FRAMES;

static void ResetTrackState(int trackId)
{
    if (trackId < 0 || trackId >= MAX_TRACKED_FACES) return;
    s_prevLip[trackId] = PrevLipState{};
}

static void InitTrackingState()
{
    for (int i = 0; i < MAX_TRACKED_FACES; i++) {
        ResetTrackState(i);
        s_speakingConfirmCount[i] = 0;
        s_speakingReleaseCount[i] = 0;
        s_speakingDurationCount[i] = 0;
        s_speaking[i] = false;
    }
    s_speakingDetectFrameCount = 0;
}

static float Distance2D(
    const std::vector<arm::app::face_landmark::KeypointResult> &keypoints,
    int idxA,
    int idxB)
{
    if (idxA >= (int)keypoints.size() || idxB >= (int)keypoints.size()) return 0.0f;

    float dx = (float)(keypoints[idxA].m_x - keypoints[idxB].m_x);
    float dy = (float)(keypoints[idxA].m_y - keypoints[idxB].m_y);
    return std::sqrt(dx * dx + dy * dy);
}

static float ComputeMouthOpenNorm(
    const std::vector<arm::app::face_landmark::KeypointResult> &keypoints)
{
    if (keypoints.size() < 468) return 0.0f;

    float mouthOpen = Distance2D(keypoints, MOUTH_UPPER_INDEX, MOUTH_LOWER_INDEX);
    float mouthWidth = Distance2D(keypoints, MOUTH_LEFT_INDEX, MOUTH_RIGHT_INDEX);
    if (mouthWidth < 1.0f) return 0.0f;
    return mouthOpen / mouthWidth;
}

static float MeanWindow(const float *values, int count)
{
    if (count <= 0) return 0.0f;

    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / (float)count;
}

static void UpdateMotionWindow(
    int trackId,
    float mouthOpenNorm,
    float *outMouthOpenMean,
    float *outMotionEnergy)
{
    PrevLipState *state = &s_prevLip[trackId];
    float velocity = 0.0f;

    if (state->prevMouthValid) {
        velocity = std::fabs(mouthOpenNorm - state->prevMouthOpenNorm);
    }

    state->mouthOpenWindow[state->windowIndex] = mouthOpenNorm;
    state->motionWindow[state->windowIndex] = velocity;
    state->windowIndex = (state->windowIndex + 1) % MOTION_ENERGY_WINDOW;
    if (state->windowCount < MOTION_ENERGY_WINDOW) {
        state->windowCount++;
    }

    state->prevMouthOpenNorm = mouthOpenNorm;
    state->prevMouthValid = 1;

    state->mouthOpenMean = MeanWindow(state->mouthOpenWindow, state->windowCount);
    state->motionEnergy = MeanWindow(state->motionWindow, state->windowCount);

    *outMouthOpenMean = state->mouthOpenMean;
    *outMotionEnergy = state->motionEnergy;
}

static void UpdateNoiseBaseline(int trackId, float motionEnergy)
{
    PrevLipState *state = &s_prevLip[trackId];

    if (!state->noiseValid) {
        state->noiseMean = motionEnergy;
        state->noiseDev = 0.0f;
        state->noiseValid = 1;
        return;
    }

    float diff = std::fabs(motionEnergy - state->noiseMean);
    state->noiseMean = (NOISE_EMA_ALPHA * motionEnergy) +
        ((1.0f - NOISE_EMA_ALPHA) * state->noiseMean);
    state->noiseDev = (NOISE_EMA_ALPHA * diff) +
        ((1.0f - NOISE_EMA_ALPHA) * state->noiseDev);
}

static float ClampThreshold(float value, float minValue)
{
    if (value < minValue) return minValue;
    if (value > ADAPTIVE_THRESHOLD_MAX) return ADAPTIVE_THRESHOLD_MAX;
    return value;
}

static void GetAdaptiveThresholds(int trackId, float *outOn, float *outOff)
{
    PrevLipState *state = &s_prevLip[trackId];

    float adaptiveOn = s_motionEnergyOn;
    float adaptiveOff = s_motionEnergyOff;
    if (state->noiseValid) {
        adaptiveOn = state->noiseMean + (ADAPTIVE_K_ON * state->noiseDev);
        adaptiveOff = state->noiseMean + (ADAPTIVE_K_OFF * state->noiseDev);
    }

    *outOn = ClampThreshold(adaptiveOn, s_motionEnergyOn);
    *outOff = ClampThreshold(adaptiveOff, s_motionEnergyOff);
}

static void ApplyDisplayLipSmoothing(
    const std::vector<arm::app::face_landmark::KeypointResult> &keypoints,
    int trackId, int faceW, int faceH,
    float *outRelX, float *outRelY)
{
    for (int i = 0; i < SPEAKING_MAX_LIP_LANDMARKS; i++) {
        int idx = s_lipLandmarkIndices[i];
        float rawRelX = 0.0f;
        float rawRelY = 0.0f;

        if (idx < (int)keypoints.size() && faceW > 0 && faceH > 0) {
            rawRelX = (float)keypoints[idx].m_x / (float)faceW;
            rawRelY = (float)keypoints[idx].m_y / (float)faceH;
        }

        if (trackId >= 0 && trackId < MAX_TRACKED_FACES && s_prevLip[trackId].valid) {
            outRelX[i] = DISPLAY_LANDMARK_SMOOTHING_ALPHA * rawRelX +
                (1.0f - DISPLAY_LANDMARK_SMOOTHING_ALPHA) * s_prevLip[trackId].lipX[i];
            outRelY[i] = DISPLAY_LANDMARK_SMOOTHING_ALPHA * rawRelY +
                (1.0f - DISPLAY_LANDMARK_SMOOTHING_ALPHA) * s_prevLip[trackId].lipY[i];
        } else {
            outRelX[i] = rawRelX;
            outRelY[i] = rawRelY;
        }
    }
}

static int FindMatchingPrevFace(int curX0, int curY0, int curW, int curH)
{
    int bestIdx = -1;
    float bestDist = 1e9f;
    int curCx = curX0 + curW / 2;
    int curCy = curY0 + curH / 2;

    for (int j = 0; j < MAX_TRACKED_FACES; j++) {
        if (!s_prevLip[j].valid) continue;
        int prevCx = s_prevLip[j].x0 + s_prevLip[j].w / 2;
        int prevCy = s_prevLip[j].y0 + s_prevLip[j].h / 2;
        float dx = (float)(curCx - prevCx);
        float dy = (float)(curCy - prevCy);
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < bestDist && dist < (float)(curW + curH)) {
            bestDist = dist;
            bestIdx = j;
        }
    }

    return bestIdx;
}

static int AllocateTrack(int faceIdx)
{
    for (int j = 0; j < MAX_TRACKED_FACES; j++) {
        if (!s_prevLip[j].valid) return j;
    }
    return faceIdx % MAX_TRACKED_FACES;
}

static float HeadMoveAmount(int trackId, int curX0, int curY0, int curW, int curH)
{
    if (trackId < 0 || trackId >= MAX_TRACKED_FACES || !s_prevLip[trackId].valid ||
        curW <= 0 || curH <= 0) {
        return 0.0f;
    }

    int prevCx = s_prevLip[trackId].x0 + s_prevLip[trackId].w / 2;
    int prevCy = s_prevLip[trackId].y0 + s_prevLip[trackId].h / 2;
    int curCx = curX0 + curW / 2;
    int curCy = curY0 + curH / 2;
    float dx = (float)(curCx - prevCx) / (float)curW;
    float dy = (float)(curCy - prevCy) / (float)curH;
    return std::sqrt(dx * dx + dy * dy);
}

static int HeadUnstable(int trackId, int curX0, int curY0, int curW, int curH)
{
    if (HeadMoveAmount(trackId, curX0, curY0, curW, curH) > HEAD_MOVE_THRESHOLD) {
        return 1;
    }
    if (trackId < 0 || trackId >= MAX_TRACKED_FACES || !s_prevLip[trackId].valid ||
        curW <= 0 || curH <= 0) {
        return 0;
    }

    int prevW = s_prevLip[trackId].w;
    int prevH = s_prevLip[trackId].h;
    float sizeChangeW = (prevW > 0) ? std::fabs((float)(curW - prevW) / (float)prevW) : 0.0f;
    float sizeChangeH = (prevH > 0) ? std::fabs((float)(curH - prevH) / (float)prevH) : 0.0f;
    return (sizeChangeW > HEAD_SIZE_CHANGE_THRESHOLD ||
            sizeChangeH > HEAD_SIZE_CHANGE_THRESHOLD) ? 1 : 0;
}

static void StoreLipState(
    const float *displayRelX, const float *displayRelY,
    int trackId, int faceX0, int faceY0, int faceW, int faceH)
{
    if (trackId < 0 || trackId >= MAX_TRACKED_FACES) return;

    PrevLipState *state = &s_prevLip[trackId];
    for (int i = 0; i < SPEAKING_MAX_LIP_LANDMARKS; i++) {
        state->lipX[i] = displayRelX[i];
        state->lipY[i] = displayRelY[i];
    }

    state->x0 = faceX0;
    state->y0 = faceY0;
    state->w = faceW;
    state->h = faceH;
    state->valid = 1;
}

static void FillLipDrawPoints(
    SpeakingFaceResult *result,
    const float *displayRelX,
    const float *displayRelY,
    int faceX0,
    int faceY0,
    int faceW,
    int faceH)
{
    result->lipCount = SPEAKING_MAX_LIP_LANDMARKS;
    for (int i = 0; i < SPEAKING_MAX_LIP_LANDMARKS; i++) {
        result->lipX[i] = faceX0 + (int)(displayRelX[i] * faceW) + LIP_OFFSET_X;
        result->lipY[i] = faceY0 + (int)(displayRelY[i] * faceH) + LIP_OFFSET_Y;
    }
}

static void ExpandFaceBoxes(int frameW, int frameH)
{
    const float scale = 1.4f;

    for (size_t i = 0; i < s_faceResults.size(); i++) {
        arm::app::face_detection::DetectionResult *faceBox = &s_faceResults[i];
        float scaledH = scale * faceBox->m_h;
        float scaledW = scaledH;
        int newW = (int)scaledW;
        int newH = (int)scaledH;
        int newX = faceBox->m_x0 - (int)((scaledW - faceBox->m_w) / 2.0f);
        int newY = faceBox->m_y0 - (int)((scaledH - faceBox->m_h) / 2.0f);

        if (newX < 0) newX = 0;
        if (newY < 0) newY = 0;
        if (newX + newW >= frameW) newW = frameW - newX;
        if (newY + newH >= frameH) newH = frameH - newY;

        faceBox->m_x0 = newX;
        faceBox->m_y0 = newY;
        faceBox->m_w = newW;
        faceBox->m_h = newH;
    }
}

static void PopulateDefaults(SpeakingFaceResult *result, const arm::app::face_detection::DetectionResult &faceBox)
{
    result->x = faceBox.m_x0;
    result->y = faceBox.m_y0;
    result->w = faceBox.m_w;
    result->h = faceBox.m_h;
    result->isSpeaking = false;
    result->rawMouthOpen = false;
    result->lipCount = 0;
    result->mouthOpenNorm = 0.0f;
    result->motionEnergy = 0.0f;
    result->onThreshold = 0.0f;
    result->offThreshold = 0.0f;
    result->headMove = 0.0f;
}

} /* anonymous namespace */

/* ================================================================== */
/*  PUBLIC API                                                         */
/* ================================================================== */

int SpeakingDetector_Init(const SpeakingDetectorConfig *cfg)
{
    s_faceThreshold = DEFAULT_FACE_THRESHOLD;
    s_landmarkThreshold = DEFAULT_LANDMARK_THRESHOLD;
    s_motionEnergyOn = MOTION_ENERGY_THRESHOLD_ON;
    s_motionEnergyOff = MOTION_ENERGY_THRESHOLD_OFF;
    s_mouthOpenOn = OPEN_MIN_THRESHOLD;
    s_mouthOpenOff = CLOSE_MIN_THRESHOLD;
    s_confirmFrames = SPEAKING_SMOOTHING_FRAMES;
    s_releaseFrames = SPEAKING_RELEASE_FRAMES;

    if (cfg) {
        s_faceThreshold = cfg->faceThreshold;
        s_landmarkThreshold = cfg->landmarkThreshold;
        s_motionEnergyOn = cfg->marVelocityOn;
        s_motionEnergyOff = cfg->marVelocityOff;
        s_mouthOpenOn = cfg->marOn;
        s_mouthOpenOff = cfg->marOff;
        s_confirmFrames = cfg->confirmFrames;
        s_releaseFrames = cfg->releaseFrames;
    }

    if (!s_faceModel.Init(s_arenaFace, sizeof(s_arenaFace),
            (unsigned char *)arm::app::face_detection::GetModelPointer(),
            arm::app::face_detection::GetModelLen())) {
        printf_err("SpeakingDetector: face model init failed\n");
        return -1;
    }
    info("SpeakingDetector: face model OK\n");

    if (!s_landmarkModel.Init(s_arenaLandmark, sizeof(s_arenaLandmark),
            (unsigned char *)arm::app::face_landmark::GetModelPointer(),
            arm::app::face_landmark::GetModelLen())) {
        printf_err("SpeakingDetector: face landmark model init failed\n");
        return -2;
    }
    info("SpeakingDetector: face landmark model OK\n");

    TfLiteIntArray *faceShape = s_faceModel.GetInputShape(0);
    s_faceCols = faceShape->data[arm::app::FaceDetectionModel::ms_inputColsIdx];
    s_faceRows = faceShape->data[arm::app::FaceDetectionModel::ms_inputRowsIdx];
    s_faceInput = s_faceModel.GetInputTensor(0);

    TfLiteIntArray *landmarkShape = s_landmarkModel.GetInputShape(0);
    s_landmarkCols = landmarkShape->data[arm::app::FaceLandmarkModel::ms_inputColsIdx];
    s_landmarkRows = landmarkShape->data[arm::app::FaceLandmarkModel::ms_inputRowsIdx];
    s_landmarkInput = s_landmarkModel.GetInputTensor(0);

    if (!s_faceInput || !s_landmarkInput) {
        printf_err("SpeakingDetector: missing input tensor\n");
        return -3;
    }

    TfLiteTensor *fdOut0 = s_faceModel.GetOutputTensor(0);
    TfLiteTensor *fdOut1 = s_faceModel.GetOutputTensor(1);

    static arm::app::face_detection::PostProcessParams s_fdParams;
    s_fdParams.inputImgRows = s_faceRows;
    s_fdParams.inputImgCols = s_faceCols;
    s_fdParams.originalImageRows = 0;
    s_fdParams.originalImageCols = 0;
    s_fdParams.anchor1 = anchor1;
    s_fdParams.anchor2 = anchor2;
    s_fdParams.threshold = s_faceThreshold;
    s_fdParams.nms = 0.45f;
    s_fdParams.numClasses = 1;
    s_fdParams.topN = 0;
    s_fdParamsPtr = &s_fdParams;

    static arm::app::FaceDetectorPostProcess s_fdPP(fdOut0, fdOut1, s_faceResults, s_fdParams);
    s_postFace = &s_fdPP;

    static arm::app::face_landmark::FaceLandmarkPostProcessing s_flPP(DEFAULT_LANDMARK_THRESHOLD);
    s_flPP = arm::app::face_landmark::FaceLandmarkPostProcessing(s_landmarkThreshold);
    s_postLandmark = &s_flPP;

    InitTrackingState();

    info("SpeakingDetector: init complete (face %dx%d, landmark %dx%d)\n",
         s_faceRows, s_faceCols, s_landmarkRows, s_landmarkCols);
    return 0;
}

int SpeakingDetector_RunFrame(
    uint8_t *rgb565Data, int frameW, int frameH,
    SpeakingFaceResult results[], int maxResults)
{
    if (!s_postFace || !s_postLandmark || !results || maxResults <= 0) return 0;

    image_t srcImg;
    srcImg.w = frameW;
    srcImg.h = frameH;
    srcImg.data = rgb565Data;
    srcImg.pixfmt = PIXFORMAT_RGB565;

    rectangle_t roi;

    s_faceResults.clear();
    s_fdParamsPtr->originalImageRows = frameH;
    s_fdParamsPtr->originalImageCols = frameW;

    roi.x = 0;
    roi.y = 0;
    roi.w = frameW;
    roi.h = frameH;

    image_t faceResized;
    faceResized.w = s_faceCols;
    faceResized.h = s_faceRows;
    faceResized.data = (uint8_t *)s_faceInput->data.data;
    faceResized.pixfmt = PIXFORMAT_GRAYSCALE;
    imlib_nvt_scale(&srcImg, &faceResized, &roi);

    {
        uint8_t *u = (uint8_t *)s_faceInput->data.data;
        int8_t *s = (int8_t *)s_faceInput->data.data;
        for (size_t n = 0; n < s_faceInput->bytes; n++) {
            s[n] = (int8_t)((int32_t)u[n] - 128);
        }
    }

    s_faceModel.RunInference();
    s_postFace->RunPostProcess(s_faceResults);
    ExpandFaceBoxes(frameW, frameH);

    int numFaces = (int)s_faceResults.size();
    if (numFaces > maxResults) numFaces = maxResults;
    if (numFaces > MAX_TRACKED_FACES) numFaces = MAX_TRACKED_FACES;

    s_speakingDetectFrameCount++;
    const int runDetectionThisFrame =
        (s_speakingDetectFrameCount % SPEAKING_DETECT_EVERY_N_FRAMES) == 0;

    for (int i = 0; i < numFaces; i++) {
        const arm::app::face_detection::DetectionResult &faceBox = s_faceResults[i];
        PopulateDefaults(&results[i], faceBox);

        int matchedPrev = FindMatchingPrevFace(faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);
        if (matchedPrev < 0 && numFaces == 1) {
            matchedPrev = s_prevLip[0].valid ? 0 : -1;
        }

        int trackId = (matchedPrev >= 0) ? matchedPrev : AllocateTrack(i);
        if (matchedPrev < 0 && trackId < MAX_TRACKED_FACES) {
            ResetTrackState(trackId);
            s_speaking[trackId] = false;
            s_speakingConfirmCount[trackId] = 0;
            s_speakingReleaseCount[trackId] = 0;
            s_speakingDurationCount[trackId] = 0;
        }

        roi.x = faceBox.m_x0;
        roi.y = faceBox.m_y0;
        roi.w = faceBox.m_w;
        roi.h = faceBox.m_h;

        if (roi.w <= 0 || roi.h <= 0) {
            continue;
        }

        image_t landmarkResized;
        landmarkResized.w = s_landmarkCols;
        landmarkResized.h = s_landmarkRows;
        landmarkResized.data = (uint8_t *)s_landmarkInput->data.data;
        landmarkResized.pixfmt = PIXFORMAT_RGB888;
        imlib_nvt_scale(&srcImg, &landmarkResized, &roi);

        {
            uint8_t *u = (uint8_t *)s_landmarkInput->data.data;
            int8_t *s = (int8_t *)s_landmarkInput->data.data;
            for (size_t n = 0; n < s_landmarkInput->bytes; n++) {
                s[n] = (int8_t)((int32_t)u[n] - 128);
            }
        }

        s_landmarkModel.RunInference();

        TfLiteTensor *meshTensor = s_landmarkModel.GetOutputTensor(FACE_LANDMARK_MESH_TENSOR_INDEX);
#if defined(FACE_LANDMARK_LEFT_IRIS_TENSOR_INDEX)
        TfLiteTensor *leftIrisTensor = s_landmarkModel.GetOutputTensor(FACE_LANDMARK_LEFT_IRIS_TENSOR_INDEX);
#else
        TfLiteTensor *leftIrisTensor = nullptr;
#endif
#if defined(FACE_LANDMARK_RIGHT_IRIS_TENSOR_INDEX)
        TfLiteTensor *rightIrisTensor = s_landmarkModel.GetOutputTensor(FACE_LANDMARK_RIGHT_IRIS_TENSOR_INDEX);
#else
        TfLiteTensor *rightIrisTensor = nullptr;
#endif
        TfLiteTensor *presenceTensor = s_landmarkModel.GetOutputTensor(FACE_LANDMARK_FACE_FLAG_TENSOR_INDEX);

        s_postLandmark->RunPostProcessing(
            s_landmarkRows,
            s_landmarkCols,
            (uint32_t)roi.h,
            (uint32_t)roi.w,
            meshTensor,
            leftIrisTensor,
            rightIrisTensor,
            presenceTensor,
            s_keypoints);

        if (trackId < MAX_TRACKED_FACES && s_keypoints.size() >= 468) {
            ApplyDisplayLipSmoothing(s_keypoints, trackId, roi.w, roi.h, s_displayRelX, s_displayRelY);

            if (runDetectionThisFrame) {
                float mouthOpenNorm = ComputeMouthOpenNorm(s_keypoints);
                float mouthOpenMean = 0.0f;
                float motionEnergy = 0.0f;
                float thresholdOn = 0.0f;
                float thresholdOff = 0.0f;
                float headMove = HeadMoveAmount(trackId, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);
                int headUnstable = HeadUnstable(trackId, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);

                UpdateMotionWindow(trackId, mouthOpenNorm, &mouthOpenMean, &motionEnergy);
                if (!s_speaking[trackId] &&
                    mouthOpenMean < s_mouthOpenOff &&
                    mouthOpenNorm < s_mouthOpenOff) {
                    UpdateNoiseBaseline(trackId, motionEnergy);
                }
                GetAdaptiveThresholds(trackId, &thresholdOn, &thresholdOff);

                int signalAboveOn = !headUnstable &&
                    (motionEnergy > thresholdOn) &&
                    ((mouthOpenNorm > s_mouthOpenOn) || (mouthOpenMean > s_mouthOpenOn));
                int signalBelowOff =
                    (motionEnergy < thresholdOff) ||
                    ((mouthOpenNorm < s_mouthOpenOff) && (mouthOpenMean < s_mouthOpenOff));

                results[i].rawMouthOpen = signalAboveOn ? true : false;
                results[i].mouthOpenNorm = mouthOpenMean;
                results[i].motionEnergy = motionEnergy;
                results[i].onThreshold = thresholdOn;
                results[i].offThreshold = thresholdOff;
                results[i].headMove = headMove;

                if (signalAboveOn) {
                    s_speakingConfirmCount[trackId] =
                        (s_speakingConfirmCount[trackId] < s_confirmFrames) ?
                        s_speakingConfirmCount[trackId] + 1 : s_confirmFrames;
                    s_speakingReleaseCount[trackId] = 0;
                    if (s_speakingConfirmCount[trackId] >= s_confirmFrames) {
                        s_speaking[trackId] = true;
                    }
                } else if (s_speaking[trackId]) {
                    s_speakingDurationCount[trackId]++;
                    if (signalBelowOff) {
                        s_speakingReleaseCount[trackId]++;
                        if (s_speakingReleaseCount[trackId] >= s_releaseFrames &&
                            s_speakingDurationCount[trackId] >= SPEAKING_MIN_DURATION_FRAMES) {
                            s_speaking[trackId] = false;
                        }
                    } else {
                        s_speakingReleaseCount[trackId] = 0;
                    }
                    s_speakingConfirmCount[trackId] = 0;
                } else {
                    s_speakingReleaseCount[trackId] = 0;
                    s_speakingConfirmCount[trackId] = 0;
                    s_speakingDurationCount[trackId] = 0;
                }

                if (!s_speaking[trackId]) {
                    s_speakingDurationCount[trackId] = 0;
                }

                StoreLipState(
                    s_displayRelX, s_displayRelY,
                    trackId, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);
            }

            results[i].isSpeaking = s_speaking[trackId];
            FillLipDrawPoints(
                &results[i],
                s_displayRelX,
                s_displayRelY,
                faceBox.m_x0,
                faceBox.m_y0,
                faceBox.m_w,
                faceBox.m_h);
        } else if (trackId < MAX_TRACKED_FACES) {
            s_prevLip[trackId].valid = 0;
            results[i].isSpeaking = s_speaking[trackId];
        }
    }

    return numFaces;
}

void SpeakingDetector_Draw(
    uint8_t *rgb565Data, int frameW, int frameH,
    const SpeakingFaceResult results[], int numResults)
{
    image_t drawImg;
    drawImg.w = frameW;
    drawImg.h = frameH;
    drawImg.data = rgb565Data;
    drawImg.pixfmt = PIXFORMAT_RGB565;

    int green = COLOR_R5_G6_B5_TO_RGB565(0, COLOR_G6_MAX, 0);
    int blue = COLOR_B5_MAX;

    for (int i = 0; i < numResults; i++) {
        bool speaking = results[i].isSpeaking;
        int boxColor = speaking ? green : blue;
        const char *label = speaking ? "Speaking" : "Not Speaking";

        imlib_draw_rectangle(&drawImg,
            results[i].x, results[i].y,
            results[i].w, results[i].h,
            boxColor, 2, false);

        int labelY = (results[i].y - 14 > 0) ? (results[i].y - 14) : results[i].y;
        imlib_draw_string(&drawImg, results[i].x, labelY, label, boxColor,
                          2, 0, 0, false, false, false, false, 0, false, false);

        for (int k = 0; k < results[i].lipCount && k < SPEAKING_MAX_LIP_LANDMARKS; k++) {
            imlib_draw_circle(&drawImg, results[i].lipX[k], results[i].lipY[k],
                              2, green, 1, true);
        }

#if (SPEAKING_DEBUG_OVERLAY)
        {
            char dbg[80];
            snprintf(dbg, sizeof(dbg), "op %.2f me %.3f on %.3f off %.3f hm %.2f",
                     results[i].mouthOpenNorm,
                     results[i].motionEnergy,
                     results[i].onThreshold,
                     results[i].offThreshold,
                     results[i].headMove);
            imlib_draw_string(&drawImg, results[i].x, results[i].y + results[i].h + 2,
                              dbg, boxColor, 1, 0, 0, false, false, false, false,
                              0, false, false);
        }
#endif
    }
}

void SpeakingDetector_GetTensorArenas(
    void **faceArena, uint32_t *faceArenaSize,
    void **landmarkArena, uint32_t *landmarkArenaSize)
{
    *faceArena = s_arenaFace;
    *faceArenaSize = sizeof(s_arenaFace);
    *landmarkArena = s_arenaLandmark;
    *landmarkArenaSize = sizeof(s_arenaLandmark);
}
