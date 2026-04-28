/**************************************************************************//**
 * @file     SpeakingDetector.cpp
 * @brief    Self-contained speaking-detection module.
 *           Loads face-detection (embedded) + mouth YOLOv8n (SD/HyperRAM),
 *           runs per-face inference, tracks faces across frames, and applies
 *           temporal smoothing to produce a stable Speaking / Not Speaking state.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "SpeakingDetector.hpp"

#include "BufAttributes.hpp"
#include "MouthDetectionModel.hpp"
#include "MouthYOLOv8PostProcessing.hpp"
#include "FaceDetectionModel.hpp"
#include "FaceDetectorPostProcessing.hpp"
#include "FaceDetectionResult.hpp"
#include "ModelFileReader.h"
#include "log_macros.h"
#include "imlib.h"
#include "ff.h"

#undef PI
#include "NuMicro.h"

#include <vector>
#include <cstring>

/* ------------------------------------------------------------------ */
/*  Internal constants                                                 */
/* ------------------------------------------------------------------ */
#define MODEL_AT_HYPERRAM_ADDR  (0x82400000)
#define MODEL_FILE              "0:\\best_full_integer_quant_vela.tflite"
#define EACH_READ_SIZE          512

#define FACE_DETECTION_ACTIVATION_BUF_SZ  (460000)
#define MOUTH_ACTIVATION_BUF_SZ           (512 * 1024)

/* ------------------------------------------------------------------ */
/*  Tensor arenas (file-scope, outside anonymous namespace so the      */
/*  GetTensorArenas helper can reference them)                         */
/* ------------------------------------------------------------------ */
ACTIVATION_BUF_ATTRIBUTE
static uint8_t s_arenaFace[FACE_DETECTION_ACTIVATION_BUF_SZ];

ACTIVATION_BUF_ATTRIBUTE
static uint8_t s_arenaMouth[MOUTH_ACTIVATION_BUF_SZ];

/* ------------------------------------------------------------------ */
/*  Remaining internal state (anonymous namespace)                     */
/* ------------------------------------------------------------------ */
namespace {

static arm::app::FaceDetectionModel     s_faceModel;
static arm::app::MouthDetectionModel     s_mouthModel;

static arm::app::FaceDetectorPostProcess *s_postFace  = nullptr;
static arm::app::face_detection::PostProcessParams *s_fdParamsPtr = nullptr;
static arm::app::mouth_detection::MouthYOLOv8PostProcessing *s_postMouth = nullptr;

static std::vector<arm::app::face_detection::DetectionResult> s_faceResults;
static std::vector<arm::app::face_detection::DetectionResult> s_mouthTemp;

static TfLiteTensor   *s_mouthInput     = nullptr;
static TfLiteTensor   *s_faceInput      = nullptr;
static int             s_mouthCols      = 0;
static int             s_mouthRows      = 0;
static int             s_faceCols       = 0;
static int             s_faceRows       = 0;

/* ------------------------------------------------------------------ */
/*  Face tracker state                                                 */
/* ------------------------------------------------------------------ */
struct TrackedFace {
    float cx, cy;
    bool  isSpeaking;
    int   counter;
    int   missedFrames;
};

static TrackedFace s_tracked[MAX_TRACKED_FACES];
static int  s_hysteresisOn  = 3;
static int  s_hysteresisOff = 4;

static void InitTrackedFaces()
{
    int i;
    for (i = 0; i < MAX_TRACKED_FACES; i++) {
        s_tracked[i].cx  = -1.f;
        s_tracked[i].cy  = -1.f;
        s_tracked[i].isSpeaking  = false;
        s_tracked[i].counter     = 0;
        s_tracked[i].missedFrames = 99;
    }
}

static int MatchOrCreateSlot(float cx, float cy)
{
    int bestSlot = -1;
    float bestDist = 9999999.f;
    int i;

    for (i = 0; i < MAX_TRACKED_FACES; i++) {
        if (s_tracked[i].missedFrames < 10) {
            float dx = s_tracked[i].cx - cx;
            float dy = s_tracked[i].cy - cy;
            float dist = dx * dx + dy * dy;
            if (dist < bestDist) {
                bestDist = dist;
                bestSlot = i;
            }
        }
    }

    if (bestSlot >= 0 && bestDist < (80.f * 80.f))
        return bestSlot;

    int oldestSlot = 0;
    int oldestMissed = -1;
    for (i = 0; i < MAX_TRACKED_FACES; i++) {
        if (s_tracked[i].missedFrames >= 10) {
            s_tracked[i].isSpeaking = false;
            s_tracked[i].counter = 0;
            return i;
        }
        if (s_tracked[i].missedFrames > oldestMissed) {
            oldestMissed = s_tracked[i].missedFrames;
            oldestSlot = i;
        }
    }
    s_tracked[oldestSlot].isSpeaking = false;
    s_tracked[oldestSlot].counter = 0;
    return oldestSlot;
}

static bool SmoothPerFace(int slot, bool rawMouthOpen)
{
    if (rawMouthOpen) {
        if (!s_tracked[slot].isSpeaking) {
            s_tracked[slot].counter++;
            if (s_tracked[slot].counter >= s_hysteresisOn) {
                s_tracked[slot].isSpeaking = true;
                s_tracked[slot].counter = 0;
            }
        } else {
            s_tracked[slot].counter = 0;
        }
    } else {
        if (s_tracked[slot].isSpeaking) {
            s_tracked[slot].counter++;
            if (s_tracked[slot].counter >= s_hysteresisOff) {
                s_tracked[slot].isSpeaking = false;
                s_tracked[slot].counter = 0;
            }
        } else {
            s_tracked[slot].counter = 0;
        }
    }
    return s_tracked[slot].isSpeaking;
}

/* ------------------------------------------------------------------ */
/*  SD → HyperRAM model loader                                        */
/* ------------------------------------------------------------------ */
static int32_t PrepareModelToHyperRAM(void)
{
    TCHAR sd_path[] = { '0', ':', 0 };
    f_chdrive(sd_path);

    int32_t i32FileSize;
    int32_t i32FileReadIndex = 0;
    int32_t i32Read;

    if (!ModelFileReader_Initialize(MODEL_FILE)) {
        printf_err("Unable open model %s\n", MODEL_FILE);
        return -1;
    }

    i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

    while (i32FileReadIndex < i32FileSize) {
        i32Read = ModelFileReader_ReadData(
            (BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
        if (i32Read < 0) break;
        i32FileReadIndex += i32Read;
    }

    if (i32FileReadIndex < i32FileSize) {
        printf_err("Read Model file size is not enough\n");
        return -2;
    }

    ModelFileReader_Finish();
    return i32FileSize;
}

} /* anonymous namespace */

/* ================================================================== */
/*  PUBLIC API                                                         */
/* ================================================================== */

int SpeakingDetector_Init(const SpeakingDetectorConfig *cfg)
{
    float mouthTh = 0.05f;
    float nmsTh   = 0.45f;
    float faceTh  = 0.4f;
    s_hysteresisOn  = 3;
    s_hysteresisOff = 4;

    if (cfg) {
        mouthTh         = cfg->mouthThreshold;
        nmsTh           = cfg->nmsThreshold;
        faceTh          = cfg->faceThreshold;
        s_hysteresisOn  = cfg->hysteresisOn;
        s_hysteresisOff = cfg->hysteresisOff;
    }

    /* --- Load mouth model from SD to HyperRAM --- */
    int32_t modelSize = PrepareModelToHyperRAM();
    if (modelSize <= 0) {
        printf_err("SpeakingDetector: failed to load mouth model from SD\n");
        return -1;
    }

    __DSB();
    __DMB();

    const uint8_t *pModel = (const uint8_t *)MODEL_AT_HYPERRAM_ADDR;
    if (modelSize < 12 ||
        pModel[4] != 0x54 || pModel[5] != 0x46 ||
        pModel[6] != 0x4c || pModel[7] != 0x33)
    {
        printf_err("SpeakingDetector: invalid TFLite magic\n");
        return -2;
    }
    info("SpeakingDetector: model TFL3 OK\n");

    /* --- Face detection model (embedded in flash) --- */
    if (!s_faceModel.Init(s_arenaFace, sizeof(s_arenaFace),
            (unsigned char *)arm::app::face_detection::GetModelPointer(),
            arm::app::face_detection::GetModelLen()))
    {
        printf_err("SpeakingDetector: face model init failed\n");
        return -3;
    }
    info("SpeakingDetector: face model OK\n");

    /* --- Mouth model (from HyperRAM) --- */
    if (!s_mouthModel.Init(s_arenaMouth, sizeof(s_arenaMouth),
            (unsigned char *)MODEL_AT_HYPERRAM_ADDR, modelSize))
    {
        printf_err("SpeakingDetector: mouth model init failed\n");
        return -4;
    }
    info("SpeakingDetector: mouth model OK\n");

    /* NOTE: MPU setup is the caller's responsibility.  Call
       SpeakingDetector_GetTensorArenas() to obtain the arena addresses,
       combine them with your frame-buffer regions, and call
       InitPreDefMPURegion() once before calling this function. */

    /* --- Mouth model input shape --- */
    s_mouthInput = s_mouthModel.GetInputTensor(0);
    if (!s_mouthInput || !s_mouthInput->dims || s_mouthInput->dims->size < 3) {
        printf_err("SpeakingDetector: bad mouth input tensor\n");
        return -5;
    }

    TfLiteIntArray *mShape = s_mouthModel.GetInputShape(0);
    s_mouthCols = mShape->data[arm::app::MouthDetectionModel::ms_inputColsIdx];
    s_mouthRows = mShape->data[arm::app::MouthDetectionModel::ms_inputRowsIdx];

    if (s_mouthRows != s_mouthCols) {
        printf_err("SpeakingDetector: mouth input must be square (%dx%d)\n",
                   s_mouthRows, s_mouthCols);
        return -6;
    }

    /* --- Face detection input shape --- */
    TfLiteIntArray *fShape = s_faceModel.GetInputShape(0);
    s_faceCols = fShape->data[arm::app::FaceDetectionModel::ms_inputColsIdx];
    s_faceRows = fShape->data[arm::app::FaceDetectionModel::ms_inputRowsIdx];
    s_faceInput = s_faceModel.GetInputTensor(0);

    /* --- Post-processors (static lifetime, created once) --- */
    TfLiteTensor *fdOut0 = s_faceModel.GetOutputTensor(0);
    TfLiteTensor *fdOut1 = s_faceModel.GetOutputTensor(1);

    static arm::app::face_detection::PostProcessParams s_fdParams;
    s_fdParams.inputImgRows     = s_faceRows;
    s_fdParams.inputImgCols     = s_faceCols;
    s_fdParams.originalImageRows = 0;  /* updated each frame in RunFrame */
    s_fdParams.originalImageCols = 0;
    s_fdParams.anchor1      = anchor1;
    s_fdParams.anchor2      = anchor2;
    s_fdParams.threshold    = faceTh;
    s_fdParams.nms          = 0.45f;
    s_fdParams.numClasses   = 1;
    s_fdParams.topN         = 0;

    s_fdParamsPtr = &s_fdParams;

    static arm::app::FaceDetectorPostProcess s_fdPP(fdOut0, fdOut1, s_faceResults, s_fdParams);
    s_postFace = &s_fdPP;

    static arm::app::mouth_detection::MouthYOLOv8PostProcessing s_mouthPP(
        &s_mouthModel, mouthTh, nmsTh, s_mouthRows);
    s_postMouth = &s_mouthPP;

    InitTrackedFaces();

    info("SpeakingDetector: init complete (mouth %dx%d, face %dx%d)\n",
         s_mouthRows, s_mouthCols, s_faceRows, s_faceCols);
    return 0;
}

int SpeakingDetector_RunFrame(
    uint8_t *rgb565Data, int frameW, int frameH,
    SpeakingFaceResult results[], int maxResults)
{
    if (!s_postFace || !s_postMouth) return 0;

    image_t srcImg;
    srcImg.w = frameW;
    srcImg.h = frameH;
    srcImg.data = rgb565Data;
    srcImg.pixfmt = PIXFORMAT_RGB565;

    rectangle_t roi;

    /* --- Step 1: face detection (grayscale 192x192) --- */
    s_faceResults.clear();

    s_fdParamsPtr->originalImageRows = frameH;
    s_fdParamsPtr->originalImageCols = frameW;

    roi.x = 0;  roi.y = 0;
    roi.w = frameW;  roi.h = frameH;

    image_t faceResized;
    faceResized.w = s_faceCols;
    faceResized.h = s_faceRows;
    faceResized.data = (uint8_t *)s_faceInput->data.data;
    faceResized.pixfmt = PIXFORMAT_GRAYSCALE;
    imlib_nvt_scale(&srcImg, &faceResized, &roi);

    {
        uint8_t *u = (uint8_t *)s_faceInput->data.data;
        int8_t  *s = (int8_t  *)s_faceInput->data.data;
        size_t n;
        for (n = 0; n < s_faceInput->bytes; n++)
            s[n] = (int8_t)((int32_t)u[n] - 128);
    }

    s_faceModel.RunInference();
    s_postFace->RunPostProcess(s_faceResults);

    /* Expand face boxes 1.4x */
    {
        float sf = 1.4f;
        size_t i;
        for (i = 0; i < s_faceResults.size(); i++) {
            auto *fb = &s_faceResults[i];
            float sw = sf * fb->m_h;
            float sh = sf * fb->m_h;
            int nx = fb->m_x0 - (int)((sw - fb->m_w) / 2);
            int ny = fb->m_y0 - (int)((sh - fb->m_h) / 2);
            if (nx < 0) nx = 0;
            if (ny < 0) ny = 0;
            int nw = (int)sw;
            int nh = (int)sh;
            if (nx + nw >= frameW) nw = frameW - nx;
            if (ny + nh >= frameH) nh = frameH - ny;
            fb->m_x0 = nx;  fb->m_y0 = ny;
            fb->m_w  = nw;  fb->m_h  = nh;
        }
    }

    /* --- Step 2: per-face mouth inference --- */
    int numFaces = (int)s_faceResults.size();
    if (numFaces > maxResults) numFaces = maxResults;
    if (numFaces > MAX_TRACKED_FACES) numFaces = MAX_TRACKED_FACES;

    bool perFaceRawOpen[MAX_TRACKED_FACES];
    {
        int f;
        for (f = 0; f < MAX_TRACKED_FACES; f++)
            perFaceRawOpen[f] = false;

        for (f = 0; f < numFaces; f++) {
            const auto &faceBox = s_faceResults[f];
            roi.x = faceBox.m_x0;
            roi.y = faceBox.m_y0;
            roi.w = faceBox.m_w;
            roi.h = faceBox.m_h;

            image_t mouthResized;
            mouthResized.w = s_mouthCols;
            mouthResized.h = s_mouthRows;
            mouthResized.data = (uint8_t *)s_mouthInput->data.data;
            mouthResized.pixfmt = PIXFORMAT_RGB888;
            imlib_nvt_scale(&srcImg, &mouthResized, &roi);

            {
                uint8_t *u = (uint8_t *)s_mouthInput->data.data;
                int8_t  *s = (int8_t  *)s_mouthInput->data.data;
                size_t n;
                for (n = 0; n < s_mouthInput->bytes; n++)
                    s[n] = (int8_t)((int32_t)u[n] - 128);
            }

            s_mouthModel.RunInference();

            s_mouthTemp.clear();
            s_postMouth->RunPostProcessing(
                s_mouthRows, s_mouthCols,
                (uint32_t)roi.h, (uint32_t)roi.w, s_mouthTemp);

            size_t m;
            for (m = 0; m < s_mouthTemp.size(); m++) {
                if (s_mouthTemp[m].m_classId == 1) {
                    perFaceRawOpen[f] = true;
                    break;
                }
            }
        }
    }

    /* --- Step 3: tracking + smoothing --- */
    {
        int i;
        for (i = 0; i < MAX_TRACKED_FACES; i++)
            s_tracked[i].missedFrames++;

        for (i = 0; i < numFaces; i++) {
            const auto &fb = s_faceResults[i];
            float cx = fb.m_x0 + fb.m_w * 0.5f;
            float cy = fb.m_y0 + fb.m_h * 0.5f;

            int slot = MatchOrCreateSlot(cx, cy);
            s_tracked[slot].cx = cx;
            s_tracked[slot].cy = cy;
            s_tracked[slot].missedFrames = 0;

            results[i].x = fb.m_x0;
            results[i].y = fb.m_y0;
            results[i].w = fb.m_w;
            results[i].h = fb.m_h;
            results[i].rawMouthOpen = perFaceRawOpen[i];
            results[i].isSpeaking   = SmoothPerFace(slot, perFaceRawOpen[i]);
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

    int greenBox = COLOR_R5_G6_B5_TO_RGB565(0, COLOR_G6_MAX, 0);
    int redBox   = COLOR_R5_G6_B5_TO_RGB565(COLOR_R5_MAX, 0, 0);
    int i;

    for (i = 0; i < numResults; i++) {
        bool speaking = results[i].isSpeaking;
        int boxColor  = speaking ? greenBox : redBox;
        const char *label = speaking ? "Speaking" : "Not Speaking";

        imlib_draw_rectangle(&drawImg,
            results[i].x, results[i].y,
            results[i].w, results[i].h,
            boxColor, 2, false);

        int labelY = (results[i].y - 14 > 0) ? (results[i].y - 14) : results[i].y;
        imlib_draw_string(&drawImg, results[i].x, labelY, label, boxColor,
                          2, 0, 0, false, false, false, false, 0, false, false);
    }
}

void SpeakingDetector_GetTensorArenas(
    void **faceArena, uint32_t *faceArenaSize,
    void **mouthArena, uint32_t *mouthArenaSize)
{
    *faceArena     = s_arenaFace;
    *faceArenaSize = sizeof(s_arenaFace);
    *mouthArena     = s_arenaMouth;
    *mouthArenaSize = sizeof(s_arenaMouth);
}
