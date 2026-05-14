/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    face landmark network sample. Demonstrate face landmark detect.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "FaceLandmarkModel.hpp"       /* Model API */
#include "FaceDetectionModel.hpp"       /* Model API */
#include "FaceLandmarkPostProcessing.hpp"
#include "FaceDetectorPostProcessing.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"
#include <cmath>
#include <cstdio>
#include "ModelFileReader.h"
#include "ff.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_DISPLAY__
//#define __USE_UVC__
#if defined (FACE_LANDMARK_ATTENTION_MODEL)
#define __LOAD_MODEL_FROM_SD__
#endif

#include "Profiler.hpp"

#include "ImageSensor.h"

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#define NUM_FRAMEBUF 2  //1 or 2
#define MODEL_AT_HYPERRAM_ADDR (0x82400000)
#define FACE_PRESENCE_THRESHOLD  				(0.4)

/* Speaking detection - MAR + MAR velocity (simple, reliable) */
#define SPEAKING_MAR_VELOCITY_THRESHOLD_ON	(0.022f) /* MAR change when mouth opens/closes (raised - less sensitive) */
#define SPEAKING_MAR_VELOCITY_THRESHOLD_OFF	(0.008f) /* Release when below */
#define SPEAKING_MAR_THRESHOLD_ON			(0.26f)  /* Mouth open to trigger (raised - less sensitive to head shake) */
#define SPEAKING_MAR_THRESHOLD_OFF			(0.18f)  /* Mouth closed to release */
#define SPEAKING_SMOOTHING_FRAMES			(3)      /* Consecutive above threshold to trigger (reduces flicker) */
#define SPEAKING_RELEASE_FRAMES				(5)      /* Consecutive below to release */
#define SPEAKING_MIN_DURATION_FRAMES		(5)      /* Min frames speaking before can release */
#define SPEAKING_DETECT_EVERY_N_FRAMES		(1)      /* Run detection every frame (no skip) */
#define MAR_SMOOTHING_ALPHA				(0.3f)   /* MAR smoothing */
#define BBOX_SMOOTHING_ALPHA				(0.25f)  /* For storage only */
#define HEAD_MOVE_THRESHOLD				(0.08f)  /* Reject ON if head moved > 8% (was 15% - too lenient, caused false speaking) */
#define HEAD_SIZE_CHANGE_THRESHOLD		(0.12f)  /* Reject ON if bbox size changed > 12% (catches zoom/tilt) */

/* Landmark smoothing */
#define LANDMARK_SMOOTHING_ALPHA			(0.28f)  /* Lower = more smoothing */

/* Lip offset: landmarks too high = shift down. Tune per camera/model. */
#define LIP_OFFSET_X  (4)   /* Pixels to shift right */
#define LIP_OFFSET_Y  (3)   /* Positive = shift down (corrects "too high") */

/* Lip indices: lip contour + chin for MAR. */
#define LIP_LANDMARK_NUM		(24)
#define MAR_CHIN_INDEX		152
static const int s_i32LipLandmarkIndices[LIP_LANDMARK_NUM] = {
	61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
	78, 308, 87, 14
};

typedef enum
{
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct
{
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
    std::vector<arm::app::face_landmark::KeypointResult> results_KP;
    std::vector<arm::app::face_detection::DetectionResult> results_FD;
    std::vector<bool> isSpeaking;   /* Per-face speaking state */
} S_FRAMEBUF;

/* Previous lip positions - stores SMOOTHED RELATIVE coords (0-1 within face bbox) for velocity + EMA */
#define MAX_TRACKED_FACES  5
typedef struct {
    float lipX[LIP_LANDMARK_NUM];
    float lipY[LIP_LANDMARK_NUM];
    float prevMAR;
    float prevPrevMAR;
    int prevPrevValid;
    int x0, y0, w, h;
    int smoothX0, smoothY0, smoothW, smoothH;
    int valid;
} S_PREV_LIP_STATE;
static S_PREV_LIP_STATE s_asPrevLipState[MAX_TRACKED_FACES];
static int s_i32SpeakingConfirmCount[MAX_TRACKED_FACES];
static int s_i32SpeakingReleaseCount[MAX_TRACKED_FACES];
static int s_i32SpeakingDurationCount[MAX_TRACKED_FACES];
static bool s_abSpeaking[MAX_TRACKED_FACES];  /* Per-track speaking state (trackId-indexed) */

static float s_afSmoothedRelX[LIP_LANDMARK_NUM];  /* Relative coords 0-1 within face bbox */
static float s_afSmoothedRelY[LIP_LANDMARK_NUM];
static uint32_t s_u32SpeakingDetectFrameCount = 0;  /* Skip frames: only run detection every N frames */

S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
/* Tensor arena buffer */
#undef ACTIVATION_BUF_SZ
#if defined(FACE_LANDMARK_ATTENTION_MODEL)
#define FACE_LANDMARK_ACTIVATION_BUF_SZ	 (500000)
#else
#define FACE_LANDMARK_ACTIVATION_BUF_SZ	 (460000)
#endif
#define FACE_DETECTION_ACTIVATION_BUF_SZ (460000)
static uint8_t tensorArena_FaceLandmark[FACE_LANDMARK_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
static uint8_t tensorArena_FaceDetection[FACE_DETECTION_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

#if !defined(__LOAD_MODEL_FROM_SD__)
/* Optional getter function for the model pointer and its size. */
namespace face_landmark
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace face_landmark */

namespace face_detection
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace face_detection */

#endif

} /* namespace app */
} /* namespace arm */

//frame buffer managemnet function
static S_FRAMEBUF *get_empty_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF)
            return &s_asFramebuf[i];
    }

    return NULL;
}

#define IMAGE_DISP_UPSCALE_FACTOR 2
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

/* Image processing initiate function */
//Used by omv library
#if defined(__USE_UVC__)
//UVC only support QVGA, QQVGA
#define GLCD_WIDTH	320
#define GLCD_HEIGHT	240
#else
#define GLCD_WIDTH	240
#define GLCD_HEIGHT	240
#endif

#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".bss.vram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];
#endif

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0 ; i < NUM_FRAMEBUF; i++)
    {
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;
    }

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

/* Draw lip keypoints: convert relative (0-1) to absolute using face bbox. */
static void DrawLipLandmark(
    const float *smoothedRelX, const float *smoothedRelY,
    int faceX0, int faceY0, int faceW, int faceH,
    image_t *drawImg
)
{
	int lipColor = COLOR_R5_G6_B5_TO_RGB565(0, COLOR_G6_MAX, 0);
	for (int i = 0; i < LIP_LANDMARK_NUM; i++) {
		int drawX = faceX0 + (int)(smoothedRelX[i] * faceW) + LIP_OFFSET_X;
		int drawY = faceY0 + (int)(smoothedRelY[i] * faceH) + LIP_OFFSET_Y;
		imlib_draw_circle(drawImg, drawX, drawY, 2, lipColor, 1, true);
	}
}

/* Apply EMA in RELATIVE space (0-1 within face bbox). Head movement doesn't affect relative lip position. */
static void ApplyLipSmoothing(
    const std::vector<arm::app::face_landmark::KeypointResult> &results_KP,
    int prevFaceIdx, int faceW, int faceH,
    float *outSmoothedRelX, float *outSmoothedRelY
)
{
    for (int i = 0; i < LIP_LANDMARK_NUM; i++) {
        int idx = s_i32LipLandmarkIndices[i];
        if (idx >= (int)results_KP.size()) continue;
        /* Landmark m_x,m_y are in face crop coords (0 to faceW, 0 to faceH). Normalize to 0-1. */
        float rawRelX = (faceW > 0) ? ((float)results_KP[idx].m_x / (float)faceW) : 0.0f;
        float rawRelY = (faceH > 0) ? ((float)results_KP[idx].m_y / (float)faceH) : 0.0f;
        if (prevFaceIdx >= 0 && prevFaceIdx < MAX_TRACKED_FACES && s_asPrevLipState[prevFaceIdx].valid) {
            outSmoothedRelX[i] = LANDMARK_SMOOTHING_ALPHA * rawRelX + (1.0f - LANDMARK_SMOOTHING_ALPHA) * s_asPrevLipState[prevFaceIdx].lipX[i];
            outSmoothedRelY[i] = LANDMARK_SMOOTHING_ALPHA * rawRelY + (1.0f - LANDMARK_SMOOTHING_ALPHA) * s_asPrevLipState[prevFaceIdx].lipY[i];
        } else {
            outSmoothedRelX[i] = rawRelX;
            outSmoothedRelY[i] = rawRelY;
        }
    }
}

/* MAR: vertical/horizontal extent of lip bbox + chin. Lip-relative. */
static float ComputeMAR(
    const std::vector<arm::app::face_landmark::KeypointResult> &results_KP,
    const float *smoothedRelX, const float *smoothedRelY,
    int faceW, int faceH
)
{
    if (results_KP.size() < 468) return 0.0f;
    float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
    for (int i = 0; i < LIP_LANDMARK_NUM; i++) {
        if (smoothedRelX[i] < minX) minX = smoothedRelX[i];  if (smoothedRelX[i] > maxX) maxX = smoothedRelX[i];
        if (smoothedRelY[i] < minY) minY = smoothedRelY[i];  if (smoothedRelY[i] > maxY) maxY = smoothedRelY[i];
    }
    if (MAR_CHIN_INDEX < (int)results_KP.size() && faceW > 0 && faceH > 0) {
        float cx = (float)results_KP[MAR_CHIN_INDEX].m_x / (float)faceW;
        float cy = (float)results_KP[MAR_CHIN_INDEX].m_y / (float)faceH;
        if (cx < minX) minX = cx;  if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;  if (cy > maxY) maxY = cy;
    }
    float horiz = maxX - minX;
    float vert = maxY - minY;
    if (horiz < 0.02f) return 0.0f;
    return vert / horiz;
}

/* MAR velocity from smoothed MAR. */
static float ComputeMARVelocityAndSmooth(float rawMAR, int trackId, float *outSmoothedMAR)
{
    float smoothed = rawMAR;
    float velocity = 0.0f;
    if (trackId >= 0 && trackId < MAX_TRACKED_FACES && s_asPrevLipState[trackId].valid) {
        float prev = s_asPrevLipState[trackId].prevMAR;
        smoothed = MAR_SMOOTHING_ALPHA * rawMAR + (1.0f - MAR_SMOOTHING_ALPHA) * prev;
        if (s_asPrevLipState[trackId].prevPrevValid) {
            float old = s_asPrevLipState[trackId].prevPrevMAR;
            velocity = (smoothed > old) ? (smoothed - old) : (old - smoothed);
        } else {
            velocity = (smoothed > prev) ? (smoothed - prev) : (prev - smoothed);
        }
    }
    *outSmoothedMAR = smoothed;
    return velocity;
}

/* Find best-matching previous face by bbox center distance (for face order changes) */
static int FindMatchingPrevFace(int curX0, int curY0, int curW, int curH)
{
    int bestIdx = -1;
    float bestDist = 1e9f;
    int curCx = curX0 + curW / 2;
    int curCy = curY0 + curH / 2;

    for (int j = 0; j < MAX_TRACKED_FACES; j++) {
        if (!s_asPrevLipState[j].valid) continue;
        int prevCx = s_asPrevLipState[j].x0 + s_asPrevLipState[j].w / 2;
        int prevCy = s_asPrevLipState[j].y0 + s_asPrevLipState[j].h / 2;
        float dx = (float)(curCx - prevCx);
        float dy = (float)(curCy - prevCy);
        float dist = std::sqrt(dx*dx + dy*dy);
        if (dist < bestDist && dist < (float)(curW + curH)) {  /* Allow face to move within ~2x face size */
            bestDist = dist;
            bestIdx = j;
        }
    }
    return bestIdx;
}

/* Allocate new track when no match. Use first invalid slot. */
static int AllocateTrack(int faceIdx)
{
    for (int j = 0; j < MAX_TRACKED_FACES; j++) {
        if (!s_asPrevLipState[j].valid) return j;
    }
    return faceIdx % MAX_TRACKED_FACES;  /* Fallback: evict by face index */
}

/* Compute normalized bbox center movement. Returns sqrt(dx^2+dy^2) in units of bbox size. */
static float HeadMoveAmount(int trackId, int curX0, int curY0, int curW, int curH)
{
    if (trackId < 0 || trackId >= MAX_TRACKED_FACES || !s_asPrevLipState[trackId].valid || curW <= 0 || curH <= 0)
        return 0.0f;
    int prevCx = s_asPrevLipState[trackId].x0 + s_asPrevLipState[trackId].w / 2;
    int prevCy = s_asPrevLipState[trackId].y0 + s_asPrevLipState[trackId].h / 2;
    int curCx = curX0 + curW / 2;
    int curCy = curY0 + curH / 2;
    float dx = (float)(curCx - prevCx) / (float)curW;
    float dy = (float)(curCy - prevCy) / (float)curH;
    return std::sqrt(dx*dx + dy*dy);
}

/* True if bbox size changed significantly (head moved closer/further or detector resized). */
static int HeadUnstable(int trackId, int curX0, int curY0, int curW, int curH)
{
    float posMove = HeadMoveAmount(trackId, curX0, curY0, curW, curH);
    if (posMove > HEAD_MOVE_THRESHOLD) return 1;
    if (trackId < 0 || trackId >= MAX_TRACKED_FACES || !s_asPrevLipState[trackId].valid || curW <= 0 || curH <= 0)
        return 0;
    int prevW = s_asPrevLipState[trackId].w;
    int prevH = s_asPrevLipState[trackId].h;
    float sizeChangeW = (prevW > 0) ? (float)std::abs(curW - prevW) / (float)prevW : 0.0f;
    float sizeChangeH = (prevH > 0) ? (float)std::abs(curH - prevH) / (float)prevH : 0.0f;
    return (sizeChangeW > HEAD_SIZE_CHANGE_THRESHOLD || sizeChangeH > HEAD_SIZE_CHANGE_THRESHOLD) ? 1 : 0;
}

/* Update smoothed bbox every frame - reduces keypoint jitter when face detector output wobbles. */
static void UpdateSmoothedBbox(int trackId, int faceX0, int faceY0, int faceW, int faceH)
{
    if (trackId < 0 || trackId >= MAX_TRACKED_FACES) return;
    S_PREV_LIP_STATE *s = &s_asPrevLipState[trackId];
    float a = BBOX_SMOOTHING_ALPHA;
    if (s->valid) {
        s->smoothX0 = (int)(a * faceX0 + (1.0f - a) * s->smoothX0);
        s->smoothY0 = (int)(a * faceY0 + (1.0f - a) * s->smoothY0);
        s->smoothW  = (int)(a * faceW  + (1.0f - a) * s->smoothW);
        s->smoothH  = (int)(a * faceH  + (1.0f - a) * s->smoothH);
    } else {
        s->smoothX0 = faceX0;
        s->smoothY0 = faceY0;
        s->smoothW  = faceW;
        s->smoothH  = faceH;
    }
}

/* Store smoothed lip positions, MAR, and bbox. */
static void StoreLipState(
    const float *smoothedRelX, const float *smoothedRelY, float smoothedMAR,
    int storeIdx, int faceX0, int faceY0, int faceW, int faceH
)
{
    if (storeIdx >= MAX_TRACKED_FACES || storeIdx < 0) return;
    S_PREV_LIP_STATE *s = &s_asPrevLipState[storeIdx];
    for (int i = 0; i < LIP_LANDMARK_NUM; i++) {
        s->lipX[i] = smoothedRelX[i];
        s->lipY[i] = smoothedRelY[i];
    }
    s->prevPrevMAR = s->prevMAR;
    s->prevPrevValid = s->valid;
    s->prevMAR = smoothedMAR;
    s->x0 = faceX0;
    s->y0 = faceY0;
    s->w = faceW;
    s->h = faceH;
    if (s->valid) {
        float a = BBOX_SMOOTHING_ALPHA;
        s->smoothX0 = (int)(a * faceX0 + (1.0f - a) * s->smoothX0);
        s->smoothY0 = (int)(a * faceY0 + (1.0f - a) * s->smoothY0);
        s->smoothW  = (int)(a * faceW  + (1.0f - a) * s->smoothW);
        s->smoothH  = (int)(a * faceH  + (1.0f - a) * s->smoothH);
    } else {
        s->smoothX0 = faceX0;
        s->smoothY0 = faceY0;
        s->smoothW  = faceW;
        s->smoothH  = faceH;
    }
    s->valid = 1;
}

static void DrawDetectFace(
    std::vector<arm::app::face_detection::DetectionResult> &results,
    std::vector<bool> &isSpeaking,
    image_t *drawImg
)
{
	arm::app::face_detection::DetectionResult faceBox;
	int faceBoxSize = results.size();
	
	for(int i = 0; i < faceBoxSize; i ++)
	{
		faceBox = results[i];
		/* Green when speaking, blue otherwise */
		int boxColor = (i < (int)isSpeaking.size() && isSpeaking[i])
			? COLOR_R5_G6_B5_TO_RGB565(0, COLOR_G6_MAX, 0)  /* Green */
			: COLOR_B5_MAX;  /* Blue */
		imlib_draw_rectangle(drawImg, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h, boxColor, 2, false);
		/* Draw "Speaking" label when speaking */
		if (i < (int)isSpeaking.size() && isSpeaking[i]) {
			imlib_draw_string(drawImg, faceBox.m_x0, faceBox.m_y0 - 14, "Speaking", COLOR_R5_G6_B5_TO_RGB565(0, COLOR_G6_MAX, 0), 1, 0, 0, false, 0, false, false, 0, false, false);
		}
	}
}

static void DetectFaceRegion(
    S_FRAMEBUF *infFramebuf,
    arm::app::FaceDetectionModel *faceDetectionModel,	
	arm::app::FaceDetectorPostProcess *postProcess,
	arm::app::Profiler *profiler
)
{
    TfLiteIntArray *inputShape = faceDetectionModel->GetInputShape(0);
    TfLiteTensor *inputTensor   = faceDetectionModel->GetInputTensor(0);
    rectangle_t roi;

    const int inputImgCols = inputShape->data[arm::app::FaceDetectionModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::FaceDetectionModel::ms_inputRowsIdx];

	uint64_t u64StartCycle;
	uint64_t u64EndCycle;

	auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
	auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);
	
	//resize full image to input tensor
	image_t resizeImg;

	roi.x = 0;
	roi.y = 0;
	roi.w = infFramebuf->frameImage.w;
	roi.h = infFramebuf->frameImage.h;

	resizeImg.w = inputImgCols;
	resizeImg.h = inputImgRows;
	resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
	resizeImg.pixfmt = PIXFORMAT_GRAYSCALE;

	if(profiler)
		u64StartCycle = pmu_get_systick_Count();

	imlib_nvt_scale(&infFramebuf->frameImage, &resizeImg, &roi);

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("face detect resize cycles %llu \n", (u64EndCycle - u64StartCycle));
	}

	if(profiler){
		u64StartCycle = pmu_get_systick_Count();
	}
		
    /* Original quant: pixel-128. Model was trained with this. */
	for (size_t i = 0; i < inputTensor->bytes; i++)
	{
		signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
	}

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("face detect quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
	}

	if(profiler){
		profiler->StartProfiling("Inference");
	}

	faceDetectionModel->RunInference();

	if(profiler){
		profiler->StopProfiling();
		profiler->PrintProfilingResult();
	}

	if(profiler){
		u64StartCycle = pmu_get_systick_Count();
	}
	
	postProcess->RunPostProcess(infFramebuf->results_FD);

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("face detect post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
	}

	float scaleFactoryW = 1.4;
	float scaleFactoryH = 1.4;
	arm::app::face_detection::DetectionResult *faceBox;
	//fine tune face region
	for(int i = 0 ; i < infFramebuf->results_FD.size(); i ++)
	{
		float scaleW;
		float scaleH;
		int newX;
		int newY;
		
		int newW;
		int newH;

		faceBox = &(infFramebuf->results_FD[i]);
	
		scaleH = scaleFactoryH * faceBox->m_h;
//		scaleW = scaleFactoryW * faceBox->m_w;
		scaleW = scaleH;
		newW = scaleW;
		newH = scaleH;

		newX = faceBox->m_x0 - ((scaleW - faceBox->m_w) / 2);
		newY = faceBox->m_y0 - ((scaleH - faceBox->m_h) / 2);
		
		if(newX < 0)
			newX = 0;
		
		if(newY < 0)
			newY = 0;

		if(newX + newW >= infFramebuf->frameImage.w)
			newW = infFramebuf->frameImage.w - newX;
		
		if(newY + newH >= infFramebuf->frameImage.h)
			newH = infFramebuf->frameImage.h - newY;

		faceBox->m_x0 = newX;
		faceBox->m_y0 = newY;
		faceBox->m_w = newW;
		faceBox->m_h = newH;
	}
}

static void DetectFaceLandmark_DrawResult(
    S_FRAMEBUF *infFramebuf,
    arm::app::FaceLandmarkModel *faceLandmarkModel,	
	arm::app::face_landmark::FaceLandmarkPostProcessing *postProcess,
	arm::app::Profiler *profiler
)
{
	int i;
	arm::app::face_detection::DetectionResult faceBox;
    rectangle_t roi;
    TfLiteIntArray *inputShape = faceLandmarkModel->GetInputShape(0);
    TfLiteTensor *inputTensor   = faceLandmarkModel->GetInputTensor(0);

    const int inputImgCols = inputShape->data[arm::app::FaceLandmarkModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::FaceLandmarkModel::ms_inputRowsIdx];

	uint64_t u64StartCycle;
	uint64_t u64EndCycle;

	//Quantize input tensor data
	auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
	auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

	TfLiteTensor *modelOutput0 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_MESH_TENSOR_INDEX);

	#if defined(FACE_LANDMARK_LEFT_IRIS_TENSOR_INDEX)
		TfLiteTensor *modelOutput1 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_LEFT_IRIS_TENSOR_INDEX);
	#else
		TfLiteTensor *modelOutput1 = NULL;
	#endif

	#if defined(FACE_LANDMARK_RIGHT_IRIS_TENSOR_INDEX)
		TfLiteTensor *modelOutput2 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_RIGHT_IRIS_TENSOR_INDEX);			
	#else
		TfLiteTensor *modelOutput2 = NULL;
	#endif

	TfLiteTensor *modelOutput3 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_FACE_FLAG_TENSOR_INDEX);

	/* Resize speaking state to match face count */
	infFramebuf->isSpeaking.resize(infFramebuf->results_FD.size(), false);

	/* Frame skip: only run speaking detection every N frames - reduces jitter-induced false triggers */
	s_u32SpeakingDetectFrameCount++;
	const int runDetectionThisFrame = (s_u32SpeakingDetectFrameCount % SPEAKING_DETECT_EVERY_N_FRAMES) == 0;
	
	for(i = 0 ; i < infFramebuf->results_FD.size(); i ++)
	{
		faceBox = infFramebuf->results_FD[i];

		/* Stable trackId: use for all state. Fixes flicker when face order changes. */
		int matchedPrev = FindMatchingPrevFace(faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);
		if (matchedPrev < 0 && infFramebuf->results_FD.size() == 1) {
			matchedPrev = (s_asPrevLipState[0].valid) ? 0 : -1;
		}
		int trackId = (matchedPrev >= 0) ? matchedPrev : AllocateTrack(i);
		if (matchedPrev < 0 && trackId < MAX_TRACKED_FACES) {
			s_abSpeaking[trackId] = false;
			s_i32SpeakingConfirmCount[trackId] = 0;
			s_i32SpeakingReleaseCount[trackId] = 0;
			s_i32SpeakingDurationCount[trackId] = 0;
		}

		/* Raw face bbox for crop - no padding. */
		roi.x = faceBox.m_x0;
		roi.y = faceBox.m_y0;
		roi.w = faceBox.m_w;
		roi.h = faceBox.m_h;

		//resize face region image to input tensor
		image_t resizeImg;

		resizeImg.w = inputImgCols;
		resizeImg.h = inputImgRows;
		resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
		resizeImg.pixfmt = PIXFORMAT_RGB888;

		if(profiler)
			u64StartCycle = pmu_get_systick_Count();

        imlib_nvt_scale(&infFramebuf->frameImage, &resizeImg, &roi);

		if(profiler){
			u64EndCycle = pmu_get_systick_Count();
			info("face landmark resize cycles %llu \n", (u64EndCycle - u64StartCycle));
		}
			
		if(profiler){
			u64StartCycle = pmu_get_systick_Count();
		}
			
		/* Original quant: pixel-128. Model was trained with this. */
		for (size_t i = 0; i < inputTensor->bytes; i++)
		{
			signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
		}

		if(profiler){
			u64EndCycle = pmu_get_systick_Count();
			info("face landmark quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
		}

		if(profiler){
			profiler->StartProfiling("Inference");
		}

		faceLandmarkModel->RunInference();

		if(profiler){
			profiler->StopProfiling();
			profiler->PrintProfilingResult();
		}

		if(profiler){
			u64StartCycle = pmu_get_systick_Count();
		}

		postProcess->RunPostProcessing(
			inputImgCols,
			inputImgRows,
			roi.w,
			roi.h,
			modelOutput0,
			modelOutput1,
			modelOutput2,
			modelOutput3,
			infFramebuf->results_KP);

		if(profiler){
			u64EndCycle = pmu_get_systick_Count();
			info("face landmark post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
		}

		/* Always: smoothing + drawing. Detection + store only every N frames. */
		if (trackId < MAX_TRACKED_FACES && infFramebuf->results_KP.size() >= 468) {
			ApplyLipSmoothing(infFramebuf->results_KP, trackId, roi.w, roi.h, s_afSmoothedRelX, s_afSmoothedRelY);

			if (runDetectionThisFrame) {
				/* Simple MAR + MAR velocity */
				float rawMAR = ComputeMAR(infFramebuf->results_KP, s_afSmoothedRelX, s_afSmoothedRelY, roi.w, roi.h);
				float smoothedMAR;
				float marVelocity = ComputeMARVelocityAndSmooth(rawMAR, trackId, &smoothedMAR) / (float)SPEAKING_DETECT_EVERY_N_FRAMES;

				int headUnstable = HeadUnstable(trackId, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);

				int signalAboveOn  = !headUnstable && (marVelocity > SPEAKING_MAR_VELOCITY_THRESHOLD_ON) && (smoothedMAR > SPEAKING_MAR_THRESHOLD_ON);
				int signalBelowOff = (marVelocity < SPEAKING_MAR_VELOCITY_THRESHOLD_OFF) || (smoothedMAR < SPEAKING_MAR_THRESHOLD_OFF);

				/* All state indexed by trackId - fixes flicker when face order changes */
				if (signalAboveOn) {
					s_i32SpeakingConfirmCount[trackId] = (s_i32SpeakingConfirmCount[trackId] < SPEAKING_SMOOTHING_FRAMES) ? s_i32SpeakingConfirmCount[trackId] + 1 : SPEAKING_SMOOTHING_FRAMES;
					s_i32SpeakingReleaseCount[trackId] = 0;
					if (s_i32SpeakingConfirmCount[trackId] >= SPEAKING_SMOOTHING_FRAMES) {
						s_abSpeaking[trackId] = true;
					}
				} else if (s_abSpeaking[trackId]) {
					s_i32SpeakingDurationCount[trackId]++;
					if (signalBelowOff) {
						s_i32SpeakingReleaseCount[trackId]++;
						if (s_i32SpeakingReleaseCount[trackId] >= SPEAKING_RELEASE_FRAMES &&
							s_i32SpeakingDurationCount[trackId] >= SPEAKING_MIN_DURATION_FRAMES) {
							s_abSpeaking[trackId] = false;
						}
					} else {
						s_i32SpeakingReleaseCount[trackId] = 0;
					}
					s_i32SpeakingConfirmCount[trackId] = 0;
				} else {
					s_i32SpeakingReleaseCount[trackId] = 0;
					s_i32SpeakingConfirmCount[trackId] = 0;
					s_i32SpeakingDurationCount[trackId] = 0;
				}
				if (!s_abSpeaking[trackId]) {
					s_i32SpeakingDurationCount[trackId] = 0;
				}
				/* Map trackId -> face index for display */
				infFramebuf->isSpeaking[i] = s_abSpeaking[trackId];
				StoreLipState(s_afSmoothedRelX, s_afSmoothedRelY, smoothedMAR, trackId, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);
			} else {
				/* Not a detection frame - still need to map trackId to face for display */
				infFramebuf->isSpeaking[i] = s_abSpeaking[trackId];
			}
		} else if (trackId < MAX_TRACKED_FACES) {
			s_asPrevLipState[trackId].valid = 0;
		}

		//Draw lip landmarks (relative to face bbox)
		if(infFramebuf->results_KP.size() >= 468)
		{
			/* Use smoothed bbox for drawing - reduces jitter when face detector output wobbles */
			UpdateSmoothedBbox(trackId, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);
			int drawX0 = faceBox.m_x0, drawY0 = faceBox.m_y0, drawW = faceBox.m_w, drawH = faceBox.m_h;
			if (trackId >= 0 && trackId < MAX_TRACKED_FACES && s_asPrevLipState[trackId].valid) {
				drawX0 = s_asPrevLipState[trackId].smoothX0;
				drawY0 = s_asPrevLipState[trackId].smoothY0;
				drawW  = s_asPrevLipState[trackId].smoothW;
				drawH  = s_asPrevLipState[trackId].smoothH;
			}
			if(profiler){
				u64StartCycle = pmu_get_systick_Count();
			}
			DrawLipLandmark(s_afSmoothedRelX, s_afSmoothedRelY, drawX0, drawY0, drawW, drawH, &infFramebuf->frameImage);

			if(profiler){
				u64EndCycle = pmu_get_systick_Count();
				info("draw face landmark cycles %llu \n", (u64EndCycle - u64StartCycle));
			}
		}
	}

	if(profiler){
		u64StartCycle = pmu_get_systick_Count();
	}

	DrawDetectFace(infFramebuf->results_FD, infFramebuf->isSpeaking, &infFramebuf->frameImage);

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("draw face region cycles %llu \n", (u64EndCycle - u64StartCycle));
	}
}

static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\face_landmark.tflite"
#define EACH_READ_SIZE 512
	
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */	
    f_chdrive(sd_path);          /* set default path */

	int32_t i32FileSize;
	int32_t i32FileReadIndex = 0;
	int32_t i32Read;
	
	if(!ModelFileReader_Initialize(MODEL_FILE))
	{
        printf_err("Unable open model %s\n", MODEL_FILE);		
		return -1;
	}
	
	i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		i32FileReadIndex += i32Read;
	}
	
	if(i32FileReadIndex < i32FileSize)
	{
        printf_err("Read Model file size is not enough\n");		
		return -2;
	}
	
#if 0
	/* verify */
	i32FileReadIndex = 0;
	ModelFileReader_Rewind();
	BYTE au8TempBuf[EACH_READ_SIZE];
	
	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)au8TempBuf, EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		
		if(std::memcmp(au8TempBuf, (void *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), i32Read)!= 0)
		{
			printf_err("verify the model file content is incorrect at %i \n", i32FileReadIndex);		
			return -3;
		}
		i32FileReadIndex += i32Read;
	}
	
#endif	
	ModelFileReader_Finish();
	
	return i32FileSize;
}	

int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

#if defined(__LOAD_MODEL_FROM_SD__)

	/* Copy model file from SD to HyperRAM*/
	int32_t i32ModelSize;

	i32ModelSize = PrepareModelToHyperRAM();
	
	if(i32ModelSize <= 0 )
	{
        printf_err("Failed to prepare model\n");
        return 1;
	}

    /* Model object creation and initialisation. */
    arm::app::FaceLandmarkModel faceLandmarkModel;

    if (!faceLandmarkModel.Init(arm::app::tensorArena_FaceLandmark,
                    sizeof(arm::app::tensorArena_FaceLandmark),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

#else
    /* Model object creation and initialisation. */
    arm::app::FaceLandmarkModel faceLandmarkModel;

    if (!faceLandmarkModel.Init(arm::app::tensorArena_FaceLandmark,
                    sizeof(arm::app::tensorArena_FaceLandmark),
					arm::app::face_landmark::GetModelPointer(),
                    arm::app::face_landmark::GetModelLen()))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }
	
#endif

    /* Model object creation and initialisation. */
    arm::app::FaceDetectionModel faceDetectionModel;

    if (!faceDetectionModel.Init(arm::app::tensorArena_FaceDetection,
                    sizeof(arm::app::tensorArena_FaceDetection),
					arm::app::face_detection::GetModelPointer(),
                    arm::app::face_detection::GetModelLen()))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }



    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_FaceLandmark),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_FaceLandmark) + FACE_LANDMARK_ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_FaceDetection),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_FaceDetection) + FACE_DETECTION_ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#if (NUM_FRAMEBUF == 2)
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)frame_buf1),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#endif
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    TfLiteIntArray *inputShape_FD = faceDetectionModel.GetInputShape(0);

    const int inputImgCols_FD = inputShape_FD->data[arm::app::FaceDetectionModel::ms_inputColsIdx];
    const int inputImgRows_FD = inputShape_FD->data[arm::app::FaceDetectionModel::ms_inputRowsIdx];

    TfLiteTensor* outputTensor0_FD = faceDetectionModel.GetOutputTensor(0);
    TfLiteTensor* outputTensor1_FD = faceDetectionModel.GetOutputTensor(1);
	
    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

    // postProcess
    arm::app::face_landmark::FaceLandmarkPostProcessing postProcess_FL(FACE_PRESENCE_THRESHOLD);

    const arm::app::face_detection::PostProcessParams postProcessParams{
            inputImgRows_FD,
            inputImgCols_FD,
            (int)s_asFramebuf[0].frameImage.h,
            (int)s_asFramebuf[0].frameImage.w,
            anchor1,
            anchor2};

	arm::app::FaceDetectorPostProcess postProcess_FD =
            arm::app::FaceDetectorPostProcess(outputTensor0_FD, outputTensor1_FD, s_asFramebuf[0].results_FD, postProcessParams);

#if defined(__PROFILE__)

    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);

#if defined (__USE_DISPLAY__)
    char szDisplayText[100];
    S_DISP_RECT sDispRect;

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
	UVC_Init();
    HSUSBD_Start();
#endif

	bool bDoFaceLandmark = false;

    while(1)
    {
        emptyFramebuf = get_empty_framebuf();

        if (emptyFramebuf)
        {
            //capture frame from CCAP
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif

            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
		}
		
        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
#if defined(__PROFILE__)
			DetectFaceRegion(
					fullFramebuf,
					&faceDetectionModel,
					&postProcess_FD,
					&profiler);
#else
			DetectFaceRegion(
					fullFramebuf,
					&faceDetectionModel,
					&postProcess_FD,
					nullptr);
#endif
			fullFramebuf->eState = eFRAMEBUF_INF;
        }
		
        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {

#if defined(__PROFILE__)
			if(infFramebuf->results_FD.size())
				DetectFaceLandmark_DrawResult(
						infFramebuf,
						&faceLandmarkModel,
						&postProcess_FL,
						&profiler);
#else
			if(infFramebuf->results_FD.size())
				DetectFaceLandmark_DrawResult(
						infFramebuf,
						&faceLandmarkModel,
						&postProcess_FL,
						nullptr);
#endif

            //display result image
#if defined (__USE_DISPLAY__)
            //Display image on LCD - scale to fill screen, centered
            {
                uint32_t lcdW = Disaplay_GetLCDWidth();
                uint32_t lcdH = Disaplay_GetLCDHeight();
                uint32_t dispW = frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR;
                uint32_t dispH = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
                /* Center the scaled image on the LCD (clamp to fit) */
                if (dispW > lcdW) dispW = lcdW;
                if (dispH > lcdH) dispH = lcdH;
                sDispRect.u32TopLeftX = (lcdW > dispW) ? ((lcdW - dispW) / 2) : 0;
                sDispRect.u32TopLeftY = (lcdH > dispH) ? ((lcdH - dispH) / 2) : 0;
                sDispRect.u32BottonRightX = sDispRect.u32TopLeftX + dispW - 1;
                sDispRect.u32BottonRightY = sDispRect.u32TopLeftY + dispH - 1;
            }


#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif

            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect,IMAGE_DISP_UPSCALE_FACTOR);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#endif

#if defined (__USE_UVC__)
			if(UVC_IsConnect())
			{
#if (UVC_Color_Format == UVC_Format_YUY2)
				image_t RGB565Img;
				image_t YUV422Img;

				RGB565Img.w = infFramebuf->frameImage.w;
				RGB565Img.h = infFramebuf->frameImage.h;
				RGB565Img.data = (uint8_t *)infFramebuf->frameImage.data;
				RGB565Img.pixfmt = PIXFORMAT_RGB565;

				YUV422Img.w = RGB565Img.w;
				YUV422Img.h = RGB565Img.h;
				YUV422Img.data = (uint8_t *)infFramebuf->frameImage.data;
				YUV422Img.pixfmt = PIXFORMAT_YUV422;
				
				roi.x = 0;
				roi.y = 0;
				roi.w = RGB565Img.w;
				roi.h = RGB565Img.h;
				imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);
				
#else
				image_t origImg;
				image_t vflipImg;

				origImg.w = infFramebuf->frameImage.w;
				origImg.h = infFramebuf->frameImage.h;
				origImg.data = (uint8_t *)infFramebuf->frameImage.data;
				origImg.pixfmt = PIXFORMAT_RGB565;

				vflipImg.w = origImg.w;
				vflipImg.h = origImg.h;
				vflipImg.data = (uint8_t *)infFramebuf->frameImage.data;
				vflipImg.pixfmt = PIXFORMAT_RGB565;

				imlib_nvt_vflip(&origImg, &vflipImg);
#endif
				UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE, uvcStatus.StillImage);				

			}

#endif

            u64PerfFrames ++;
			if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);
                //sprintf(szDisplayText,"Time %llu",(uint64_t) pmu_get_systick_Count() / (uint64_t)SystemCoreClock);
                //info("Running %s sec \n", szDisplayText);

                /* Overlay frame rate at top-left (full-screen mode) */
                sDispRect.u32TopLeftX = 0;
                sDispRect.u32TopLeftY = 0;
                sDispRect.u32BottonRightX = (FONT_WIDTH * 18 * FONT_DISP_UPSCALE_FACTOR) - 1;
                sDispRect.u32BottonRightY = (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1;

                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
                    0,
                    C_BLUE,
                    C_WHITE,
                    false,
					FONT_DISP_UPSCALE_FACTOR
                );
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
			}

            infFramebuf->eState = eFRAMEBUF_EMPTY;
		}

		//Wait CCAP ready
		if (emptyFramebuf)
		{
			//Capture new image

			ImageSensor_WaitCaptureDone();
#if defined(__PROFILE__)
			u64CCAPEndCycle = pmu_get_systick_Count();
			info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif
            emptyFramebuf->eState = eFRAMEBUF_FULL;		
		}
    }

    return 0;
}
