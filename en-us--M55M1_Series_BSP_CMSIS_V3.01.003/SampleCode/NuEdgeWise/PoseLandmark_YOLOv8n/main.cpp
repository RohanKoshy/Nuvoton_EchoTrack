/**************************************************************************//**
 * @file     main.cpp
 * @version  V2.00
 * @brief    Thin shell: camera capture → SpeakingDetector → display.
 *           All ML logic lives in SpeakingDetector.cpp.
 *           Also runs DoA on I2S0/I2S1 microphones between frames.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "BoardInit.hpp"
#include "SpeakingDetector.hpp"
#include "log_macros.h"
#include "imlib.h"
#include "framebuffer.h"
//#include <intrins.h>

#undef PI
#include "NuMicro.h"

#include "Profiler.hpp"
#include "ImageSensor.h"

#include <vector>
#include <cstring>

/* ---- DoA module (pure C) ---- */
extern "C" {
    #include "config.h"
    #include "/Keil/DoA.h"
    #include "/Keil/motor.h"
    #include "/Keil/microphones.h"
}

//#define __PROFILE__
#define __USE_DISPLAY__
//#define __USE_UVC__

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

/* ------------------------------------------------------------------ */
/*  Frame buffer management                                            */
/* ------------------------------------------------------------------ */
#define NUM_FRAMEBUF 2

typedef enum {
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct {
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
} S_FRAMEBUF;

static S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

static S_FRAMEBUF *get_empty_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i++)
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY) return &s_asFramebuf[i];
    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i++)
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL) return &s_asFramebuf[i];
    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i++)
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF) return &s_asFramebuf[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Display / image constants                                          */
/* ------------------------------------------------------------------ */
#define IMAGE_DISP_UPSCALE_FACTOR 2
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

#if defined(__USE_UVC__)
#define GLCD_WIDTH   320
#define GLCD_HEIGHT  240
#else
#define GLCD_WIDTH   320
#define GLCD_HEIGHT  240
#endif

#define IMAGE_FB_SIZE  (GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef  OMV_FB_SIZE
#define OMV_FB_SIZE    (IMAGE_FB_SIZE + 1024)

#undef  OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE (1 * 1024)

__attribute__((section(".bss.vram.data"), aligned(32)))
static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];

__attribute__((section(".bss.vram.data"), aligned(32)))
static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
__attribute__((section(".bss.vram.data"), aligned(32)))
static char frame_buf1[OMV_FB_SIZE];
#endif

char *_fb_base  = NULL;
char *_fb_end   = NULL;
char *_jpeg_buf = NULL;
char *_fballoc  = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end  = fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();
    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0; i < NUM_FRAMEBUF; i++)
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

/* ------------------------------------------------------------------ */
/*  DoA audio-path init                                                */
/*  Pin MFPs and module clocks are already handled by Mic_Sys_Init.   */
/*  This function only configures I2S peripherals + PDMA.              */
/* ------------------------------------------------------------------ */

extern volatile uint8_t speech_detected;

static void DoA_AudioPath_Init()
{
    SYS_UnlockReg();

    /* I2S module clock source: HIRC (12 MHz) */
    //CLK_SetModuleClock(I2S0_MODULE, CLK_I2SSEL_I2S0SEL_HIRC, 0);
    //CLK_SetModuleClock(I2S1_MODULE, CLK_I2SSEL_I2S1SEL_HIRC, 0);
		CLK_SetModuleClock(I2S0_MODULE, CLK_I2SSEL_I2S0SEL_HIRC, 0);
		CLK_SetModuleClock(I2S1_MODULE, CLK_I2SSEL_I2S1SEL_HIRC, 0);

		printf("I2SSEL after clock set = %08X\n", (unsigned)CLK->I2SSEL);

		uint32_t actual0 = I2S_Open(I2S0, I2S_MODE_MASTER, AUDIO_FS_HZ,
                            I2S_DATABIT_32, I2S_STEREO, I2S_FORMAT_I2S);
		printf("I2S0 Open returned %u (requested %u)\n",
    (unsigned)actual0, (unsigned)AUDIO_FS_HZ);
		printf("I2S0 CTL0=%08X CLKDIV=%08X\n",
    (unsigned)I2S0->CTL0, (unsigned)I2S0->CLKDIV);
    /* I2S0 master */

    /* I2S1 slave 
    I2S_Open(I2S1,
             I2S_MODE_SLAVE,
             AUDIO_FS_HZ,
             I2S_DATABIT_32,
             I2S_STEREO,
             I2S_FORMAT_I2S);
		*/
    I2S_SetFIFO(I2S0,
                I2S_FIFO_TX_LEVEL_WORD_8,
                I2S_FIFO_RX_LEVEL_WORD_8);
								
		/*						
    I2S_SetFIFO(I2S1,
                I2S_FIFO_TX_LEVEL_WORD_8,
                I2S_FIFO_RX_LEVEL_WORD_8);
		*/
    SYS_LockReg();

    PDMA_Init_For_I2S0_RX();
    //PDMA_Init_For_I2S1_RX();

    I2S_ENABLE_RXDMA(I2S0);
    I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_RX(I2S0);
    I2S_ENABLE_TX(I2S0);

    //I2S_ENABLE_RXDMA(I2S1);
    //I2S_ENABLE_TXDMA(I2S1);
    //I2S_ENABLE_RX(I2S1);
    //I2S_ENABLE_TX(I2S1);
}

/* ------------------------------------------------------------------ */
/*  Collect one full 256-sample DoA frame and process it.              */
/*  Blocks until 8 PDMA buffers have been collected contiguously.      */
/* ------------------------------------------------------------------ */
static void doa_collect_and_process()
{
    Frame_Index = 0;
    BufferIndex = 0;

    while (Frame_Index < BLOCKS_PER_CALCULATION) {
        if (g_NewFrame) {
            __disable_irq();
            uint8_t buf = g_RxBufIdx ^ 1u;
            g_NewFrame = 0;
            __enable_irq();

            for (int i = 0; i < BUFF_LEN_1; i += 2) {
                lBuffer[BufferIndex] = ((int32_t)g_PcmRxBuf[buf][i])     >> 8;
                rBuffer[BufferIndex] = ((int32_t)g_PcmRxBuf[buf][i + 1]) >> 8;
                BufferIndex++;
            }
            Frame_Index++;
        }
    }

    for (int n = 0; n < SAMPLES_PER_CALCULATION; n++) {
        left_buffer[n]  = (float)lBuffer[n] / 8388608.0f;
        right_buffer[n] = (float)rBuffer[n] / 8388608.0f;
    }

    float angle = doa_process_one_frame();
    if (angle == -180.0f) {
        if (g_state != STATE_SILENT) {
            printf("silent\n");
        }
        g_state = STATE_SILENT;
    } else {
        g_state = STATE_SINGLE_SPEAKER;
        Servo_SetAngle(90-angle);
    }
}


/* ================================================================== */
/*  main — capture → DoA → ML → display loop                          */
/* ================================================================== */
extern "C" int nu_pdma_mempush(void *dest, void *src,
                                uint32_t data_width, unsigned int transfer_count);

int main()
{
    BoardInit();
    info("main: BoardInit done\n");
    omv_init();
    Servo_PWM_Init();
    image_t frameBuffer;
    framebuffer_init_image(&frameBuffer);

    /* --- MPU setup (tensor arenas + frame buffers, one call) --- */
    {
        void *faceArena, *landmarkArena;
        uint32_t faceSize, landmarkSize;
        SpeakingDetector_GetTensorArenas(&faceArena, &faceSize,
                                         &landmarkArena, &landmarkSize);

        const std::vector<ARM_MPU_Region_t> mpuConfig = {
            {
                ARM_MPU_RBAR((unsigned int)faceArena,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)faceArena + faceSize - 1,
                             eMPU_ATTR_CACHEABLE_WTRA)
            },
            {
                ARM_MPU_RBAR((unsigned int)landmarkArena,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)landmarkArena + landmarkSize - 1,
                             eMPU_ATTR_CACHEABLE_WTRA)
            },
            {
                ARM_MPU_RBAR((unsigned int)fb_array,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)fb_array + OMV_FB_SIZE - 1,
                             eMPU_ATTR_NON_CACHEABLE)
            },
#if (NUM_FRAMEBUF == 2)
            {
                ARM_MPU_RBAR((unsigned int)frame_buf1,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)frame_buf1 + OMV_FB_SIZE - 1,
                             eMPU_ATTR_NON_CACHEABLE)
            },
#endif
        };
        InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());
    }

    /* --- Initialise the speaking detector module --- */
    int rc = SpeakingDetector_Init(NULL);
    if (rc != 0) {
        printf_err("SpeakingDetector_Init failed (%d)\n", rc);
        return 1;
    }

    /* --- Camera --- */
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

    /* --- Prime BSP PDMA framework before claiming our channel --- */
    {
        static uint32_t dummy_src = 0xDEADBEEF;
        static uint32_t dummy_dst = 0;
        nu_pdma_mempush(&dummy_dst, &dummy_src, 32, 1);
    }

    /* --- DoA audio init (after BoardInit / camera / display / PDMA prime) --- */
    Mic_Sys_Init();
    DoA_AudioPath_Init();
    doa_init();
    info("main: DoA audio path initialised\n");

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle, u64EndCycle;
    uint64_t u64CCAPStartCycle, u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle = pmu_get_systick_Count()
                          + (SystemCoreClock * EACH_PERF_SEC);
    uint64_t u64PerfFrames = 0;

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

    SpeakingFaceResult faceResults[MAX_TRACKED_FACES];
    int numFaces = 0;

    /* ---- Main loop ---- */
		uint8_t ml_it=0;
    while (1)
    {
				
        /* 1. Trigger camera capture (runs in parallel via CCAP hardware) */
        emptyFramebuf = get_empty_framebuf();
        if (emptyFramebuf) {
            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
        }

        /* 2. Collect + process one DoA frame on contiguous audio (~5.5 ms).
         *    Camera capture continues in parallel during this. */
        doa_collect_and_process();

        /* 3. Run ML on a full buffer */
				if(ml_it%10==0)
				{
					fullFramebuf = get_full_framebuf();
					if (fullFramebuf) {
							numFaces = SpeakingDetector_RunFrame(
									fullFramebuf->frameImage.data,
									fullFramebuf->frameImage.w,
									fullFramebuf->frameImage.h,
									faceResults, MAX_TRACKED_FACES);

							fullFramebuf->eState = eFRAMEBUF_INF;
					}
				}
        /* 4. Draw + display an inference-done buffer */
				for(uint8_t i=0;i<MAX_TRACKED_FACES;i++)
				{
					faceResults[i].isSpeaking=(faceResults[i].isSpeaking&speech_detected);
				}
					
				
        infFramebuf = get_inf_framebuf();
        if (infFramebuf) {
            SpeakingDetector_Draw(
                infFramebuf->frameImage.data,
                infFramebuf->frameImage.w,
                infFramebuf->frameImage.h,
                faceResults, numFaces);
					

#if defined (__USE_DISPLAY__)
            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
            sDispRect.u32BottonRightX = ((frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
            sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect,
                             IMAGE_DISP_UPSCALE_FACTOR);
#endif

#if defined (__USE_UVC__)
            if (UVC_IsConnect()) {
#if (UVC_Color_Format == UVC_Format_YUY2)
                image_t RGB565Img, YUV422Img;
                rectangle_t uvcRoi;

                RGB565Img.w = infFramebuf->frameImage.w;
                RGB565Img.h = infFramebuf->frameImage.h;
                RGB565Img.data = (uint8_t *)infFramebuf->frameImage.data;
                RGB565Img.pixfmt = PIXFORMAT_RGB565;

                YUV422Img.w = RGB565Img.w;
                YUV422Img.h = RGB565Img.h;
                YUV422Img.data = (uint8_t *)infFramebuf->frameImage.data;
                YUV422Img.pixfmt = PIXFORMAT_YUV422;

                uvcRoi.x = 0;  uvcRoi.y = 0;
                uvcRoi.w = RGB565Img.w;  uvcRoi.h = RGB565Img.h;
                imlib_nvt_scale(&RGB565Img, &YUV422Img, &uvcRoi);
#else
                image_t origImg, vflipImg;

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
                UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE,
                              uvcStatus.StillImage);
            }
#endif /* __USE_UVC__ */

            /* Frame-rate counter */
            u64PerfFrames++;
            if ((uint64_t)pmu_get_systick_Count() > u64PerfCycle) {
#if defined (__USE_DISPLAY__)
                sDispRect.u32TopLeftX = 0;
                sDispRect.u32TopLeftY = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
                sDispRect.u32BottonRightX = (frameBuffer.w);
                sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR)
                                             + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(szDisplayText, strlen(szDisplayText),
                                0, frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR,
                                C_BLUE, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count()
                             + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
            }

            infFramebuf->eState = eFRAMEBUF_EMPTY;
        }

        /* 5. Wait for camera capture to finish */
        if (emptyFramebuf) {
            ImageSensor_WaitCaptureDone();
            emptyFramebuf->eState = eFRAMEBUF_FULL;
        }
				ml_it++;
    }
		
    return 0;
}

