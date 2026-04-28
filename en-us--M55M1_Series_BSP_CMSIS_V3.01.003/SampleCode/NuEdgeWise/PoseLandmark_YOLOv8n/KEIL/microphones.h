#ifndef MICROPHONES_H
#define MICROPHONES_H

#include <stdint.h>
#include "NuMicro.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_FS_HZ     48000U
#define BUFF_LEN_1        64  
#define SAMPLES_PER_CALCULATION 256
#define BLOCKS_PER_CALCULATION  (SAMPLES_PER_CALCULATION / 32)

typedef enum {
    STATE_SILENT = 0,
    STATE_SINGLE_SPEAKER,
    STATE_MULTI_SPEAKER
} doa_state_t;

/* ===== Shared state (defined in microphones.c) ===== */
extern doa_state_t g_state;

extern volatile uint8_t g_RxBufIdx;
extern volatile uint8_t g_NewFrame;
extern volatile uint8_t g_RxBufIdx1;
extern volatile uint8_t g_NewFrame1;

extern volatile uint8_t Frame_Index;
extern volatile uint8_t BufferIndex;
extern volatile int32_t lBuffer[SAMPLES_PER_CALCULATION];
extern volatile int32_t rBuffer[SAMPLES_PER_CALCULATION];
extern volatile int32_t lBuffer1[SAMPLES_PER_CALCULATION];
extern volatile int32_t rBuffer1[SAMPLES_PER_CALCULATION];

extern volatile uint8_t ready0;
extern volatile uint8_t ready1;

extern volatile uint8_t Frame_Index0, BufferIndex0;
extern volatile uint8_t Frame_Index1, BufferIndex1;

extern float mic0[SAMPLES_PER_CALCULATION];
extern float mic1[SAMPLES_PER_CALCULATION];
extern float mic2[SAMPLES_PER_CALCULATION];
extern float mic3[SAMPLES_PER_CALCULATION];

/* Also expose the raw PDMA buffers, because the main loop reads from them */
extern uint32_t g_PcmRxBuf[2][BUFF_LEN_1];
extern uint32_t g_PcmRxBuf1[2][BUFF_LEN_1];

/* ===== Setup functions ===== */
void Mic_Sys_Init(void);
void PDMA_Init_For_I2S0_RX(void);
void PDMA_Init_For_I2S1_RX(void);

#ifdef __cplusplus
}
#endif

#endif /* MICROPHONES_H */