#include <stdio.h>
#include "NuMicro.h"
#include "config.h"
#include "/Keil/DoA.h"
#include "/Keil/motor.h"
#include "microphones.h"

doa_state_t g_state = STATE_SILENT;

//NVT_NONCACHEABLE uint32_t g_PcmRxBuf[2][BUFF_LEN_1] = {0};
//static NVT_NONCACHEABLE DMA_DESC_T g_RxDesc[2] = {0};

NVT_NONCACHEABLE __attribute__((aligned(32)))
uint32_t g_PcmRxBuf[2][BUFF_LEN_1] = {0};

//static NVT_NONCACHEABLE __attribute__((aligned(32)))
//DMA_DESC_T g_RxDesc[2] = {0};
static NVT_NONCACHEABLE __attribute__((aligned(32))) DMA_DESC_T g_RxDesc_0 = {0};
static NVT_NONCACHEABLE __attribute__((aligned(32))) DMA_DESC_T g_RxDesc_1 = {0};
//NVT_NONCACHEABLE __attribute__((aligned(32)))
//uint32_t g_PcmRxBuf1[2][BUFF_LEN_1] = {0};

//static NVT_NONCACHEABLE __attribute__((aligned(32)))
//DMA_DESC_T g_RxDesc1[2] = {0};
volatile uint8_t g_RxBufIdx = 0;
volatile uint8_t g_NewFrame = 0;

// I2S1 setup for 4 mic
NVT_NONCACHEABLE uint32_t g_PcmRxBuf1[2][BUFF_LEN_1] = {0};
static NVT_NONCACHEABLE DMA_DESC_T g_RxDesc1[2] = {0};
volatile uint8_t g_RxBufIdx1 = 0;
volatile uint8_t g_NewFrame1 = 0;


volatile uint8_t Frame_Index=0;
volatile uint8_t BufferIndex=0;
volatile int32_t lBuffer[SAMPLES_PER_CALCULATION]= {0};
volatile int32_t rBuffer[SAMPLES_PER_CALCULATION]= {0};
// I2S1 added buffers
volatile int32_t lBuffer1[SAMPLES_PER_CALCULATION]= {0};
volatile int32_t rBuffer1[SAMPLES_PER_CALCULATION]= {0};

volatile uint8_t ready0 = 0;
volatile uint8_t ready1 = 0;

// Separate indices
volatile uint8_t Frame_Index0 = 0, BufferIndex0 = 0;
volatile uint8_t Frame_Index1 = 0, BufferIndex1 = 0;

// Final float buffers for 4 mics
float mic0[SAMPLES_PER_CALCULATION];
float mic1[SAMPLES_PER_CALCULATION];
float mic2[SAMPLES_PER_CALCULATION];
float mic3[SAMPLES_PER_CALCULATION];



void Mic_Sys_Init(void)
{
    SYS_UnlockReg();
		/*
    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0,
                CLK_APLLCTL_APLLSRC_HXT,
                FREQ_220MHZ);
		*/
		SystemCoreClockUpdate();
	
   
    CLK_EnableModuleClock(GPIOB_MODULE);

    CLK_EnableModuleClock(I2S0_MODULE);
		CLK_EnableModuleClock(I2S1_MODULE); // extra I2S for 4 mic
    CLK_EnableModuleClock(PDMA1_MODULE);

    //SetDebugUartCLK();
    //SetDebugUartMFP();

    SET_I2S0_BCLK_PB5();
    SET_I2S0_LRCK_PB1();
    SET_I2S0_DI_PB3();
	
		// 4 mic I2S setup
		//SET_I2S1_BCLK_PA11(); // Not used if i2s1 is a slave
		//SET_I2S1_LRCK_PB0(); // Not used if i2s1 is a slave
		SET_I2S1_DI_PA9();

    SYS_LockReg();
}



NVT_ITCM void PDMA1_IRQHandler(void)
{
		uint32_t status = PDMA_GET_INT_STATUS(PDMA1);
		

		
    //Check Transfer Interrupt flag
    if (status & PDMA_INTSTS_TDIF_Msk)
    {
        uint32_t td = PDMA_GET_TD_STS(PDMA1);

        if (td & (1u << 15))
        {
            // Toggle next active buffer to read into
            g_RxBufIdx ^= 1u;
            g_NewFrame = 1u;

            // Clear TD flag
            PDMA_CLR_TD_FLAG(PDMA1, (1u << 15));
        }
				// I2S1 case
				if (td & (1u << 15))
				{
						g_RxBufIdx1 ^= 1u;
						g_NewFrame1 = 1u;

						PDMA_CLR_TD_FLAG(PDMA1, (1u << 3));
				}
				
				
    }

}

void PDMA_Init_For_I2S0_RX(void)
{

    // Configure Descriptor 2
    g_RxDesc_0.ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |  
        PDMA_WIDTH_32 |                                
        PDMA_SAR_FIX  |                                 
        PDMA_DAR_INC  |
				PDMA_BURST_1	|	
        PDMA_REQ_SINGLE |                              
        PDMA_OP_SCATTER;                                // scatter-gather mode
    g_RxDesc_0.src    = (uint32_t)&I2S0->RXFIFO;
    g_RxDesc_0.dest   = (uint32_t)&g_PcmRxBuf[0][0];
    g_RxDesc_0.offset = (uint32_t)&g_RxDesc_1;       
    
		//Configure Descriptor 1
		g_RxDesc_1.ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |
        PDMA_WIDTH_32 |
        PDMA_SAR_FIX  |
        PDMA_DAR_INC  |
				PDMA_BURST_1	|
        PDMA_REQ_SINGLE |
        PDMA_OP_SCATTER;
    g_RxDesc_1.src    = (uint32_t)&I2S0->RXFIFO;
    g_RxDesc_1.dest   = (uint32_t)&g_PcmRxBuf[1][0];
    g_RxDesc_1.offset = (uint32_t)&g_RxDesc_0;     
    //printf("desc0.ctl=%08X desc1.ctl=%08X (expect burst=1, single, scatter)\n",(unsigned)g_RxDesc_0.ctl, (unsigned)g_RxDesc_1.ctl);
		// Open channel 2 
    PDMA_Open(PDMA1, (1u << 15));

    PDMA_SetTransferMode(PDMA1,
                         15,
                         PDMA_I2S0_RX,
                         1,               //scatter-gather configuration
                         (uint32_t)&g_RxDesc_0);
		PDMA_DisableTimeout(PDMA1, (1u << 15));

    PDMA_EnableInt(PDMA1, 15, PDMA_INT_TRANS_DONE);

    NVIC_EnableIRQ(PDMA1_IRQn);
		//printf("PDMA1 INTEN=%08X\n", (unsigned)PDMA1->INTEN);
		//printf("came here\n");
}




// PDMA for I2S1
void PDMA_Init_For_I2S1_RX(void)
{
    g_RxDesc1[0].ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |
        PDMA_WIDTH_32 |
        PDMA_SAR_FIX |
        PDMA_DAR_INC |
        PDMA_REQ_SINGLE |
        PDMA_OP_SCATTER;

    g_RxDesc1[0].src    = (uint32_t)&I2S1->RXFIFO;
    g_RxDesc1[0].dest   = (uint32_t)&g_PcmRxBuf1[0][0];
    g_RxDesc1[0].offset = (uint32_t)&g_RxDesc1[1];

    g_RxDesc1[1].ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |
        PDMA_WIDTH_32 |
        PDMA_SAR_FIX |
        PDMA_DAR_INC |
        PDMA_REQ_SINGLE |
        PDMA_OP_SCATTER;

    g_RxDesc1[1].src    = (uint32_t)&I2S1->RXFIFO;
    g_RxDesc1[1].dest   = (uint32_t)&g_PcmRxBuf1[1][0];
    g_RxDesc1[1].offset = (uint32_t)&g_RxDesc1[0];

    // Use channel 3 instead of 2
    PDMA_Open(PDMA1, (1u << 3));

    PDMA_SetTransferMode(PDMA1,
                         3,
                         PDMA_I2S1_RX,
                         1,
                         (uint32_t)&g_RxDesc1[0]);

    PDMA_EnableInt(PDMA1, 3, PDMA_INT_TRANS_DONE);
}