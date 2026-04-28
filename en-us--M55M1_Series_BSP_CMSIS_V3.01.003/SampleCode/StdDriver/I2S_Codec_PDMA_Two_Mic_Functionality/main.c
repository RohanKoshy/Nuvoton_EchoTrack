#include <stdio.h>
#include "NuMicro.h"
#include "config.h"
#include "/Keil/DoA.h"
#include "/Keil/motor.h"

#define AUDIO_FS_HZ     48000U //change based on wanted sampling speed. Original was 48000U
#define BUFF_LEN_1        64  
#define SAMPLES_PER_CALCULATION 256
#define BLOCKS_PER_CALCULATION  (SAMPLES_PER_CALCULATION / 32) // 8

//will do 128 Left/Right samples per DoA

//three different states of the microphone system
typedef enum {
    STATE_SILENT = 0,
    STATE_SINGLE_SPEAKER,
    STATE_MULTI_SPEAKER
} doa_state_t;


static doa_state_t g_state = STATE_SILENT;
static NVT_NONCACHEABLE uint32_t g_PcmRxBuf[2][BUFF_LEN_1] = {0};
static NVT_NONCACHEABLE DMA_DESC_T g_RxDesc[2] = {0};
static volatile uint8_t g_RxBufIdx = 0;
static volatile uint8_t g_NewFrame = 0;

// I2S1 setup for 4 mic
static NVT_NONCACHEABLE uint32_t g_PcmRxBuf1[2][BUFF_LEN_1] = {0};
static NVT_NONCACHEABLE DMA_DESC_T g_RxDesc1[2] = {0};
static volatile uint8_t g_RxBufIdx1 = 0;
static volatile uint8_t g_NewFrame1 = 0;


static volatile uint8_t Frame_Index=0;
static volatile uint8_t BufferIndex=0;
static volatile int32_t lBuffer[SAMPLES_PER_CALCULATION]= {0};
static volatile int32_t rBuffer[SAMPLES_PER_CALCULATION]= {0};
// I2S1 added buffers
static volatile int32_t lBuffer1[SAMPLES_PER_CALCULATION]= {0};
static volatile int32_t rBuffer1[SAMPLES_PER_CALCULATION]= {0};

volatile uint8_t ready0 = 0;
volatile uint8_t ready1 = 0;

// Separate indices
static volatile uint8_t Frame_Index0 = 0, BufferIndex0 = 0;
static volatile uint8_t Frame_Index1 = 0, BufferIndex1 = 0;

// Final float buffers for 4 mics
float mic0[SAMPLES_PER_CALCULATION];
float mic1[SAMPLES_PER_CALCULATION];
float mic2[SAMPLES_PER_CALCULATION];
float mic3[SAMPLES_PER_CALCULATION];



static void SYS_Init(void)
{
    SYS_UnlockReg();

    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0,
                    CLK_APLLCTL_APLLSRC_HXT,
                    FREQ_220MHZ);
    SystemCoreClockUpdate();

    /* GPIO clocks */
    CLK_EnableModuleClock(GPIOB_MODULE);

    CLK_EnableModuleClock(I2S0_MODULE);
		CLK_EnableModuleClock(I2S1_MODULE); // extra I2S for 4 mic
    CLK_EnableModuleClock(PDMA0_MODULE);

    SetDebugUartCLK();
    SetDebugUartMFP();

    SET_I2S0_BCLK_PB5();
    SET_I2S0_LRCK_PB1();
    SET_I2S0_DI_PB3();
	
		// 4 mic I2S setup
		//SET_I2S1_BCLK_PA11(); // Not used if i2s1 is a slave
		//SET_I2S1_LRCK_PB0(); // Not used if i2s1 is a slave
		SET_I2S1_DI_PA9();

    SYS_LockReg();
}

NVT_ITCM void PDMA0_IRQHandler(void)
{
		uint32_t status = PDMA_GET_INT_STATUS(PDMA0);

    //Check Transfer Interrupt flag
    if (status & PDMA_INTSTS_TDIF_Msk)
    {
        uint32_t td = PDMA_GET_TD_STS(PDMA0);

        if (td & (1u << 2))
        {
            // Toggle next active buffer to read into
            g_RxBufIdx ^= 1u;
            g_NewFrame = 1u;

            // Clear TD flag
            PDMA_CLR_TD_FLAG(PDMA0, (1u << 2));
        }
				// I2S1 case
				if (td & (1u << 3))
				{
						printf("Ch3 IRQ \n");
						g_RxBufIdx1 ^= 1u;
						g_NewFrame1 = 1u;

						PDMA_CLR_TD_FLAG(PDMA0, (1u << 3));
				}
    }
}


static void PDMA_Init_For_I2S0_RX(void)
{

    // Configure Descriptor 2
    g_RxDesc[0].ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |  
        PDMA_WIDTH_32 |                                
        PDMA_SAR_FIX  |                                 
        PDMA_DAR_INC  |                           
        PDMA_REQ_SINGLE |                              
        PDMA_OP_SCATTER;                                // scatter-gather mode
    g_RxDesc[0].src    = (uint32_t)&I2S0->RXFIFO;
    g_RxDesc[0].dest   = (uint32_t)&g_PcmRxBuf[0][0];
    g_RxDesc[0].offset = (uint32_t)&g_RxDesc[1];       
    
		//Configure Descriptor 1
		g_RxDesc[1].ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |
        PDMA_WIDTH_32 |
        PDMA_SAR_FIX  |
        PDMA_DAR_INC  |
        PDMA_REQ_SINGLE |
        PDMA_OP_SCATTER;
    g_RxDesc[1].src    = (uint32_t)&I2S0->RXFIFO;
    g_RxDesc[1].dest   = (uint32_t)&g_PcmRxBuf[1][0];
    g_RxDesc[1].offset = (uint32_t)&g_RxDesc[0];     
    
		// Open channel 2 
    PDMA_Open(PDMA0, (1u << 2));

    PDMA_SetTransferMode(PDMA0,
                         2,
                         PDMA_I2S0_RX,
                         1,               //scatter-gather configuration
                         (uint32_t)&g_RxDesc[0]);

    PDMA_EnableInt(PDMA0, 2, PDMA_INT_TRANS_DONE);

    NVIC_EnableIRQ(PDMA0_IRQn);
		//printf("came here\n");
}





// PDMA for I2S1
static void PDMA_Init_For_I2S1_RX(void)
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
    PDMA_Open(PDMA0, (1u << 3));

    PDMA_SetTransferMode(PDMA0,
                         3,
                         PDMA_I2S1_RX,
                         1,
                         (uint32_t)&g_RxDesc1[0]);

    PDMA_EnableInt(PDMA0, 3, PDMA_INT_TRANS_DONE);
		
		NVIC_EnableIRQ(PDMA0_IRQn);
}




// 2 mic implementation

int main(void)
{
    SYS_Init();
    InitDebugUart();

    printf("I2S0 + PDMA RX (I2S microphone) demo\n");
    SYS_UnlockReg();
		Servo_PWM_Init();

    //clock source = HIRC (12 MHz) 
    CLK_SetModuleClock(I2S0_MODULE,
                       CLK_I2SSEL_I2S0SEL_HIRC,
                       0);
	
 
    I2S_Open(I2S0,
             I2S_MODE_MASTER,
             AUDIO_FS_HZ,
             I2S_DATABIT_32,
             I2S_STEREO,
             I2S_FORMAT_I2S);
						 

    //I2S_SET_MONO_RX_CHANNEL(I2S0, I2S_MONO_LEFT);
		
    I2S_SetFIFO(I2S0,
                I2S_FIFO_TX_LEVEL_WORD_8,
                I2S_FIFO_RX_LEVEL_WORD_8);
								

    SYS_LockReg();


    PDMA_Init_For_I2S0_RX();
	
    I2S_ENABLE_RXDMA(I2S0);
		I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_RX(I2S0);
		I2S_ENABLE_TX(I2S0);
		

    printf("I2S0 + PDMA started. Capturing audio...\n\n");


		//printf("%x\n", PDMA0->DSCT);
    uint32_t frameCount = 0;
		//code for unlimited PCM values
		int it=0;
		doa_init();
		//doa_run_synthetic_delay_test();
		//while (1) { }
    while (1)
    {
			if (g_NewFrame)
			{
            it++;
						Frame_Index++;
						__disable_irq();
            uint8_t buf = g_RxBufIdx ^ 1u;
            g_NewFrame = 0;
						__enable_irq();
						for(int i=0;i<BUFF_LEN_1;i+=2)
						{
							//int32_t s24L = ((int32_t)calcBuffer[Frame_Index][i]) >> 8;
							//int32_t s24R =  ((int32_t)calcBuffer[Frame_Index][i+1]) >> 8;
								lBuffer[BufferIndex] = ((int32_t)g_PcmRxBuf[buf][i]) >> 8;
								rBuffer[BufferIndex] = ((int32_t)g_PcmRxBuf[buf][i+1]) >> 8;
								BufferIndex++;
							//int32_t s24R =  ((int32_t)calcBuffer[Frame_Index][i+1]) >> 8;
						}
						//printf("came here %u\n", Frame_Index);
						if(Frame_Index>=BLOCKS_PER_CALCULATION)
						{
							//printf("went through %u\n", Frame_Index);
							Frame_Index=0;
							BufferIndex=0;
							 for (int n = 0; n <SAMPLES_PER_CALCULATION ; n++)
							{
                // scale 24-bit to [-1, 1)
                left_buffer[n]  = (float)lBuffer[n] / 8388608.0f;  // 2^23
                right_buffer[n] = (float)rBuffer[n] / 8388608.0f;
							}
							//printf("lbuffer 0: %f lbuffer 255: %f\n", left_buffer[0], left_buffer[255]);
							//printf("rbuffer 0: %f rbuffer 255: %f", right_buffer[0], right_buffer[255]);
							float angle = doa_process_one_frame();
							if(angle==-180.0f)
							{
								if(g_state!=STATE_SILENT)
								{
									printf("silent\n");
								}
								g_state = STATE_SILENT;
							}
							else
							{
									g_state = STATE_SINGLE_SPEAKER;
									if(angle==-79.95f)
									{
										continue;
									}
									Servo_SetAngle(-angle);
							}
							//printf("DOA = %.2f deg\n", angle);
							Frame_Index=0;
							BufferIndex=0;
						}
            frameCount++;
        }
    }
}







/*
// ===== Main ===== 

// 4 mic implementation
int main(void)
{
    SYS_Init();
    InitDebugUart();

    printf("I2S0 + PDMA RX (I2S microphone) demo\n");
    SYS_UnlockReg();
		Servo_PWM_Init();

    //clock source = HIRC (12 MHz) 
    CLK_SetModuleClock(I2S0_MODULE,
                       CLK_I2SSEL_I2S0SEL_HIRC,
                       0);
	
		// I2S1 setup for 4 mic
		CLK_SetModuleClock(I2S1_MODULE,
                       CLK_I2SSEL_I2S1SEL_HIRC,
                       0);

    I2S_Open(I2S0,
             I2S_MODE_MASTER,
             AUDIO_FS_HZ,
             I2S_DATABIT_32,
             I2S_STEREO,
             I2S_FORMAT_I2S);
						 
		// I2S1 setup for 4 mic
		I2S_Open(I2S1,
             I2S_MODE_SLAVE,
             AUDIO_FS_HZ,
             I2S_DATABIT_32,
             I2S_STEREO,
             I2S_FORMAT_I2S);


    //I2S_SET_MONO_RX_CHANNEL(I2S0, I2S_MONO_LEFT);
		
    I2S_SetFIFO(I2S0,
                I2S_FIFO_TX_LEVEL_WORD_8,
                I2S_FIFO_RX_LEVEL_WORD_8);
								
		// I2S1 setup for 4 mic
		I2S_SetFIFO(I2S1,
                I2S_FIFO_TX_LEVEL_WORD_8,
                I2S_FIFO_RX_LEVEL_WORD_8);

    SYS_LockReg();


    PDMA_Init_For_I2S0_RX();
		PDMA_Init_For_I2S1_RX();
	
    I2S_ENABLE_RXDMA(I2S0);
		I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_RX(I2S0);
		I2S_ENABLE_TX(I2S0);
		
		// I2S1 setup
		I2S_ENABLE_RXDMA(I2S1);
		I2S_ENABLE_TXDMA(I2S1);
		I2S_ENABLE_RX(I2S1);
		I2S_ENABLE_TX(I2S1);

    printf("I2S0 + PDMA started. Capturing audio...\n\n");


		//printf("%x\n", PDMA0->DSCT);
		//code for unlimited PCM values
		doa_init();
		//doa_run_synthetic_delay_test();
		while (1)
		{
				// printf("%x\n", I2S1->RXFIFO);
				// ================= I2S0 =================
				if (g_NewFrame)
				{
						// printf("I2S0 Frame \n");
						__disable_irq();
						uint8_t buf = g_RxBufIdx ^ 1u;
						g_NewFrame = 0;
						__enable_irq();

						for (int i = 0; i < BUFF_LEN_1; i += 2)
						{
								lBuffer[BufferIndex0] = ((int32_t)g_PcmRxBuf[buf][i]) >> 8;
								rBuffer[BufferIndex0] = ((int32_t)g_PcmRxBuf[buf][i+1]) >> 8;
								BufferIndex0++;
						}

						Frame_Index0++;

						if (Frame_Index0 >= BLOCKS_PER_CALCULATION)
						{
								Frame_Index0 = 0;
								BufferIndex0 = 0;

								for (int n = 0; n < SAMPLES_PER_CALCULATION; n++)
								{
										mic0[n] = (float)lBuffer[n] / 8388608.0f;
										mic1[n] = (float)rBuffer[n] / 8388608.0f;
								}

								ready0 = 1;
						}
				}

				// ================= I2S1 =================
				if (g_NewFrame1)
				{
						// printf("I2S1 Frame \n");
						__disable_irq();
						uint8_t buf = g_RxBufIdx1 ^ 1u;
						g_NewFrame1 = 0;
						__enable_irq();

						for (int i = 0; i < BUFF_LEN_1; i += 2)
						{
								lBuffer1[BufferIndex1] = ((int32_t)g_PcmRxBuf1[buf][i]) >> 8;
								rBuffer1[BufferIndex1] = ((int32_t)g_PcmRxBuf1[buf][i+1]) >> 8;
								BufferIndex1++;
						}

						Frame_Index1++;

						if (Frame_Index1 >= BLOCKS_PER_CALCULATION)
						{
								Frame_Index1 = 0;
								BufferIndex1 = 0;

								for (int n = 0; n < SAMPLES_PER_CALCULATION; n++)
								{
										mic2[n] = (float)lBuffer1[n] / 8388608.0f;
										mic3[n] = (float)rBuffer1[n] / 8388608.0f;
								}

								ready1 = 1;
						}
				}
				

				// ================= COMBINED PROCESS =================
				
				static int print_counter = 0;
				
				
				if (ready0 && ready1)
				{
						ready0 = 0;
						ready1 = 0;

					
						print_counter++;

						// Print only occasionally to avoid flooding UART
						if (print_counter % 20 == 0)
						{
								for (int i = 0; i < SAMPLES_PER_CALCULATION; i += 16)
								{
										printf("[%3d] m0=%f m1=%f m2=%f m3=%f\n",
													 i,
													 mic0[i],
													 mic1[i],
													 mic2[i],
													 mic3[i]);

										CLK_SysTickDelay(5000);
								}
								printf("\n");
						}
					
					
						// ?? THIS is where 4-mic processing happens
						float angle = doa_process_4mic(mic0, mic1, mic2, mic3);

						if (angle == -180.0f)
						{
								if (g_state != STATE_SILENT)
										printf("silent\n");

								g_state = STATE_SILENT;
						}
						else
						{
								g_state = STATE_SINGLE_SPEAKER;
								Servo_SetAngle(-angle);
						}
				}
		}
}
*/
	
	
/*
int main(void)
{
    SYS_Init();
    InitDebugUart();

    printf("I2S0 + PDMA RX (I2S microphone) demo\n");
    SYS_UnlockReg();

    //clock source = HIRC (12 MHz) 
    CLK_SetModuleClock(I2S0_MODULE,
                       CLK_I2SSEL_I2S0SEL_HIRC,
                       0);

    
    I2S_Open(I2S0,
             I2S_MODE_MASTER,
             AUDIO_FS_HZ,
             I2S_DATABIT_32,
             I2S_STEREO,
             I2S_FORMAT_I2S);


    //I2S_SET_MONO_RX_CHANNEL(I2S0, I2S_MONO_LEFT);
		
    I2S_SetFIFO(I2S0,
                I2S_FIFO_TX_LEVEL_WORD_8,
                I2S_FIFO_RX_LEVEL_WORD_8);

    SYS_LockReg();


    PDMA_Init_For_I2S0_RX();
		
	
    I2S_ENABLE_RXDMA(I2S0);
		I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_RX(I2S0);
		I2S_ENABLE_TX(I2S0);

    printf("I2S0 + PDMA started. Capturing audio...\n\n");


		//printf("%x\n", PDMA0->DSCT);
    uint32_t frameCount = 0;
		//code for unlimited PCM values
		int it=0;
    while (1)
    {
			
			if (g_NewFrame)
			{
            it++;
						__disable_irq();
            uint8_t buf = g_RxBufIdx ^ 1u;
            g_NewFrame = 0;
						uint32_t calcBuffer[BUFF_LEN_1];
						for(int i=0;i<BUFF_LEN_1;i++)
						{
							calcBuffer[i]=g_PcmRxBuf[buf][i];
						}
            __enable_irq();

            frameCount++;
            for (int i = 0; i < BUFF_LEN_1; i += 2)
            {
                uint32_t rawL = g_PcmRxBuf[buf][i];
								uint32_t rawR=g_PcmRxBuf[buf][i+1];

                int32_t s24L = ((int32_t)rawL) >> 8;
								int32_t s24R = ((int32_t)rawR) >> 8;
								if(i%(BUFF_LEN_1)==0)
								{
									
									printf("[%3d] s24L=%x  s24R=%x\n",
                       i,
                       (long)s24L,
                       (long)s24R);
											 CLK_SysTickDelay(5000); 
										 }

            }
        }
    }
			
		

}
*/
 


