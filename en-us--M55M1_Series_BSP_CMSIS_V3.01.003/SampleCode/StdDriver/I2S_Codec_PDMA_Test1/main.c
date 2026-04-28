#include <stdio.h>
#include "NuMicro.h"
#include "config.h"
#include "vcom_serial.h"

#define AUDIO_FS_HZ     48000U //change based on wanted sampling speed. Original was 48000U
#define BUFF_LEN_1        64  



static NVT_NONCACHEABLE uint32_t g_PcmRxBuf[2][BUFF_LEN_1] = {0};
static NVT_NONCACHEABLE DMA_DESC_T g_RxDesc[2] = {0};
static volatile uint8_t g_RxBufIdx = 0;
static volatile uint8_t g_NewFrame = 0;
static volatile int32_t s24_probe=0;

//Vcom globals defined

STR_VCOM_LINE_CODING g_LineCoding = {115200, 0, 0, 8};
uint16_t g_u16CtrlSignal = 0;

#define RXBUFSIZE 512
#define TXBUFSIZE 512

static volatile uint8_t s_au8ComRbuf[RXBUFSIZE];
volatile uint16_t g_u16ComRbytes = 0;
volatile uint16_t g_u16ComRhead  = 0;
volatile uint16_t g_u16ComRtail  = 0;

static volatile uint8_t s_au8ComTbuf[TXBUFSIZE];
volatile uint16_t g_u16ComTbytes = 0;
volatile uint16_t g_u16ComThead  = 0;
volatile uint16_t g_u16ComTtail  = 0;

static uint8_t s_au8RxBuf[64] = {0};
uint8_t *g_pu8RxBuf = 0;
uint32_t g_u32RxSize = 0;
uint32_t g_u32TxSize = 0;

volatile int8_t g_i8BulkOutReady = 0;






static void SYS_Init(void)
{
    SYS_UnlockReg();

    /* System clock from APLL0 (same as BSP examples) */
    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0,
                    CLK_APLLCTL_APLLSRC_HXT,
                    FREQ_220MHZ);
    SystemCoreClockUpdate();

    /* GPIO clocks */
    CLK_EnableModuleClock(GPIOA_MODULE);
    CLK_EnableModuleClock(GPIOB_MODULE);
    CLK_EnableModuleClock(GPIOC_MODULE);
    CLK_EnableModuleClock(GPIOD_MODULE);
    CLK_EnableModuleClock(GPIOE_MODULE);
    CLK_EnableModuleClock(GPIOF_MODULE);
    CLK_EnableModuleClock(GPIOG_MODULE);
    CLK_EnableModuleClock(GPIOH_MODULE);
    CLK_EnableModuleClock(GPIOI_MODULE);
    CLK_EnableModuleClock(GPIOJ_MODULE);
	
		 /* USB clock = APLL1 / 2 (same as example) */
    CLK_EnableAPLL(CLK_APLLCTL_APLLSRC_HXT, FREQ_192MHZ, CLK_APLL1_SELECT);
    CLK_SetModuleClock(USBD0_MODULE,
                   CLK_USBSEL_USBSEL_APLL1_DIV2,
                   CLK_USBDIV_USBDIV(2));			 
		 //USB PHY 
    SYS->USBPHY = (SYS->USBPHY & ~SYS_USBPHY_USBROLE_Msk)
              | SYS_USBPHY_OTGPHYEN_Msk;
		SET_USB_VBUS_PA12();
    SET_USB_D_MINUS_PA13();
    SET_USB_D_PLUS_PA14();
    SET_USB_OTG_ID_PA15();
    /* I2S0 + PDMA clocks */
    CLK_EnableModuleClock(I2S0_MODULE);
    CLK_EnableModuleClock(PDMA0_MODULE);
		
		SET_USB_VBUS_PA12();
    SET_USB_D_MINUS_PA13();
    SET_USB_D_PLUS_PA14();
    SET_USB_OTG_ID_PA15();

    /* Debug UART clock + pins */
    SetDebugUartCLK();
    SetDebugUartMFP();


    SET_I2S0_BCLK_PB5();
    SET_I2S0_LRCK_PB1();
    SET_I2S0_DI_PB3();

    SYS_LockReg();
}

/* ===== PDMA0 IRQ (RX only) ===== */
NVT_ITCM void PDMA0_IRQHandler(void)
{
		uint32_t status = PDMA_GET_INT_STATUS(PDMA0);

    /* Transfer-done interrupt flag group? */
    if (status & PDMA_INTSTS_TDIF_Msk)
    {
        uint32_t td = PDMA_GET_TD_STS(PDMA0);

        /* Channel 2 (we use ch2 for I2S0 RX) */
        if (td & (1u << 2))
        {
            /* Toggle which buffer is now ACTIVE for next transfer */
            g_RxBufIdx ^= 1u;
            g_NewFrame = 1u;

            /* Clear TD flag for ch2 */
            PDMA_CLR_TD_FLAG(PDMA0, (1u << 2));
        }
    }

    /* Optional: handle abort flags if you want */
}

/* ===== PDMA init: RX only, double buffer, scatter-gather ===== */
static void PDMA_Init_For_I2S0_RX(void)
{
    /* Configure descriptors (very similar to BSP demo, RX only) */

    /* Descriptor 0 -> fills g_PcmRxBuf[0] then jumps to desc1 */
    g_RxDesc[0].ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |   /* transfer count = BUFF_LEN */
        PDMA_WIDTH_32 |                                 /* 32-bit transfers */
        PDMA_SAR_FIX  |                                 /* source addr fixed (I2S0->RXFIFO) */
        PDMA_DAR_INC  |                                 /* dest addr increments (buffer) */
        PDMA_REQ_SINGLE |                               /* single request */
        PDMA_OP_SCATTER;                                /* scatter-gather mode */
    g_RxDesc[0].src    = (uint32_t)&I2S0->RXFIFO;
    g_RxDesc[0].dest   = (uint32_t)&g_PcmRxBuf[0][0];
    g_RxDesc[0].offset = (uint32_t)&g_RxDesc[1];        /* link to desc1 */
    /* Descriptor 1 -> fills g_PcmRxBuf[1] then jumps back to desc0 */
    g_RxDesc[1].ctl =
        ((BUFF_LEN_1 - 1) << PDMA_DSCT_CTL_TXCNT_Pos) |
        PDMA_WIDTH_32 |
        PDMA_SAR_FIX  |
        PDMA_DAR_INC  |
        PDMA_REQ_SINGLE |
        PDMA_OP_SCATTER;
    g_RxDesc[1].src    = (uint32_t)&I2S0->RXFIFO;
    g_RxDesc[1].dest   = (uint32_t)&g_PcmRxBuf[1][0];
    g_RxDesc[1].offset = (uint32_t)&g_RxDesc[0];        /* link back to desc0 */

    /* Open channel 2 only (bit2) */
    PDMA_Open(PDMA0, (1u << 2));

    /* Attach ch2 to I2S0 RX, scatter-gather starting at desc0 */
    PDMA_SetTransferMode(PDMA0,
                         2,               /* channel number */
                         PDMA_I2S0_RX,    /* request source */
                         1,               /* scatter-gather */
                         (uint32_t)&g_RxDesc[0]);

    /* Enable transfer-done interrupt on ch2 */
    PDMA_EnableInt(PDMA0, 2, PDMA_INT_TRANS_DONE);

    /* Enable PDMA0 IRQ in NVIC */
    NVIC_EnableIRQ(PDMA0_IRQn);
		printf("came here\n");
}

static void USB_Send(const uint8_t *buf, uint32_t len)
{
    while (len)
    {
        uint32_t pkt = (len > EP2_MAX_PKT_SIZE) ? EP2_MAX_PKT_SIZE : len;

        /* Wait for previous IN transfer done */
        while (g_u32TxSize);
						if (g_u8Suspend){
							 return;
						}

        USBD_MemCopy(
            (uint8_t *)(USBD_BUF_BASE + USBD_GET_EP_BUF_ADDR(EP2)),
            buf,
            pkt
        );

        g_u32TxSize = pkt;
        USBD_SET_PAYLOAD_LEN(EP2, pkt);

        buf += pkt;
        len -= pkt;
    }
  }

int main(void)
{
		CLK_SysTickDelay(5000);
    SYS_Init();
    InitDebugUart();
		USBD_Open(&gsInfo, VCOM_ClassRequest, NULL);
		VCOM_Init();
    USBD_Start();
		NVIC_EnableIRQ(USBD_IRQn);

    printf("I2S0 + PDMA RX (I2S microphone) demo\n");

    /* --- Configure I2S0 as MASTER, 48 kHz, 32-bit stereo --- */
    SYS_UnlockReg();

    /* Select I2S0 clock source = HIRC (12 MHz) */
    CLK_SetModuleClock(I2S0_MODULE,
                       CLK_I2SSEL_I2S0SEL_HIRC,
                       0);

    /* Open I2S0:
       - MASTER
       - 48 kHz sample rate
       - 32-bit data
       - stereo (mic presents left/right slots; we?ll just read raw words)
       - Philips I2S format
       I2S_Open will compute BCLKDIV for us.
    */
    I2S_Open(I2S0,
             I2S_MODE_MASTER,
             AUDIO_FS_HZ,
             I2S_DATABIT_32,
             I2S_STEREO,
             I2S_FORMAT_I2S);

    /* (Optional) route mono RX to left channel if you later use I2S_GET_RX_SAMPLE() helpers
       but here we just read raw 32-bit words from buffer.
    */
    I2S_SET_MONO_RX_CHANNEL(I2S0, I2S_MONO_LEFT);

    /* Set FIFO thresholds (not critical for PDMA, but good defaults) */
    I2S_SetFIFO(I2S0,
                I2S_FIFO_TX_LEVEL_WORD_8,
                I2S_FIFO_RX_LEVEL_WORD_8);

    SYS_LockReg();

    /* --- Setup PDMA for I2S0 RX --- */
    PDMA_Init_For_I2S0_RX();
		
		
    /* Enable I2S0 RX + RXDMA
       (Master mode will generate BCLK/LRCK when RX or TX is active)
    */
    I2S_ENABLE_RXDMA(I2S0);
		I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_RX(I2S0);
		I2S_ENABLE_TX(I2S0);

    printf("I2S0 + PDMA started. Capturing audio...\n\n");


		//printf("%x\n", PDMA0->DSCT);
    /* === Main loop: whenever a PDMA frame is ready, print a subset of samples === */
    uint32_t frameCount = 0;
		//code for unlimited PCM values
		int it=0;
     while (1)
    {
			printf("came here\n");
			CLK_SysTickDelay(500);
			if (g_NewFrame)
			{
					__disable_irq();
					uint8_t buf = g_RxBufIdx ^ 1u;
					g_NewFrame = 0;
					__enable_irq();

					static uint8_t usbBuf[BUFF_LEN_1 * 3];
					uint32_t idx = 0;

					for (int i = 0; i < BUFF_LEN_1; i += 2)
					{
							int32_t s24 = ((int32_t)g_PcmRxBuf[buf][i]) >> 8;

							usbBuf[idx++] = (uint8_t)(s24);
							usbBuf[idx++] = (uint8_t)(s24 >> 8);
							usbBuf[idx++] = (uint8_t)(s24 >> 16);
					}

					USB_Send(usbBuf, idx);
			}

    }
			
		

}


