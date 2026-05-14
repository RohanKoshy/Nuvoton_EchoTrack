#include "motor.h"
#include "NuMicro.h"

#define SERVO_FREQ 50   // 50 Hz

void Servo_PWM_Init(void)
{
    /* Enable EPWM1 clock */
    CLK_EnableModuleClock(EPWM1_MODULE);

    /* Select EPWM1 clock source */
    CLK_SetModuleClock(EPWM1_MODULE, CLK_EPWMSEL_EPWM1SEL_PCLK2, 0);

    /* Enable GPIOB clock */
    CLK_EnableModuleClock(GPIOC_MODULE);
	
	SYS->GPC_MFP3 &= ~SYS_GPC_MFP3_PC12MFP_Msk;
	SYS->GPC_MFP3 |=  SYS_GPC_MFP3_PC12MFP_EPWM1_CH0;

    /* Configure EPWM channel 0 to 50 Hz, 5% duty (1 ms) */
    EPWM_ConfigOutputChannel(EPWM1, 0, SERVO_FREQ, 7);

    /* Enable output */
    EPWM_EnableOutput(EPWM1, EPWM_CH_0_MASK);

    /* Start EPWM */
    EPWM_Start(EPWM1, EPWM_CH_0_MASK);
}

void Servo_SetAngle(float angle)
{
    if (angle > 135.0f) angle = 135.0f;
    if (angle < -135.0f) angle = -135.0f;

    /* Convert angle to pulse width in microseconds */
    float pulse_us = 1500.0f + (angle * (2000.0f / 270.0f));

    uint32_t period = EPWM_GET_CNR(EPWM1, 0) + 1;

    /* 20 ms = 20000 us */
    uint32_t cmp = (uint32_t)((pulse_us * period) / 20000.0f + 0.5f);

    EPWM_SET_CMR(EPWM1, 0, cmp);
}