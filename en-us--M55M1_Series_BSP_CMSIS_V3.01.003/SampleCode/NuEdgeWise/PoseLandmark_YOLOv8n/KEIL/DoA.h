#pragma once
#include <stdint.h>

void  doa_init(void);

// feed one full 256-sample frame (per channel) already converted to float in doa.c
float doa_process_one_frame(void);
void doa_run_synthetic_delay_test(void);

// these are filled by your main when a frame is ready
extern float left_buffer[256];
extern float right_buffer[256];


