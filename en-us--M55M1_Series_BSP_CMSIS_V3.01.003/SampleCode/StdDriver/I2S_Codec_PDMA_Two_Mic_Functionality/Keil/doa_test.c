// doa_test.c
// Standalone synthetic known-delay validation for your GCC-PHAT DoA pipeline.
//
// This file DEFINES its own test macros (Fs, frame size, mic spacing, etc.)
// so it will compile even if DoA.h doesn't export them.
//
// It assumes your doa.c provides these symbols:
//   - void  doa_init(void);
//   - float doa_process_one_frame(void);
//   - float left_buffer[DOA_FRAME_SIZE];
//   - float right_buffer[DOA_FRAME_SIZE];
//
// If your doa.c uses FRAME_SIZE=256, set DOA_FRAME_SIZE to 256 here to match.

#include <stdio.h>
#include <stdint.h>

// --------------------------
// TEST CONFIG (edit if needed)
// --------------------------
#ifndef DOA_SAMPLE_RATE
#define DOA_SAMPLE_RATE      48000
#endif

#ifndef DOA_FRAME_SIZE
#define DOA_FRAME_SIZE       256
#endif

#ifndef DOA_MIC_SPACING_M
#define DOA_MIC_SPACING_M    0.0508f
#endif

#ifndef DOA_SPEED_OF_SOUND
#define DOA_SPEED_OF_SOUND   343.0f
#endif

#ifndef DOA_MAX_SAMPLES_DELAY
#define DOA_MAX_SAMPLES_DELAY  ((int)((DOA_MIC_SPACING_M / DOA_SPEED_OF_SOUND) * DOA_SAMPLE_RATE))
#endif

// --------------------------
// Pull in your DoA API
// --------------------------
#include "DoA.h"

// If your DoA.h does not declare these, uncomment:
extern void  doa_init(void);
extern float doa_process_one_frame(void);
extern float left_buffer[DOA_FRAME_SIZE];
extern float right_buffer[DOA_FRAME_SIZE];

// ------------------------------------------------------------
// Deterministic PRNG (xorshift32)
// ------------------------------------------------------------
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ------------------------------------------------------------
// Fill deterministic broadband-ish signal into x[] (float ~[-1,1])
// ------------------------------------------------------------
static void fill_prn_signal(float *x, int N, uint32_t seed)
{
    uint32_t st = (seed == 0u) ? 1u : seed;

    for (int i = 0; i < N; i++)
    {
        uint32_t r = xorshift32(&st) >> 9;          // 23 bits
        float v = ((float)r / 4194304.0f) - 1.0f;   // 2^22 => [-1,1)
        x[i] = v;
    }

    // Remove DC
    float mean = 0.0f;
    for (int i = 0; i < N; i++) mean += x[i];
    mean /= (float)N;
    for (int i = 0; i < N; i++) x[i] -= mean;
}

// ------------------------------------------------------------
// Circular shift: y[n] = x[n - D] (wrap)
// ------------------------------------------------------------
static void circular_shift(const float *x, float *y, int N, int D)
{
    int d = D % N;
    if (d < 0) d += N;

    for (int n = 0; n < N; n++)
    {
        int src = n - d;
        if (src < 0) src += N;
        y[n] = x[src];
    }
}

// ------------------------------------------------------------
// Optional noise (amp=0 disables)
// ------------------------------------------------------------
static void add_white_noise(float *x, int N, float amp, uint32_t seed)
{
    if (amp <= 0.0f) return;

    uint32_t st = (seed == 0u) ? 123u : seed;

    for (int i = 0; i < N; i++)
    {
        uint32_t r = xorshift32(&st) >> 9;
        float v = ((float)r / 4194304.0f) - 1.0f;  // [-1,1)
        x[i] += amp * v;
    }
}

// ------------------------------------------------------------
// Public test entrypoint
// ------------------------------------------------------------
void doa_run_synthetic_delay_test(void)
{
    // Make sure DoA init is called (window + RFFT plan)
    doa_init();

    printf("\n=== DoA synthetic known-delay test ===\n");
    printf("Fs=%d, N=%d, spacing=%.4f m => MAX_SAMPLES_DELAY=%d\n",
           (int)DOA_SAMPLE_RATE,
           (int)DOA_FRAME_SIZE,
           (double)DOA_MIC_SPACING_M,
           (int)DOA_MAX_SAMPLES_DELAY);

    // Choose delays INSIDE ±DOA_MAX_SAMPLES_DELAY
    // These defaults work well for ~5 cm spacing at 48 kHz (max ~7 samples).
    const int delays[] = { -6, -4, -2, -1, 0, 1, 2, 4, 6 };
    const int num_delays = (int)(sizeof(delays) / sizeof(delays[0]));

    for (int di = 0; di < num_delays; di++)
    {
        int D = delays[di];
        if (D < -DOA_MAX_SAMPLES_DELAY || D > DOA_MAX_SAMPLES_DELAY)
        {
            printf("\nSkipping D=%d (outside ±%d)\n", D, (int)DOA_MAX_SAMPLES_DELAY);
            continue;
        }

        // Build deterministic test signal
        fill_prn_signal(left_buffer, DOA_FRAME_SIZE, 0xC0FFEEu + (uint32_t)(di * 17u));

        // Right = shifted Left by D
        circular_shift(left_buffer, right_buffer, DOA_FRAME_SIZE, D);

        // Optional: add tiny noise (should still be stable if pipeline is correct)
        // add_white_noise(left_buffer,  DOA_FRAME_SIZE, 0.001f, 0x111u);
        // add_white_noise(right_buffer, DOA_FRAME_SIZE, 0.001f, 0x222u);

        printf("\nInjected integer lag D = %d samples\n", D);

        for (int rep = 0; rep < 5; rep++)
        {
            float angle = doa_process_one_frame();
            // Your doa_process_one_frame should print best_n/max_val itself
            printf("rep=%d  angle=%.2f\n", rep, angle);
        }
    }

    printf("\n=== End synthetic test ===\n\n");
}
