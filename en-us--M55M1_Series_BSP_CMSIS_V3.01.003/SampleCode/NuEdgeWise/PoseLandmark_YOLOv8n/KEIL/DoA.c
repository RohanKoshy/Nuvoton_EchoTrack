// doa.c  (fixed, completed, minimal-change)
// - Keeps your variables + flow
// - Fixes the BIG bug: your fft() wrapper used ONE static packed_tmp for BOTH channels.
//   That caused left and right FFT outputs to alias / get overwritten.
// - Also checks init status and hard-zeros temps.
// - Keeps your GCC-PHAT + peak search logic unchanged.
//
// Requirements:
//   - call doa_init() ONCE at startup
//   - fill left_buffer[] and right_buffer[] (FRAME_SIZE samples each) before calling doa_process_one_frame()
//   - include DoA.h with externs for left_buffer/right_buffer and prototypes

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "arm_math_types.h"
#include "dsp/transform_functions.h"
#include "DoA.h"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#define SAMPLE_RATE       48000
#define FRAME_SIZE        256
#define FRAME_SIZE_LOG2   8

#define SPEED_OF_SOUND    343.0f
#define MIC_SPACING       0.07f  //0.0508f    FOR 2 MIC ONLY
#define MAX_DELAY         (MIC_SPACING / SPEED_OF_SOUND)
#define MAX_SAMPLES_DELAY ((int)(MAX_DELAY * SAMPLE_RATE))
	
//Gates / thresholds for Energy
#define ENERGY_THRESH      (2.0e-5f)   // start here, tune
#define PEAK_THRESH        (0.25f)     // GCC-PHAT peak must be at least this
#define PEAK_RATIO_THRESH  (1.30f)     // peak must beat 2nd peak by this ratio

#define BAND_LOW_HZ   150.0f
#define BAND_HIGH_HZ  3500.0f
#define BIN_LOW  ((int)(BAND_LOW_HZ  * FRAME_SIZE / SAMPLE_RATE))
#define BIN_HIGH ((int)(BAND_HIGH_HZ * FRAME_SIZE / SAMPLE_RATE))
	



// ==========================
// Your existing arrays/names
// ==========================
float left_buffer[FRAME_SIZE];
float right_buffer[FRAME_SIZE];

float cross_corr[FRAME_SIZE];

float fft_left_real[FRAME_SIZE],  fft_left_imag[FRAME_SIZE];
float fft_right_real[FRAME_SIZE], fft_right_imag[FRAME_SIZE];

float hann_window[FRAME_SIZE];
uint32_t angle_calc_amount=0;

volatile uint8_t speech_detected=0;


// 4 Microphone additional add-on inits
float mic_spacing[6] = {
	0.035f, // (1,2)
	0.07f, // (1,3)
	0.105f, // (1,4)
	0.035f, // (2,3)
	0.07f, // (2,4)
	0.035f // (3,4)
};

float mic_spacing_tdoa[6]; // Results of each TDOA for the combination of mics are stored here

float mic_buffer1[FRAME_SIZE]; // Left-most
float mic_buffer2[FRAME_SIZE];
float mic_buffer3[FRAME_SIZE];
float mic_buffer4[FRAME_SIZE]; // Right-most



// ==========================
// CMSIS RFFT state + temp buffers
// ==========================
static arm_rfft_fast_instance_f32 g_rfft;

// Dedicated packed buffers (length N floats)
// IMPORTANT: we keep separate packed buffers for L and R to avoid aliasing.
static float32_t specL_packed[FRAME_SIZE];
static float32_t specR_packed[FRAME_SIZE];
static float32_t cpsd_packed[FRAME_SIZE];
static float32_t ifft_time_tmp[FRAME_SIZE];

// ==========================
// Hann window (manual)
// ==========================
void init_hann_window(float* window, int size)
{
    for (int n = 0; n < size; n++)
        window[n] = 0.5f * (1.0f - cosf((2.0f * PI * n) / (size - 1)));
}

// ==========================
// Unpack CMSIS packed RFFT output into full complex arrays (real/imag, length N)
// Convention:
//   packed[0] = Re{X[0]} (DC)
//   packed[1] = Re{X[N/2]} (Nyquist)
//   for k=1..N/2-1: packed[2k]=Re{X[k]}, packed[2k+1]=Im{X[k]}
// ==========================

static float frame_energy_mean_square(const float *x, int N, float *mean_out)
{
    float mean = 0.0f;
    for (int i = 0; i < N; i++) mean += x[i];
    mean /= (float)N;

    float e = 0.0f;
    for (int i = 0; i < N; i++) {
        float v = x[i] - mean;
        e += v * v;
    }
    e /= (float)N;

    if (mean_out) *mean_out = mean;
    return e;
}

static void unpack_rfft_packed_to_full(const float32_t *packed,
                                      float *outRe, float *outIm,
                                      int N)
{
    outRe[0]   = packed[0];
    outIm[0]   = 0.0f;

    outRe[N/2] = packed[1];
    outIm[N/2] = 0.0f;

    for (int k = 1; k < N/2; k++)
    {
        float32_t re = packed[2*k];
        float32_t im = packed[2*k + 1];

        outRe[k] = re;
        outIm[k] = im;

        // Hermitian symmetry
        outRe[N - k] = re;
        outIm[N - k] = -im;
    }
}

// Pack a Hermitian full spectrum (Re/Im arrays length N) back to CMSIS packed format length N
static void pack_full_to_rfft_packed(const float *inRe, const float *inIm,
                                    float32_t *packed, int N)
{
    packed[0] = inRe[0];     // DC
    packed[1] = inRe[N/2];   // Nyquist (real)

    for (int k = 1; k < N/2; k++)
    {
        packed[2*k]     = inRe[k];
        packed[2*k + 1] = inIm[k];
    }
}

// ==========================
// Fixed FFT wrappers
// Instead of a single static packed_tmp shared by both calls,
// this version writes into a packed buffer you provide.
// ==========================
static void fft_rfft_full(float *real, float *imag, int N, float32_t *packed_out)
{
    // RFFT input is real-only in real[]. imag[] is ignored (kept for minimal-change API).
    // arm_rfft_fast_f32 may overwrite its input buffer in some builds; we pass real[] directly.
    arm_rfft_fast_f32(&g_rfft, (float32_t*)real, packed_out, 0);

    // Unpack so rest of your code can keep using full complex arrays
    unpack_rfft_packed_to_full(packed_out, real, imag, N);
}

static void ifft_irfft_full(float *real, float *imag, int N)
{
    // Pack your Hermitian full spectrum into CMSIS packed format
    pack_full_to_rfft_packed(real, imag, cpsd_packed, N);

    // Inverse RFFT
    arm_rfft_fast_f32(&g_rfft, cpsd_packed, ifft_time_tmp, 1);

    // Copy back to your arrays (imag ~ 0)
    for (int i = 0; i < N; i++)
    {
        real[i] = ifft_time_tmp[i];
        imag[i] = 0.0f;
    }
}

// ==========================
// Init (call once)
// ==========================
void doa_init(void)
{
    arm_status st = arm_rfft_fast_init_f32(&g_rfft, FRAME_SIZE);
    if (st != ARM_MATH_SUCCESS)
    {
        // If this prints, your CMSIS component/config is wrong.
        printf("doa_init: arm_rfft_fast_init_f32 failed (st=%d)\n", (int)st);
    }

    init_hann_window(hann_window, FRAME_SIZE);

    // Hard-zero packed temps (nice for debugging / determinism)
    for (int i = 0; i < FRAME_SIZE; i++) {
        specL_packed[i] = 0.0f;
        specR_packed[i] = 0.0f;
        cpsd_packed[i]  = 0.0f;
        ifft_time_tmp[i]= 0.0f;
    }
}


// Clamp helper for 4 mic sin(x) calculation
static inline float clamp(float x, float min, float max) 
{
	if (x < min) return min;
	if (x > max) return max;
	return x;
}





// 4 Microphone Main DOA Algorithm
/*
float doa_process_4mic(float *mic0, float *mic1, float *mic2, float *mic3)
{
		// Declare new variables same name as global ones to save effort
		float *left_buffer;
		float *right_buffer;
	
		int silent_pairs = 0;
		int valid_mask[6] = {0};
	
		// Big for-loop to do TDOA on all 6 combinations of microphones
		for (unsigned int p = 0; p < (unsigned int)(sizeof(mic_spacing) / sizeof(mic_spacing[0])); p++) 
		{
			switch (p) 
			{
					case 0: left_buffer = mic0; right_buffer = mic1; break;
					case 1: left_buffer = mic0; right_buffer = mic2; break;
					case 2: left_buffer = mic0; right_buffer = mic3; break;
					case 3: left_buffer = mic1; right_buffer = mic2; break;
					case 4: left_buffer = mic1; right_buffer = mic3; break;
					case 5: left_buffer = mic2; right_buffer = mic3; break;
			}
			float meanL0, meanR0;
			float EL = frame_energy_mean_square(left_buffer,  FRAME_SIZE, &meanL0);
			float ER = frame_energy_mean_square(right_buffer, FRAME_SIZE, &meanR0);

			if (EL < ENERGY_THRESH && ER < ENERGY_THRESH) 
			{
            silent_pairs++;
						valid_mask[p] = 0;
            continue;
      }
			valid_mask[p] = 1;	
			
			// Step 1: mean removal & Hann  (REUSE meanL0/meanR0)
			float meanL = meanL0;
			float meanR = meanR0;

			for (int i = 0; i < FRAME_SIZE; i++) {
					float l = (left_buffer[i]  - meanL);
					float r = (right_buffer[i] - meanR);

					l *= hann_window[i];
					r *= hann_window[i];

					fft_left_real[i]  = l;
					fft_left_imag[i]  = 0.0f;

					fft_right_real[i] = r;
					fft_right_imag[i] = 0.0f;
			}

			// Step 2: FFT (CMSIS RFFT-backed)
			fft_rfft_full(fft_left_real,  fft_left_imag,  FRAME_SIZE, specL_packed);
			fft_rfft_full(fft_right_real, fft_right_imag, FRAME_SIZE, specR_packed);

			// Step 3: GCC-PHAT (unchanged)
			fft_left_real[0] = 0.0f;
			fft_left_imag[0] = 0.0f;

			fft_left_real[FRAME_SIZE/2] = 0.0f;
			fft_left_imag[FRAME_SIZE/2] = 0.0f;

			for (int k = 1; k < FRAME_SIZE/2; k++)
			{
					// Band Pass Filter
					if (k < BIN_LOW || k > BIN_HIGH)
					{
						fft_left_real[k] = 0.0f;
						fft_left_imag[k] = 0.0f;

						fft_left_real[FRAME_SIZE - k] = 0.0f;
						fft_left_imag[FRAME_SIZE - k] = 0.0f;
						
						continue;
					}
					
					// GCC-PHAT
					float real = fft_left_real[k]*fft_right_real[k]
										 + fft_left_imag[k]*fft_right_imag[k];

					float imag = fft_left_imag[k]*fft_right_real[k]
										 - fft_left_real[k]*fft_right_imag[k];

					float mag = sqrtf(real*real + imag*imag) + 1e-12f;

					float reN = real / mag;
					float imN = imag / mag;

					fft_left_real[k] = reN;
					fft_left_imag[k] = imN;

					// Hermitian symmetry
					fft_left_real[FRAME_SIZE - k] =  reN;
					fft_left_imag[FRAME_SIZE - k] = -imN;
			}

			// Step 4: IFFT ? cross-correlation
			ifft_irfft_full(fft_left_real, fft_left_imag, FRAME_SIZE);

			for (int i = 0; i < FRAME_SIZE; i++)
					cross_corr[i] = fft_left_real[i];

			// Step 5: Peak ? delay
			int best_n = 0;
			float max_val = -1e9f;

			for (int n = -MAX_SAMPLES_DELAY; n <= MAX_SAMPLES_DELAY; n++) {
					int idx = (n + FRAME_SIZE) % FRAME_SIZE;
					float val = cross_corr[idx];
					if (val > max_val) { max_val = val; best_n = n; }
			}

			float delta_t = (float)best_n / (float)SAMPLE_RATE;
			mic_spacing_tdoa[p] = delta_t;
		}
		
		if (silent_pairs == 6)
        return -180.0f;

    // Step 6: DoA angle (4 Microphones)
    float numerator = 0.0f;
		float denominator = 0.0f;

		// Loop through the TDOA results for the combinations of mics, do "Least Squares"
		for (int i = 0; i < (int)(sizeof(mic_spacing) / sizeof(mic_spacing[0])); i++) {
			if (!valid_mask[i]) continue;
			
			float b = SPEED_OF_SOUND * mic_spacing_tdoa[i];
			numerator += mic_spacing[i] * b;
			denominator += mic_spacing[i] * mic_spacing[i];
		}
		
		if (denominator == 0.0f) return 0.0f;
		
		float sin_theta = numerator / denominator;
		
		sin_theta = clamp(sin_theta, -1.0f, 1.0f);	
		
		// Calculate angle given the DOA equation and convert from radians into degrees
		float angle = asinf(sin_theta) * 180.0f / PI;
		
		if(1)
    {
        //printf("angle:  %.2f best_n=%d max_val=%f EL=%e ER=%e ratio =%f\n",
         //      angle, best_n, max_val, EL, ER, ratio);
			printf("angle:  %.2f\n", angle);
			return angle;
    }
}



*/
































// ==========================
// 2 Microphone Main Algorithm
// Provide left/right frames already loaded into left_buffer/right_buffer.
// ==========================


float doa_process_one_frame(void)
{
    float meanL0, meanR0;
    float EL = frame_energy_mean_square(left_buffer,  FRAME_SIZE, &meanL0);
    float ER = frame_energy_mean_square(right_buffer, FRAME_SIZE, &meanR0);

    if (EL < ENERGY_THRESH && ER < ENERGY_THRESH) {
				speech_detected=0;
        return -180.0f;   // your "silent" marker
    }

    // Step 1: mean removal & Hann  (REUSE meanL0/meanR0)
    float meanL = meanL0;
    float meanR = meanR0;

    for (int i = 0; i < FRAME_SIZE; i++) {
        float l = (left_buffer[i]  - meanL);
        float r = (right_buffer[i] - meanR);

        l *= hann_window[i];
        r *= hann_window[i];

        fft_left_real[i]  = l;
        fft_left_imag[i]  = 0.0f;

        fft_right_real[i] = r;
        fft_right_imag[i] = 0.0f;
    }

    // Step 2: FFT (CMSIS RFFT-backed)
    fft_rfft_full(fft_left_real,  fft_left_imag,  FRAME_SIZE, specL_packed);
    fft_rfft_full(fft_right_real, fft_right_imag, FRAME_SIZE, specR_packed);

    // Step 3: GCC-PHAT (unchanged)
    fft_left_real[0] = 0.0f;
    fft_left_imag[0] = 0.0f;

    fft_left_real[FRAME_SIZE/2] = 0.0f;
    fft_left_imag[FRAME_SIZE/2] = 0.0f;

    for (int k = 1; k < FRAME_SIZE/2; k++)
    {
				// Band Pass Filter
				if (k < BIN_LOW || k > BIN_HIGH)
				{
					fft_left_real[k] = 0.0f;
					fft_left_imag[k] = 0.0f;

					fft_left_real[FRAME_SIZE - k] = 0.0f;
					fft_left_imag[FRAME_SIZE - k] = 0.0f;
					
					continue;
				}
				
				// GCC-PHAT
        float real = fft_left_real[k]*fft_right_real[k]
                   + fft_left_imag[k]*fft_right_imag[k];

        float imag = fft_left_imag[k]*fft_right_real[k]
                   - fft_left_real[k]*fft_right_imag[k];

        float mag = sqrtf(real*real + imag*imag) + 1e-12f;

        float reN = real / mag;
        float imN = imag / mag;

        fft_left_real[k] = reN;
        fft_left_imag[k] = imN;

        // Hermitian symmetry
        fft_left_real[FRAME_SIZE - k] =  reN;
        fft_left_imag[FRAME_SIZE - k] = -imN;
    }

    // Step 4: IFFT to cross-correlation
    ifft_irfft_full(fft_left_real, fft_left_imag, FRAME_SIZE);

    for (int i = 0; i < FRAME_SIZE; i++)
        cross_corr[i] = fft_left_real[i];

    // Step 5: Peak to delay
    int best_n = 0;
    float max_val = -1e9f;

    for (int n = -MAX_SAMPLES_DELAY; n <= MAX_SAMPLES_DELAY; n++) {
        int idx = (n + FRAME_SIZE) % FRAME_SIZE;
        float val = cross_corr[idx];
        if (val > max_val) { 
					max_val = val;
					best_n = n;
				}
    }
		int i0 = (best_n - 1 + FRAME_SIZE) % FRAME_SIZE;
		int i1 = (best_n + FRAME_SIZE) % FRAME_SIZE;
		int i2 = (best_n + 1 + FRAME_SIZE) % FRAME_SIZE;
		float y0 = cross_corr[i0];
		float y1 = cross_corr[i1];
		float y2 = cross_corr[i2];
		float denom = y0 - 2.0f*y1 + y2;
		float frac=(fabsf(denom)>1e-12f)?0.5f*(y0 - y2) / denom : 0.0f;
		if(frac>0.5f)
		{
			frac =0.5f; // clamp to ±0.5 sample
		}
		if(frac < -0.5f)
		{
			frac = -0.5f;
		}
		float best_n_f = (float)best_n + frac;
		
		float second_max = -1e9f;
        for (int n = -MAX_SAMPLES_DELAY; n <= MAX_SAMPLES_DELAY; n++) {
            if (abs(n - best_n) <= MAX_SAMPLES_DELAY) continue;
            int   idx = (n + FRAME_SIZE) % FRAME_SIZE;
            float val = cross_corr[idx];
            if (val > second_max) second_max = val;
        }
        float peak_ratio = max_val / (fabsf(second_max) + 1e-12f);
        if (peak_ratio < PEAK_RATIO_THRESH)
				{
          speech_detected=0;  
					return -180.0f;
				}

    //float delta_t = (float)best_n / (float)SAMPLE_RATE;
		float delta_t = (float)best_n_f / (float)SAMPLE_RATE;
    // Step 6: DoA angle
    float ratio = (SPEED_OF_SOUND * delta_t) / MIC_SPACING;
    if (ratio > 1.0f)  ratio = 1.0f;
    if (ratio < -1.0f) ratio = -1.0f;

    float angle = asinf(ratio) * 180.0f / PI;

    angle_calc_amount++;
    //if (angle_calc_amount % 40 == 0)
		if(1)
    {
        //printf("angle:  %.2f best_n=%d max_val=%f EL=%e ER=%e ratio =%f\n",
         //      angle, best_n, max_val, EL, ER, ratio);
			printf("angle:  %.2f\n", angle);
			speech_detected=1;
				return angle;
    }
		speech_detected=0;
    return -180.0f;
}


