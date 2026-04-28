#ifndef MOTOR_H_
#define MOTOR_H_
#include "NuMicro.h"
#include <stdio.h>
#include <string.h>
#include "timer.h"
#include <math.h>
void Servo_PWM_Init();
void Servo_SetAngle(float angle);
#endif