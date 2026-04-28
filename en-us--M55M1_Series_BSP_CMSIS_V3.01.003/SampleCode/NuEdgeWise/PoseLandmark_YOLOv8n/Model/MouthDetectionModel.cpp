/**************************************************************************//**
 * @file     MouthDetectionModel.cpp
 * @version  V1.00
 * @brief    Mouth detection model (YOLOv8n ReLU6) source file.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "MouthDetectionModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::MouthDetectionModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::MouthDetectionModel::EnlistOperations()
{
    /* YOLOv8n: only Transpose + Ethos-U (rest delegated to NPU) */
    this->m_opResolver.AddTranspose();

#if defined(ARM_NPU)
    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
        info("Added %s support to op resolver\n",
             tflite::GetString_ETHOSU());
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
#endif /* ARM_NPU */
    return true;
}
