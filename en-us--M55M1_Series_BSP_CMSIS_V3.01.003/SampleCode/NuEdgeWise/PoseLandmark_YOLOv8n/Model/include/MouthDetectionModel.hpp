/**************************************************************************//**
 * @file     MouthDetectionModel.hpp
 * @version  V1.00
 * @brief    Mouth detection model (YOLOv8n ReLU6) header file.
 *           Input: square RGB (e.g. 128x128), int8 = uint8 - 128
 *           Output: 6 tensors (box P3/P4/P5, cls P3/P4/P5), DFL reg_max=16
 *           Classes: 0 = mouth closed, 1 = mouth open
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef MOUTH_DETECTION_MODEL_HPP
#define MOUTH_DETECTION_MODEL_HPP

#include "Model.hpp"

namespace arm
{
namespace app
{

class MouthDetectionModel : public Model
{

public:
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    const tflite::MicroOpResolver &GetOpResolver() override;
    bool EnlistOperations() override;

private:
    static constexpr int ms_maxOpCnt = 2;  /* YOLOv8n: Transpose + Ethos-U */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* MOUTH_DETECTION_MODEL_HPP */
