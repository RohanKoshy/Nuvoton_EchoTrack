/**************************************************************************//**
 * @file     FaceDetectionModel.hpp
 * @version  V1.00
 * @brief    Face detection model header file (YOLO-Fastest, embedded)
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef FACE_DETECTION_MODEL_HPP
#define FACE_DETECTION_MODEL_HPP

#include "Model.hpp"
#include <cstddef>

extern const float anchor1[];
extern const float anchor2[];

namespace arm
{
namespace app
{
namespace face_detection
{
    const uint8_t *GetModelPointer();
    size_t GetModelLen();
} /* namespace face_detection */
} /* namespace app */
} /* namespace arm */

namespace arm
{
namespace app
{

class FaceDetectionModel : public Model
{

public:
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    const tflite::MicroOpResolver &GetOpResolver() override;
    bool EnlistOperations() override;

private:
    static constexpr int ms_maxOpCnt = 1;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* FACE_DETECTION_MODEL_HPP */
