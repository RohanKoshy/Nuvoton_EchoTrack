/**************************************************************************//**
 * @file     MouthYOLOv8PostProcessing.hpp
 * @version  V1.00
 * @brief    YOLOv8n mouth detection post-processing (DFL, reg_max=16).
 *           Spatial H&W must match the TFLite input (e.g. 128 or 192); pass from model shape.
 *           Output tensor shapes follow anchor counts: P3 (H/8)^2, P4 (H/16)^2, P5 (H/32)^2.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef MOUTH_YOLOV8_POST_PROCESSING_HPP
#define MOUTH_YOLOV8_POST_PROCESSING_HPP

#include "FaceDetectionResult.hpp"
#include "MouthDetectionModel.hpp"

#include <forward_list>

#define MOUTH_NUM_CLASSES       2
#define MOUTH_REG_MAX           16
#define MOUTH_STRIDE_8          8
#define MOUTH_STRIDE_16         16
#define MOUTH_STRIDE_32         32

/* Output tensor indices: 0=cls P4, 1=box P4, 2=cls P5, 3=cls P3, 4=box P3, 5=box P5 */
#define MOUTH_BOX_P3_INDEX      4   /* [1, (H/8)^2, 64] stride 8 */
#define MOUTH_BOX_P4_INDEX      1   /* [1, (H/16)^2, 64] stride 16 */
#define MOUTH_BOX_P5_INDEX      5   /* [1, (H/32)^2, 64] stride 32 */
#define MOUTH_CLS_P3_INDEX      3   /* [1, (H/8)^2, 2] */
#define MOUTH_CLS_P4_INDEX      0   /* [1, (H/16)^2, 2] */
#define MOUTH_CLS_P5_INDEX      2   /* [1, (H/32)^2, 2] */

namespace arm
{
namespace app
{
namespace mouth_detection
{

struct Box {
    float x;
    float y;
    float w;
    float h;
};

struct MouthDetection {
    Box bbox;
    int strideIndex;
    int anchorIndex;
    int cls;
    std::vector<float> prob;
};

struct AnchorBox {
    float w;
    float h;
};

/**
 * @brief Post-processing for YOLOv8n mouth detection (2 classes, DFL reg_max=16).
 */
class MouthYOLOv8PostProcessing
{
public:
    explicit MouthYOLOv8PostProcessing(arm::app::MouthDetectionModel *model,
                                      float threshold,
                                      float iouThreshold,
                                      int inputSpatialSize);

    void RunPostProcessing(uint32_t imgNetRows,
                           uint32_t imgNetCols,
                           uint32_t imgSrcRows,
                           uint32_t imgSrcCols,
                           std::vector<face_detection::DetectionResult> &resultsOut);

private:
    arm::app::MouthDetectionModel *m_model;
    float m_threshold;
    float m_iouThreshold;
    int m_inputSpatialSize;        /* H and W of square mouth input (from TFLite shape) */
    int m_stride8_total_anchors;   /* (size/8)^2 */
    int m_stride16_total_anchors;  /* (size/16)^2 */
    int m_stride32_total_anchors;  /* (size/32)^2 */

    std::vector<AnchorBox> m_stride8_anchors;
    std::vector<AnchorBox> m_stride16_anchors;
    std::vector<AnchorBox> m_stride32_anchors;

    void GetNetworkBoxes(std::forward_list<MouthDetection> &detections);
};

} /* namespace mouth_detection */
} /* namespace app */
} /* namespace arm */

#endif /* MOUTH_YOLOV8_POST_PROCESSING_HPP */
