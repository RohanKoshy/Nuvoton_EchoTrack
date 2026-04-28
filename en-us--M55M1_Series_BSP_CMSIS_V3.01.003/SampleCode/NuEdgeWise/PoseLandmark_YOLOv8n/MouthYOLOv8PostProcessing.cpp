/**************************************************************************//**
 * @file     MouthYOLOv8PostProcessing.cpp
 * @version  V1.00
 * @brief    YOLOv8n mouth detection post-processing (DFL, reg_max=16).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "MouthYOLOv8PostProcessing.hpp"
#include "PlatformMath.hpp"
#include "TensorFlowLiteMicro.hpp"
#include "log_macros.h"

#include <cmath>

using namespace arm::app::mouth_detection;

/* Debug: set to 1 to log raw max confidence from tensor (before threshold) */
#define MOUTH_DEBUG_MAX_CONF 0

static void AnchorMatrixConstruct(
    std::vector<AnchorBox> &vAnchorBoxs,
    int inputSpatialSize,
    int i32Stride,
    int i32StrideTotalAnchors
)
{
    float fStartAnchorValue = 0.5f;
    int iMaxAnchorValue = inputSpatialSize / i32Stride;
    float fAnchor0StepValue = 0.f;
    float fAnchor1StepValue = -1.f;

    for (int i = 0; i < i32StrideTotalAnchors; i++)
    {
        AnchorBox sAnchorBox;

        if ((i % iMaxAnchorValue) == 0)
        {
            fStartAnchorValue = 0.5f;
            fAnchor0StepValue = 0.f;
            fAnchor1StepValue++;
        }

        sAnchorBox.w = fStartAnchorValue + (fAnchor0StepValue++);
        sAnchorBox.h = fStartAnchorValue + fAnchor1StepValue;

        vAnchorBoxs.push_back(sAnchorBox);
    }
}

static void CalBoxXYWH(
    TfLiteTensor *psBoxOutputTensor,
    std::vector<AnchorBox> &vAnchorBoxs,
    int i32AnchorIndex,
    int i32Stride,
    int i32StrideTotalAnchors,
    MouthDetection &sDetection
)
{
    float scaleBox;
    int zeroPointBox;
    int anchors;
    int boxDataSize;
    float XYWHResult[4];

    int8_t *tensorOutputBox = psBoxOutputTensor->data.int8;
    scaleBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->scale->data[0];
    zeroPointBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->zero_point->data[0];

    anchors = psBoxOutputTensor->dims->data[1];
    boxDataSize = psBoxOutputTensor->dims->data[2];

    if (anchors != i32StrideTotalAnchors || boxDataSize != 64)
    {
        return;
    }

    tensorOutputBox = tensorOutputBox + (i32AnchorIndex * boxDataSize);

    for (int k = 0; k < 4; k++)
    {
        std::vector<float> XYWHSoftmaxTemp(16);
        float XYWHSoftmaxResult = 0.f;

        for (int i = 0; i < 16; i++)
        {
            XYWHSoftmaxTemp[i] = scaleBox * (static_cast<float>(tensorOutputBox[k * 16 + i]) - zeroPointBox);
        }

        arm::app::math::MathUtils::SoftmaxF32(XYWHSoftmaxTemp);
        for (int i = 0; i < 16; i++)
        {
            XYWHSoftmaxResult = XYWHSoftmaxResult + XYWHSoftmaxTemp[i] * i;
        }
        XYWHResult[k] = XYWHSoftmaxResult;
    }

    /* dist2bbox */
    float x1 = vAnchorBoxs[i32AnchorIndex].w - XYWHResult[0];
    float y1 = vAnchorBoxs[i32AnchorIndex].h - XYWHResult[1];
    float x2 = vAnchorBoxs[i32AnchorIndex].w + XYWHResult[2];
    float y2 = vAnchorBoxs[i32AnchorIndex].h + XYWHResult[3];

    float cx = (x1 + x2) / 2.f;
    float cy = (y1 + y2) / 2.f;
    float w = x2 - x1;
    float h = y2 - y1;

    XYWHResult[0] = cx * i32Stride;
    XYWHResult[1] = cy * i32Stride;
    XYWHResult[2] = w * i32Stride;
    XYWHResult[3] = h * i32Stride;

    sDetection.bbox.x = XYWHResult[0] - (0.5f * XYWHResult[2]);
    sDetection.bbox.y = XYWHResult[1] - (0.5f * XYWHResult[3]);
    sDetection.bbox.w = XYWHResult[2];
    sDetection.bbox.h = XYWHResult[3];
}

static float Calculate1DOverlap(float x1Center, float width1, float x2Center, float width2)
{
    float left_1 = x1Center - width1 / 2;
    float left_2 = x2Center - width2 / 2;
    float leftest = left_1 > left_2 ? left_1 : left_2;

    float right_1 = x1Center + width1 / 2;
    float right_2 = x2Center + width2 / 2;
    float rightest = right_1 < right_2 ? right_1 : right_2;

    return rightest - leftest;
}

static float CalculateBoxIntersect(Box &box1, Box &box2)
{
    float width = Calculate1DOverlap(box1.x, box1.w, box2.x, box2.w);
    if (width < 0) return 0;
    float height = Calculate1DOverlap(box1.y, box1.h, box2.y, box2.h);
    if (height < 0) return 0;
    return width * height;
}

static float CalculateBoxUnion(Box &box1, Box &box2)
{
    float boxes_intersection = CalculateBoxIntersect(box1, box2);
    return box1.w * box1.h + box2.w * box2.h - boxes_intersection;
}

static float CalculateBoxIOU(Box &box1, Box &box2)
{
    float boxes_intersection = CalculateBoxIntersect(box1, box2);
    if (boxes_intersection == 0) return 0;
    float boxes_union = CalculateBoxUnion(box1, box2);
    if (boxes_union == 0) return 0;
    return boxes_intersection / boxes_union;
}

static void CalculateNMS(std::forward_list<MouthDetection> &detections, int classes, float iouThreshold)
{
    for (int idxClass = 0; idxClass < classes; ++idxClass)
    {
        auto CompareProbs = [idxClass](MouthDetection &a, MouthDetection &b) {
            return a.prob[idxClass] > b.prob[idxClass];
        };
        detections.sort(CompareProbs);

        for (auto it = detections.begin(); it != detections.end(); ++it)
        {
            if (it->prob[idxClass] == 0) continue;
            for (auto itc = std::next(it, 1); itc != detections.end(); ++itc)
            {
                if (itc->prob[idxClass] == 0) continue;
                if (CalculateBoxIOU(it->bbox, itc->bbox) > iouThreshold)
                {
                    itc->prob[idxClass] = 0;
                }
            }
        }
    }
}

static void CalDetectionBox(
    TfLiteTensor *psConfidenceOutputTensor,
    TfLiteTensor *psBoxOutputTensor,
    std::vector<AnchorBox> &vAnchorBoxs,
    int i32Stride,
    int i32StrideTotalAnchors,
    float fThreshold,
    std::forward_list<MouthDetection> &sDetections
)
{
    float scaleConf;
    int zeroPointConf;
    int8_t *tensorOutputConf = psConfidenceOutputTensor->data.int8;

    scaleConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->scale->data[0];
    zeroPointConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->zero_point->data[0];

    for (int i = 0; i < i32StrideTotalAnchors; i++)
    {
        float maxScore = 0.f;
        int cls = 0;
        int maxConf = -128;

        for (int j = 0; j < MOUTH_NUM_CLASSES; j++)
        {
            int confTensorData = tensorOutputConf[(i * MOUTH_NUM_CLASSES) + j];
            if (confTensorData > maxConf)
            {
                maxConf = confTensorData;
                cls = j;
            }
        }

        maxScore = arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(maxConf - zeroPointConf)));

        if (maxScore >= fThreshold)
        {
            MouthDetection det;
            det.strideIndex = i32Stride;
            det.anchorIndex = i;
            det.cls = cls;

            for (int j = 0; j < MOUTH_NUM_CLASSES; j++)
            {
                float score = arm::app::math::MathUtils::SigmoidF32(
                    scaleConf * (static_cast<float>(tensorOutputConf[(i * MOUTH_NUM_CLASSES) + j] - zeroPointConf)));
                det.prob.push_back(score);
            }

            CalBoxXYWH(psBoxOutputTensor, vAnchorBoxs, i, i32Stride, i32StrideTotalAnchors, det);
            sDetections.push_front(det);
        }
    }
}

namespace arm
{
namespace app
{
namespace mouth_detection
{

MouthYOLOv8PostProcessing::MouthYOLOv8PostProcessing(
    arm::app::MouthDetectionModel *model,
    float threshold,
    float iouThreshold,
    int inputSpatialSize)
    : m_model(model),
      m_threshold(threshold),
      m_iouThreshold(iouThreshold),
      m_inputSpatialSize(inputSpatialSize)
{
    m_stride8_total_anchors = static_cast<int>(std::pow(m_inputSpatialSize / MOUTH_STRIDE_8, 2));
    m_stride16_total_anchors = static_cast<int>(std::pow(m_inputSpatialSize / MOUTH_STRIDE_16, 2));
    m_stride32_total_anchors = static_cast<int>(std::pow(m_inputSpatialSize / MOUTH_STRIDE_32, 2));

    m_stride8_anchors.clear();
    m_stride16_anchors.clear();
    m_stride32_anchors.clear();

    AnchorMatrixConstruct(m_stride8_anchors, m_inputSpatialSize, MOUTH_STRIDE_8, m_stride8_total_anchors);
    AnchorMatrixConstruct(m_stride16_anchors, m_inputSpatialSize, MOUTH_STRIDE_16, m_stride16_total_anchors);
    AnchorMatrixConstruct(m_stride32_anchors, m_inputSpatialSize, MOUTH_STRIDE_32, m_stride32_total_anchors);

    if (m_model)
    {
        auto ok = [&](int clsIdx, int boxIdx, int expectAnchors) -> bool {
            TfLiteTensor *c = m_model->GetOutputTensor(clsIdx);
            TfLiteTensor *b = m_model->GetOutputTensor(boxIdx);
            return c && b && c->dims && b->dims && c->dims->size >= 3 && b->dims->size >= 3
                && c->dims->data[1] == expectAnchors && b->dims->data[1] == expectAnchors
                && c->dims->data[2] == MOUTH_NUM_CLASSES && b->dims->data[2] == 64;
        };
        if (!ok(MOUTH_CLS_P3_INDEX, MOUTH_BOX_P3_INDEX, m_stride8_total_anchors)
            || !ok(MOUTH_CLS_P4_INDEX, MOUTH_BOX_P4_INDEX, m_stride16_total_anchors)
            || !ok(MOUTH_CLS_P5_INDEX, MOUTH_BOX_P5_INDEX, m_stride32_total_anchors))
        {
            printf_err("Mouth PP: output shapes vs inputSpatialSize=%d (P3=%d P4=%d P5=%d) mismatch\n",
                       m_inputSpatialSize, m_stride8_total_anchors, m_stride16_total_anchors,
                       m_stride32_total_anchors);
        }
    }
}

void MouthYOLOv8PostProcessing::RunPostProcessing(
    uint32_t imgNetRows,
    uint32_t imgNetCols,
    uint32_t imgSrcRows,
    uint32_t imgSrcCols,
    std::vector<face_detection::DetectionResult> &resultsOut)
{
    (void)imgNetRows;
    (void)imgNetCols;
    float fXScale = static_cast<float>(imgSrcCols) / static_cast<float>(m_inputSpatialSize);
    float fYScale = static_cast<float>(imgSrcRows) / static_cast<float>(m_inputSpatialSize);

    /* Debug: scan raw cls tensors for max sigmoid value (before threshold filter) */
#if MOUTH_DEBUG_MAX_CONF
    {
        static int dbgCnt = 0;
        if (++dbgCnt >= 30) {
            dbgCnt = 0;
            float maxSigmoid = 0.f;
            const int clsIndices[] = { MOUTH_CLS_P4_INDEX, MOUTH_CLS_P5_INDEX, MOUTH_CLS_P3_INDEX };
            for (int t = 0; t < 3; t++) {
                TfLiteTensor *cls = m_model->GetOutputTensor(clsIndices[t]);
                float scale = ((TfLiteAffineQuantization *)(cls->quantization.params))->scale->data[0];
                int zp = ((TfLiteAffineQuantization *)(cls->quantization.params))->zero_point->data[0];
                int8_t *d = cls->data.int8;
                int numAnchors = cls->dims->data[1];
                int numCls = cls->dims->data[2];
                for (int i = 0; i < numAnchors * numCls; i++) {
                    float s = arm::app::math::MathUtils::SigmoidF32(scale * (static_cast<float>(d[i]) - zp));
                    if (s > maxSigmoid) maxSigmoid = s;
                }
            }
            info("mouth: rawMaxSigmoid=%.3f (th=%.2f)\n", maxSigmoid, m_threshold);
        }
    }
#endif

    std::forward_list<MouthDetection> sDetections;
    GetNetworkBoxes(sDetections);
    CalculateNMS(sDetections, MOUTH_NUM_CLASSES, m_iouThreshold);

    resultsOut.clear();

    for (auto box = sDetections.begin(); box != sDetections.end(); ++box)
    {
        float score = box->prob[box->cls];

        if (score > 0)
        {
            float x = box->bbox.x * fXScale;
            float y = box->bbox.y * fYScale;
            float w = box->bbox.w * fXScale;
            float h = box->bbox.h * fYScale;

            x = std::min(std::max(x, 0.f), static_cast<float>(imgSrcCols - 1));
            y = std::min(std::max(y, 0.f), static_cast<float>(imgSrcRows - 1));
            w = std::min(std::max(w, 0.f), static_cast<float>(imgSrcCols - 1));
            h = std::min(std::max(h, 0.f), static_cast<float>(imgSrcRows - 1));

            face_detection::DetectionResult r(static_cast<double>(score),
                                             static_cast<int>(x),
                                             static_cast<int>(y),
                                             static_cast<int>(w),
                                             static_cast<int>(h),
                                             box->cls);
            resultsOut.push_back(r);
        }
    }
}

void MouthYOLOv8PostProcessing::GetNetworkBoxes(std::forward_list<MouthDetection> &detections)
{
    TfLiteTensor *psConfidenceTensor;
    TfLiteTensor *psBoxTensor;

    psBoxTensor = m_model->GetOutputTensor(MOUTH_BOX_P3_INDEX);
    psConfidenceTensor = m_model->GetOutputTensor(MOUTH_CLS_P3_INDEX);
    CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride8_anchors,
                   MOUTH_STRIDE_8, m_stride8_total_anchors, m_threshold, detections);

    psBoxTensor = m_model->GetOutputTensor(MOUTH_BOX_P4_INDEX);
    psConfidenceTensor = m_model->GetOutputTensor(MOUTH_CLS_P4_INDEX);
    CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride16_anchors,
                   MOUTH_STRIDE_16, m_stride16_total_anchors, m_threshold, detections);

    psBoxTensor = m_model->GetOutputTensor(MOUTH_BOX_P5_INDEX);
    psConfidenceTensor = m_model->GetOutputTensor(MOUTH_CLS_P5_INDEX);
    CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride32_anchors,
                   MOUTH_STRIDE_32, m_stride32_total_anchors, m_threshold, detections);
}

} /* namespace mouth_detection */
} /* namespace app */
} /* namespace arm */
