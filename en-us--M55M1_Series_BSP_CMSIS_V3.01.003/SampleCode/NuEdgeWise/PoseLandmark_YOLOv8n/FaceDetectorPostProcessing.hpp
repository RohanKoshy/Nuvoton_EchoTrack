/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FACE_DETECTOR_POST_PROCESSING_HPP
#define FACE_DETECTOR_POST_PROCESSING_HPP
#include "TensorFlowLiteMicro.hpp"
#include "ImageUtils.hpp"
#include "FaceDetectionResult.hpp"
#include "BaseProcessing.hpp"
#include <forward_list>
namespace arm {
namespace app {
namespace face_detection {
    struct PostProcessParams {
        int inputImgRows{};
        int inputImgCols{};
        int originalImageRows{};
        int originalImageCols{};
        const float* anchor1;
        const float* anchor2;
        float threshold = 0.9f;
        float nms = 0.45f;
        int numClasses = 1;
        int topN = 0;
    };
    struct Branch {
        int resolution;
        int numBox;
        const float* anchor;
        int8_t* modelOutput;
        float scale;
        int zeroPoint;
        size_t size;
    };
    struct Network {
        int inputWidth;
        int inputHeight;
        int numClasses;
        std::vector<Branch> branches;
        int topN;
    };
} /* namespace face_detection */
    /**
     * @brief   Post-processing class for YOLO-Fastest style detection (face/mouth).
     */
    class FaceDetectorPostProcess : public BasePostProcess {
    public:
        explicit FaceDetectorPostProcess(TfLiteTensor* outputTensor0,
                                     TfLiteTensor* outputTensor1,
                                     std::vector<face_detection::DetectionResult>& results,
                                     const face_detection::PostProcessParams& postProcessParams);

        bool DoPostProcess() override;

        bool RunPostProcess(std::vector<face_detection::DetectionResult>& results);
    private:
        TfLiteTensor* m_outputTensor0;
        TfLiteTensor* m_outputTensor1;
        std::vector<face_detection::DetectionResult>& m_results;
        const face_detection::PostProcessParams& m_postProcessParams;
        face_detection::Network m_net;

        void InsertTopNDetections(std::forward_list<image::Detection>& detections, image::Detection& det);
        void GetNetworkBoxes(face_detection::Network& net,
                             int imageWidth,
                             int imageHeight,
                             float threshold,
                             std::forward_list<image::Detection>& detections);
    };
} /* namespace app */
} /* namespace arm */
#endif /* FACE_DETECTOR_POST_PROCESSING_HPP */
