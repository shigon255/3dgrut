// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <tiny-cuda-nn/vec.h>

namespace threedgut {

struct OpenCVPinholeProjectionParameters {
    tcnn::vec2 nearFar;
    tcnn::vec2 principalPoint;
    tcnn::vec2 focalLength;
    tcnn::vec<6> radialCoeffs;
    tcnn::vec2 tangentialCoeffs;
    tcnn::vec4 thinPrismCoeffs;
};

struct OpenCVFisheyeProjectionParameters {
    tcnn::vec2 principalPoint;
    tcnn::vec2 focalLength;
    tcnn::vec4 radialCoeffs;
    float maxAngle;
};

// Blender lens polynomial: theta_rad = k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4
// where r is in mm (sensor space) and theta is the angle from optical axis in radians.
// pixelsPerMm: uniform pixel-per-mm scale after aspect-ratio normalization (W/sensor_width_mm).
struct BlenderFisheyeProjectionParameters {
    tcnn::vec2   principalPoint; // [cx, cy] in pixels (typically [W/2, H/2])
    float        pixelsPerMm;    // uniform scale: pixels / mm (same for both axes after aspect adjustment)
    tcnn::vec<5> radialCoeffs;   // k0..k4; polynomial outputs theta in radians
    float        maxAngle;       // half-FoV in radians (= fisheye_fov_deg * pi/360)
};

struct CameraModelParameters {
    enum ShutterType {
        RollingTopToBottomShutter,
        RollingLeftToRightShutter,
        RollingBottomToTopShutter,
        RollingRightToLeftShutter,
        GlobalShutter
    } shutterType = GlobalShutter;

    enum ModelType {
        OpenCVPinholeModel,
        OpenCVFisheyeModel,
        BlenderFisheyeModel,
        EmptyModel,
        Unsupported
    } modelType = EmptyModel;

    union {
        OpenCVPinholeProjectionParameters  ocvPinholeParams;
        OpenCVFisheyeProjectionParameters  ocvFisheyeParams;
        BlenderFisheyeProjectionParameters blenderFisheyeParams;
    };
};

} // namespace threedgut
