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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgut/splatRaster.h>

#include <3dgut/sensors/cameraModels.h>

threedgut::CameraModelParameters
fromOpenCVPinholeCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                       threedgut::TSensorModel::ShutterType shutter_type,
                                       std::array<float, 2> principal_point,
                                       std::array<float, 2> focal_length,
                                       std::array<float, 6> radial_coeffs,
                                       std::array<float, 2> tangential_coeffs,
                                       std::array<float, 4> thin_prism_coeffs) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::OpenCVPinholeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec<6>), "[3dgut] typing size mismatch");
    static_assert(sizeof(tangential_coeffs) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(thin_prism_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvPinholeParams.nearFar          = tcnn::vec2{0.01f, 100.0f};
    params.ocvPinholeParams.principalPoint   = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvPinholeParams.focalLength      = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvPinholeParams.radialCoeffs     = *reinterpret_cast<const tcnn::vec<6>*>(radial_coeffs.data());
    params.ocvPinholeParams.tangentialCoeffs = *reinterpret_cast<const tcnn::vec2*>(tangential_coeffs.data());
    params.ocvPinholeParams.thinPrismCoeffs  = *reinterpret_cast<const tcnn::vec4*>(thin_prism_coeffs.data());
    return params;
}

threedgut::CameraModelParameters
fromOpenCVFisheyeCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                       threedgut::TSensorModel::ShutterType shutter_type,
                                       std::array<float, 2> principal_point,
                                       std::array<float, 2> focal_length,
                                       std::array<float, 4> radial_coeffs,
                                       float max_angle) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::OpenCVFisheyeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvFisheyeParams.principalPoint = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvFisheyeParams.focalLength    = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvFisheyeParams.radialCoeffs   = *reinterpret_cast<const tcnn::vec4*>(radial_coeffs.data());
    params.ocvFisheyeParams.maxAngle       = max_angle;
    return params;
}

threedgut::CameraModelParameters
fromBlenderFisheyeCameraModelParameters(std::array<uint64_t, 2> resolution,
                                        threedgut::TSensorModel::ShutterType shutter_type,
                                        std::array<float, 2> principal_point,
                                        std::array<float, 5> radial_coeffs,
                                        float sensor_width_mm,
                                        float sensor_height_mm,
                                        float fisheye_fov_deg) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::TSensorModel::ShutterType>(shutter_type);
    params.modelType   = threedgut::TSensorModel::BlenderFisheyeModel;

    // Replicate the aspect-ratio normalization from generate_lens_polynomial_rays_bl():
    //   if W >= H:  sensor_height_mm = sensor_width_mm * H / W
    //   else:       sensor_width_mm  = sensor_height_mm * W / H
    // After adjustment W/adj_sw == H/adj_sh (uniform pixels_per_mm).
    const float W = static_cast<float>(resolution[0]);
    const float H = static_cast<float>(resolution[1]);
    float adj_sw = sensor_width_mm;
    float adj_sh = sensor_height_mm;
    if (W >= H) {
        adj_sh = sensor_width_mm * H / W;
    } else {
        adj_sw = sensor_height_mm * W / H;
    }
    const float pixels_per_mm = W / adj_sw; // == H / adj_sh after adjustment

    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec<5>), "[3dgut] typing size mismatch");
    params.blenderFisheyeParams.principalPoint = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.blenderFisheyeParams.pixelsPerMm    = pixels_per_mm;
    params.blenderFisheyeParams.radialCoeffs   = *reinterpret_cast<const tcnn::vec<5>*>(radial_coeffs.data());

    // maxAngle = the true captured half-FOV = the Blender lens polynomial evaluated at the
    // sensor CORNER (largest r_mm any pixel maps to), NOT fisheye_fov_deg/2.
    //
    // The per-pixel rays (image_points_to_camera_rays_blender_mm) map the corner pixel to
    // r_mm = corner_px / pixels_per_mm and take theta = poly(r_mm); for this lens/sensor that
    // is ~61 deg, well below fisheye_fov_deg/2 = 90 deg, so the rays only ever cover 0..~61 deg.
    // If maxAngle is left at 90 deg, UT sigma points in the 61..90 deg "dead zone" (in front of
    // the camera but beyond any real pixel) are treated as valid and project out to ~440px,
    // inflating the projected covariance of peripheral Gaussians and producing tile-aligned
    // block artifacts near the image border. Matching maxAngle to the ray FOV makes the tiling
    // consistent with ray casting and excludes that dead zone (via the theta < maxAngle check
    // in projectPoint + the valid-sigma-point filtering in gutProjector).
    const float corner_px  = 0.5f * sqrtf(W * W + H * H);
    const float corner_mm  = corner_px / pixels_per_mm;
    const auto& kc         = params.blenderFisheyeParams.radialCoeffs;
    const float maxAnglePoly = kc[0] + corner_mm * (kc[1] + corner_mm * (kc[2] + corner_mm * (kc[3] + corner_mm * kc[4])));
    const float fovHalf      = fisheye_fov_deg * static_cast<float>(M_PI) / 180.f / 2.f;
    // Clamp into (0, fov/2] for safety against degenerate / non-monotonic coefficient sets.
    params.blenderFisheyeParams.maxAngle = fminf(fmaxf(maxAnglePoly, 1e-3f), fovHalf);
    return params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    pybind11::class_<SplatRaster>(m, "SplatRaster")
        .def(pybind11::init<const nlohmann::json&>())
        .def("trace", &SplatRaster::trace)
        .def("trace_bwd", &SplatRaster::traceBwd)
        .def("collect_times", &SplatRaster::collectTimes);

    py::enum_<threedgut::TSensorModel::ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", threedgut::TSensorModel::ShutterType::RollingTopToBottomShutter)
        .value("ROLLING_LEFT_TO_RIGHT", threedgut::TSensorModel::ShutterType::RollingLeftToRightShutter)
        .value("ROLLING_BOTTOM_TO_TOP", threedgut::TSensorModel::ShutterType::RollingBottomToTopShutter)
        .value("ROLLING_RIGHT_TO_LEFT", threedgut::TSensorModel::ShutterType::RollingRightToLeftShutter)
        .value("GLOBAL", threedgut::TSensorModel::ShutterType::GlobalShutter);

    py::class_<threedgut::CameraModelParameters>(m, "CameraModelParameters")
        .def(py::init<>());

    m.def("fromOpenCVPinholeCameraModelParameters", &fromOpenCVPinholeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("tangential_coeffs"),
          py::arg("thin_prism_coeffs"));

    m.def("fromOpenCVFisheyeCameraModelParameters", &fromOpenCVFisheyeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("max_angle"));

    m.def("fromBlenderFisheyeCameraModelParameters", &fromBlenderFisheyeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("radial_coeffs"),
          py::arg("sensor_width_mm"),
          py::arg("sensor_height_mm"),
          py::arg("fisheye_fov_deg"));
}
