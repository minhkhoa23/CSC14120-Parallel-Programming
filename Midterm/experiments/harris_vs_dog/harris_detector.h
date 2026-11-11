#pragma once
#include "../../src/common.h"

// Experimental detector only: the submitted V0 sources are not modified.
struct HarrisParams {
    double k = 0.04;
    double integration_sigma = 1.5;
    double quality_level = 0.01;
};

std::vector<Keypoint> detect_harris(const PreprocessedImage &, const Params &,
                                   const HarrisParams & = {});
