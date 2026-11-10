#include "../common.h"
#include <chrono>

PipelineResult run_v0(const Image &a, const Image &b, const Params &p) {
    PipelineResult result;
    using Clock = std::chrono::high_resolution_clock;
    auto begin = Clock::now();
    auto stamp = [&](int stage) {
        auto end = Clock::now();
        result.ms[stage] = std::chrono::duration<double, std::milli>(end - begin).count();
        begin = end;
    };
    auto prepared_a = preprocess_v0(a, p);
    auto prepared_b = preprocess_v0(b, p);
    stamp(0);
    auto keypoints_a = detect_v0(prepared_a, p);
    auto keypoints_b = detect_v0(prepared_b, p);
    stamp(1);
    auto features_a = describe_v0(prepared_a, keypoints_a, p);
    auto features_b = describe_v0(prepared_b, keypoints_b, p);
    stamp(2);
    auto matches = match_v0(features_a, features_b, p);
    stamp(3);
    result.model = estimate_homography_v0(features_a, features_b, matches, p);
    stamp(4);
    result.panorama = warp_blend_v0(a, b, result.model, p);
    stamp(5);
    result.points_a = int(features_a.points.size());
    result.points_b = int(features_b.points.size());
    result.matches = int(matches.size());
    return result;
}
