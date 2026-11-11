#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;
};
struct Params {
    int max_keypoints = 4000, iterations = 10000;
    double contrast = 0.03, ratio = 0.8, reprojection = 3;
    unsigned seed = 21;
};
struct Keypoint {
    double x, y;
    int level;
    double response;
};
using Descriptor = std::array<uint32_t, 8>;
using Matrix = std::array<double, 9>;
struct PreprocessedImage {
    int w, h;
    std::vector<std::vector<float>> gaussian;
};
struct Features {
    std::vector<Keypoint> points;
    std::vector<Descriptor> descriptors;
};
struct Match {
    int a, b, distance;
};
struct HomographyResult {
    Matrix h{};
    std::vector<int> inliers;
    double error = 0;
};
struct PipelineResult {
    Image panorama;
    HomographyResult model;
    std::array<double, 6> ms{};
    int points_a = 0, points_b = 0, matches = 0;
};




Image read_image(const std::string &path);
void write_image(const std::string &path, const Image &image);
/// Chuyển RGB sang ảnh xám và tạo năm mức Gaussian.
PreprocessedImage preprocess_v0(const Image &, const Params &);
/// Tìm cực trị DoG, loại điểm trên cạnh và giữ top-K theo đáp ứng.
std::vector<Keypoint> detect_v0(const PreprocessedImage &, const Params &);
/// Tính BRIEF; điểm và descriptor trả về có cùng thứ tự.
Features describe_v0(const PreprocessedImage &, const std::vector<Keypoint> &, const Params &);
/// Đếm số bit khác nhau, không dùng SIMD.
int hamming(const Descriptor &, const Descriptor &);
/// Ghép từng điểm của A với B, loại kết quả không đủ phân biệt.
std::vector<Match> match_v0(const Features &, const Features &, const Params &);
/// Ước lượng H từ B sang A; chỉ số inlier trỏ vào danh sách matches.
HomographyResult estimate_homography_v0(const Features &, const Features &,
                                        const std::vector<Match> &, const Params &);
/// Warp ngược từ canvas về B, rồi hòa trộn các pixel hợp lệ.
Image warp_blend_v0(const Image &, const Image &, const HomographyResult &, const Params &);
PipelineResult run_v0(const Image &, const Image &, const Params &);
/// Chiếu điểm từ B sang A; loại điểm có mẫu số gần 0.
inline std::array<double, 2> project(const Matrix &h, double x, double y) {
    double z = h[6] * x + h[7] * y + h[8];
    if (!std::isfinite(z) || std::abs(z) < 1e-10)
        throw std::runtime_error("projective horizon");
    return {(h[0] * x + h[1] * y + h[2]) / z, (h[3] * x + h[4] * y + h[5]) / z};
}
