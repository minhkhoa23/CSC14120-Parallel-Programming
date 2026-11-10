// Tham khảo inverse warping và feather blending: ParaPano/stitcher.cpp (MIT).
#include "../common.h"
#include <limits>
namespace {
Matrix inverse(const Matrix &h) {
    Matrix a{h[4] * h[8] - h[5] * h[7], h[2] * h[7] - h[1] * h[8], h[1] * h[5] - h[2] * h[4],
             h[5] * h[6] - h[3] * h[8], h[0] * h[8] - h[2] * h[6], h[2] * h[3] - h[0] * h[5],
             h[3] * h[7] - h[4] * h[6], h[1] * h[6] - h[0] * h[7], h[0] * h[4] - h[1] * h[3]};
    double det = h[0] * a[0] + h[1] * a[3] + h[2] * a[6];
    if (!std::isfinite(det) || std::abs(det) < 1e-12)
        throw std::runtime_error("Singular homography");
    for (auto &v : a)
        v /= det;
    return a;
}
double sample(const Image &image, double x, double y, std::array<double, 3> &color) {
    if (x < 0 || y < 0 || x > image.w - 1 || y > image.h - 1)
        return 0;
    int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, image.w - 1),
        y1 = std::min(y0 + 1, image.h - 1);
    double dx = x - x0, dy = y - y0;
    for (int c = 0; c < 3; ++c)
        color[c] = (1 - dy) * ((1 - dx) * image.rgb[3 * (y0 * image.w + x0) + c] +
                               dx * image.rgb[3 * (y0 * image.w + x1) + c]) +
                   dy * ((1 - dx) * image.rgb[3 * (y1 * image.w + x0) + c] +
                         dx * image.rgb[3 * (y1 * image.w + x1) + c]);
    return 1 + std::min({x, y, image.w - 1 - x, image.h - 1 - y});
}
} // namespace
Image warp_blend_v0(const Image &a, const Image &b, const HomographyResult &model, const Params &) {
    auto inv = inverse(model.h);
    double xmin = 0, ymin = 0, xmax = a.w - 1, ymax = a.h - 1;
    double sign_min = std::numeric_limits<double>::infinity(), sign_max = -sign_min;
    for (int y : {0, b.h - 1})
        for (int x : {0, b.w - 1}) {
            double z = model.h[6] * x + model.h[7] * y + model.h[8];
            sign_min = std::min(sign_min, z);
            sign_max = std::max(sign_max, z);
            auto q = project(model.h, x, y);
            xmin = std::min(xmin, q[0]);
            ymin = std::min(ymin, q[1]);
            xmax = std::max(xmax, q[0]);
            ymax = std::max(ymax, q[1]);
        }
    if (sign_min <= 0 && sign_max >= 0)
        throw std::runtime_error("Horizon crosses image B");
    xmin = std::floor(xmin);
    ymin = std::floor(ymin);
    xmax = std::ceil(xmax);
    ymax = std::ceil(ymax);
    double width = xmax - xmin + 1, height = ymax - ymin + 1;
    if (!std::isfinite(width * height) || width > 30000 || height > 30000 ||
        width * height > 80000000)
        throw std::runtime_error("Unsafe panorama canvas");
    Image out{int(width), int(height),
              std::vector<unsigned char>(size_t(width) * size_t(height) * 3)};
    for (int y = 0; y < out.h; ++y)
        for (int x = 0; x < out.w; ++x) {
            double ax = x + xmin, ay = y + ymin;
            std::array<double, 3> ca{}, cb{};
            double wa = sample(a, ax, ay, ca), wb = 0;
            try {
                auto q = project(inv, ax, ay);
                wb = sample(b, q[0], q[1], cb);
            } catch (const std::runtime_error &) {
            }
            if (wa + wb == 0)
                continue;
            for (int c = 0; c < 3; ++c)
                out.rgb[3 * (size_t(y) * out.w + x) + c] = static_cast<unsigned char>(
                    std::clamp(std::round((wa * ca[c] + wb * cb[c]) / (wa + wb)), 0.0, 255.0));
        }
    return out;
}
