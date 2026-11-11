#include "harris_detector.h"

std::vector<Keypoint> detect_harris(const PreprocessedImage &im, const Params &p,
                                   const HarrisParams &hp) {
    const int w = im.w, h = im.h;
    if (w < 31 || h < 31 || p.max_keypoints <= 0)
        return {};
    if (im.gaussian.empty() || im.gaussian[0].size() != size_t(w) * h)
        throw std::runtime_error("Invalid Harris input");
    if (hp.integration_sigma <= 0 || hp.quality_level <= 0 || hp.quality_level >= 1)
        throw std::runtime_error("Invalid Harris parameters");

    const auto &smooth = im.gaussian[0]; // Same sigma=1 image used by BRIEF.
    struct Tensor { float xx = 0, xy = 0, yy = 0; };
    std::vector<Tensor> moment(size_t(w) * h), horizontal(moment.size());
    for (int y = 1; y < h - 1; ++y)
        for (int x = 1; x < w - 1; ++x) {
            const int i = y * w + x;
            const float ix = 0.5f * (smooth[i + 1] - smooth[i - 1]);
            const float iy = 0.5f * (smooth[i + w] - smooth[i - w]);
            moment[i] = {ix * ix, ix * iy, iy * iy};
        }

    // Gaussian integration of the gradient second-moment matrix, separable.
    const int radius = int(std::ceil(3 * hp.integration_sigma));
    std::vector<float> kernel(2 * radius + 1);
    double sum = 0;
    for (int k = -radius; k <= radius; ++k) {
        kernel[k + radius] = float(std::exp(-k * k / (2 * hp.integration_sigma * hp.integration_sigma)));
        sum += kernel[k + radius];
    }
    for (auto &v : kernel)
        v = float(v / sum);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            Tensor value;
            for (int k = -radius; k <= radius; ++k) {
                const auto &v = moment[y * w + std::clamp(x + k, 0, w - 1)];
                const float weight = kernel[k + radius];
                value.xx += weight * v.xx;
                value.xy += weight * v.xy;
                value.yy += weight * v.yy;
            }
            horizontal[y * w + x] = value;
        }

    std::vector<float> response(moment.size());
    double maximum = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            Tensor value;
            for (int k = -radius; k <= radius; ++k) {
                const auto &v = horizontal[std::clamp(y + k, 0, h - 1) * w + x];
                const float weight = kernel[k + radius];
                value.xx += weight * v.xx;
                value.xy += weight * v.xy;
                value.yy += weight * v.yy;
            }
            const double trace = double(value.xx) + value.yy;
            const double r = double(value.xx) * value.yy - double(value.xy) * value.xy - hp.k * trace * trace;
            response[y * w + x] = float(r);
            if (x >= 15 && x < w - 15 && y >= 15 && y < h - 15)
                maximum = std::max(maximum, double(response[y * w + x]));
        }
    if (maximum <= 0)
        return {};

    const double threshold = maximum * hp.quality_level;
    std::vector<Keypoint> points;
    for (int y = 15; y < h - 15; ++y)
        for (int x = 15; x < w - 15; ++x) {
            const int i = y * w + x;
            const float r = response[i];
            if (r <= threshold)
                continue;
            bool keep = true;
            for (int dy = -1; dy <= 1 && keep; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0)
                        continue;
                    const int j = i + dy * w + dx;
                    if (response[j] > r || (response[j] == r && j < i)) {
                        keep = false;
                        break;
                    }
                }
            if (keep)
                points.push_back({double(x), double(y), 0, double(r)});
        }
    std::sort(points.begin(), points.end(), [](const Keypoint &a, const Keypoint &b) {
        if (a.response != b.response) return a.response > b.response;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });
    if (points.size() > size_t(p.max_keypoints))
        points.resize(p.max_keypoints);
    return points;
}
