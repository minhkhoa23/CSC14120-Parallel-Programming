// Tham khảo DoG: ParaPano/keyPointDetector.cpp (MIT).
#include "../common.h"
std::vector<Keypoint> detect_v0(const PreprocessedImage &im, const Params &p) {
    int w = im.w, h = im.h;
    std::vector<std::vector<float>> dog(4, std::vector<float>(size_t(w) * h));
    for (int s = 0; s < 4; ++s)
        for (size_t i = 0; i < dog[s].size(); ++i)
            dog[s][i] = im.gaussian[s + 1][i] - im.gaussian[s][i];
    std::vector<Keypoint> points;
    for (int s = 1; s < 3; ++s)
        for (int y = 15; y < h - 15; ++y)
            for (int x = 15; x < w - 15; ++x) {
                int i = y * w + x;
                float v = dog[s][i];
                if (std::abs(v) < p.contrast)
                    continue;
                bool maximum = true, minimum = true;
                for (int ds = -1; ds <= 1; ++ds)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (!ds && !dy && !dx)
                                continue;
                            float n = dog[s + ds][i + dy * w + dx];
                            maximum = maximum && v > n;
                            minimum = minimum && v < n;
                        }
                if (!maximum && !minimum)
                    continue;
                const auto &d = dog[s];
                double xx = d[i - 1] + d[i + 1] - 2 * v, yy = d[i - w] + d[i + w] - 2 * v;
                double xy = (d[i + w + 1] - d[i + w - 1] - d[i - w + 1] + d[i - w - 1]) * 0.25;
                double det = xx * yy - xy * xy, trace = xx + yy;
                if (det <= 0 || trace * trace / det >= 12.1)
                    continue;
                points.push_back({double(x), double(y), s, std::abs(v)});
            }
    std::sort(points.begin(), points.end(), [](const Keypoint &a, const Keypoint &b) {
        if (a.response != b.response)
            return a.response > b.response;
        if (a.y != b.y)
            return a.y < b.y;
        if (a.x != b.x)
            return a.x < b.x;
        return a.level < b.level;
    });
    if (points.size() > size_t(p.max_keypoints))
        points.resize(p.max_keypoints);
    return points;
}
