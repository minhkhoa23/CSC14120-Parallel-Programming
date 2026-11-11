// Tham khảo BRIEF: ParaPano/brief.cpp (MIT).
#include "../brief_pattern.h"
#include "../common.h"
Features describe_v0(const PreprocessedImage &im, const std::vector<Keypoint> &points,
                     const Params &) {
    Features out;
    for (auto point : points) {
        int x = int(point.x), y = int(point.y);
        if (x < 15 || y < 15 || x >= im.w - 15 || y >= im.h - 15)
            continue;
        Descriptor d{};
        const auto &smooth = im.gaussian[0];
        for (int bit = 0; bit < 256; ++bit) {
            const auto &t = brief_pattern[bit];
            if (smooth[(y + t[1]) * im.w + x + t[0]] < smooth[(y + t[3]) * im.w + x + t[2]])
                d[bit / 32] |= uint32_t(1) << (bit % 32);
        }
        out.points.push_back(point);
        out.descriptors.push_back(d);
    }
    return out;
}
