// Tham khảo Gaussian pyramid: ParaPano/filter.cpp (MIT).
#include "../common.h"
PreprocessedImage preprocess_v0(const Image &image, const Params &) {
    int w = image.w, h = image.h;
    if (w < 3 || h < 3 || image.rgb.size() != size_t(w) * h * 3)
        throw std::runtime_error("Invalid image dimensions");
    std::vector<float> gray(size_t(w) * h), temp(gray.size());
    for (size_t i = 0; i < gray.size(); ++i)
        gray[i] = (0.299f * image.rgb[3 * i] + 0.587f * image.rgb[3 * i + 1] +
                   0.114f * image.rgb[3 * i + 2]) /
                  255;
    PreprocessedImage out{w, h, {}};
    for (int level = 0; level < 5; ++level) {
        double sigma = std::pow(std::sqrt(2.0), level), sum = 0;
        int radius = int(std::ceil(3 * sigma));
        std::vector<float> kernel(2 * radius + 1), blurred(gray.size());
        for (int k = -radius; k <= radius; ++k) {
            kernel[k + radius] = std::exp(-k * k / (2 * sigma * sigma));
            sum += kernel[k + radius];
        }
        for (auto &value : kernel)
            value /= sum;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float value = 0;
                for (int k = -radius; k <= radius; ++k)
                    value += kernel[k + radius] * gray[y * w + std::clamp(x + k, 0, w - 1)];
                temp[y * w + x] = value;
            }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float value = 0;
                for (int k = -radius; k <= radius; ++k)
                    value += kernel[k + radius] * temp[std::clamp(y + k, 0, h - 1) * w + x];
                blurred[y * w + x] = value;
            }
        out.gaussian.push_back(std::move(blurred));
    }
    return out;
}
