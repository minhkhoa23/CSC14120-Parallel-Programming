// Đọc và ghi ảnh bằng stb; giấy phép giữ trong các header thư viện.
#define STBI_NO_SIMD
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <filesystem>
Image read_image(const std::string &path) {
    int w, h, channels;
    auto data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data)
        throw std::runtime_error("Cannot load image: " + path);
    Image image{w, h, std::vector<unsigned char>(data, data + size_t(w) * h * 3)};
    stbi_image_free(data);
    return image;
}
void write_image(const std::string &path, const Image &image) {
    auto extension = std::filesystem::path(path).extension().string();
    int ok = 0;
    if (extension == ".png")
        ok = stbi_write_png(path.c_str(), image.w, image.h, 3, image.rgb.data(), image.w * 3);
    else if (extension == ".jpg" || extension == ".jpeg")
        ok = stbi_write_jpg(path.c_str(), image.w, image.h, 3, image.rgb.data(), 95);
    else
        throw std::runtime_error("Output must be .png or .jpg");
    if (!ok)
        throw std::runtime_error("Cannot write image: " + path);
}
