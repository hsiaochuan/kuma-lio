#include <algorithm>
#include <cctype>
#include <fstream>

#include "common_lib.h"

// static: other libraries in the process may carry their own copy of stb
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace faster_lio {

bool RgbImage::Decode(const std::uint8_t *bytes, std::size_t size) {
    int w = 0, h = 0, channels = 0;
    stbi_uc *pixels = stbi_load_from_memory(bytes, static_cast<int>(size), &w, &h, &channels, 3);
    if (pixels == nullptr) return false;
    width = w;
    height = h;
    data.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 3);
    stbi_image_free(pixels);
    return true;
}

bool RgbImage::FromRaw(const std::uint8_t *pixels, int w, int h, int step, const std::string &encoding) {
    int channels = 0;
    if (encoding == "rgb8" || encoding == "bgr8") channels = 3;
    if (encoding == "mono8") channels = 1;
    if (channels == 0 || w <= 0 || h <= 0 || step < w * channels) return false;

    width = w;
    height = h;
    data.resize(static_cast<std::size_t>(w) * h * 3);
    std::uint8_t *out = data.data();
    for (int v = 0; v < h; ++v) {
        const std::uint8_t *row = pixels + static_cast<std::size_t>(v) * step;
        for (int u = 0; u < w; ++u, out += 3) {
            const std::uint8_t *in = row + u * channels;
            out[0] = encoding == "bgr8" ? in[2] : in[0];
            out[1] = channels == 3 ? in[1] : in[0];
            out[2] = encoding == "rgb8" ? in[2] : in[0];
        }
    }
    return true;
}

bool RgbImage::SetEncoded(const std::uint8_t *bytes, std::size_t size, const std::string &msg_format) {
    // CompressedImage::format: "jpeg", "png", or image_transport's "<encoding>; jpeg compressed <encoding>"
    std::string f = msg_format;
    std::transform(f.begin(), f.end(), f.begin(), [](unsigned char c) { return std::tolower(c); });
    if (f.find("png") != std::string::npos) {
        format = "png";
    } else if (f.find("jpeg") != std::string::npos || f.find("jpg") != std::string::npos) {
        format = "jpg";
    } else {
        return false;
    }
    encoded.assign(bytes, bytes + size);
    return true;
}

bool RgbImage::Save(const std::string &stem) const {
    if (Empty()) return false;
    const std::string fname = stem + "." + Extension();
    if (!encoded.empty()) {  // a compressed image is written as received, decoded or not
        std::ofstream ofs(fname, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
        return bool(ofs);
    }
    return stbi_write_png(fname.c_str(), width, height, 3, data.data(), width * 3) != 0;
}

}  // namespace faster_lio
