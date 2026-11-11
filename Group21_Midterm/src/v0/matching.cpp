// Tham khảo Hamming matching: ParaPano/README.md (MIT).
#include "../common.h"
int hamming(const Descriptor &a, const Descriptor &b) {
    int distance = 0;
    for (int word = 0; word < 8; ++word) {
        uint32_t bits = a[word] ^ b[word];
        while (bits) {
            bits &= bits - 1;
            ++distance;
        }
    }
    return distance;
}
std::vector<Match> match_v0(const Features &a, const Features &b, const Params &p) {
    std::vector<Match> out;
    if (b.descriptors.size() < 2)
        return out;
    for (size_t i = 0; i < a.descriptors.size(); ++i) {
        int best = 257, second = 257, index = -1;
        for (size_t j = 0; j < b.descriptors.size(); ++j) {
            int d = hamming(a.descriptors[i], b.descriptors[j]);
            if (d < best) {
                second = best;
                best = d;
                index = int(j);
            } else if (d < second)
                second = d;
        }
        if (second > 0 && best < p.ratio * second)
            out.push_back({int(i), index, best});
    }
    return out;
}
