// RANSAC: ParaPano; normalized DLT: Hartley và Zisserman.
#include "../common.h"
#include <numeric>
#include <random>

namespace {
Matrix multiply(const Matrix &a, const Matrix &b) {
    Matrix c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
    return c;
}
struct Normalization {
    double x, y, scale;
};
Normalization normalize(const Features &f, const std::vector<Match> &matches,
                        const std::vector<int> &ids, bool source) {
    double x = 0, y = 0, distance = 0;
    for (int i : ids) {
        const auto &q = f.points[source ? matches[i].b : matches[i].a];
        x += q.x;
        y += q.y;
    }
    x /= ids.size();
    y /= ids.size();
    for (int i : ids) {
        const auto &q = f.points[source ? matches[i].b : matches[i].a];
        distance += std::hypot(q.x - x, q.y - y);
    }
    distance /= ids.size();
    if (distance < 1e-8)
        throw std::runtime_error("Coincident points");
    return {x, y, std::sqrt(2.0) / distance};
}
Matrix fit(const Features &a, const Features &b, const std::vector<Match> &matches,
           const std::vector<int> &ids) {
    auto na = normalize(a, matches, ids, false), nb = normalize(b, matches, ids, true);
    double c[9][9]{}, vectors[9][9]{};
    for (int i = 0; i < 9; ++i)
        vectors[i][i] = 1;
    for (int id : ids) {
        auto u = a.points[matches[id].a], v = b.points[matches[id].b];
        double x = (v.x - nb.x) * nb.scale, y = (v.y - nb.y) * nb.scale;
        double X = (u.x - na.x) * na.scale, Y = (u.y - na.y) * na.scale;
        double rows[2][9] = {{-x, -y, -1, 0, 0, 0, X * x, X * y, X},
                             {0, 0, 0, -x, -y, -1, Y * x, Y * y, Y}};
        for (auto &row : rows)
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    c[i][j] += row[i] * row[j];
    }
    // Chéo hóa A^T A bằng Jacobi sau khi chuẩn hóa tọa độ.
    for (int iteration = 0; iteration < 1000; ++iteration) {
        int p = 0, q = 1;
        for (int i = 0; i < 9; ++i)
            for (int j = i + 1; j < 9; ++j)
                if (std::abs(c[i][j]) > std::abs(c[p][q])) {
                    p = i;
                    q = j;
                }
        if (std::abs(c[p][q]) < 1e-11)
            break;
        double angle = 0.5 * std::atan2(2 * c[p][q], c[q][q] - c[p][p]);
        double co = std::cos(angle), si = std::sin(angle);
        double pp = c[p][p], qq = c[q][q], pq = c[p][q];
        c[p][p] = co * co * pp - 2 * si * co * pq + si * si * qq;
        c[q][q] = si * si * pp + 2 * si * co * pq + co * co * qq;
        c[p][q] = c[q][p] = 0;
        for (int k = 0; k < 9; ++k) {
            if (k != p && k != q) {
                double kp = c[k][p], kq = c[k][q];
                c[k][p] = c[p][k] = co * kp - si * kq;
                c[k][q] = c[q][k] = si * kp + co * kq;
            }
            double vp = vectors[k][p], vq = vectors[k][q];
            vectors[k][p] = co * vp - si * vq;
            vectors[k][q] = si * vp + co * vq;
        }
    }
    std::array<int, 9> order{};
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) { return c[i][i] < c[j][j]; });
    if (c[order[1]][order[1]] < 1e-8)
        throw std::runtime_error("Degenerate DLT sample");
    Matrix h{};
    for (int i = 0; i < 9; ++i)
        h[i] = vectors[i][order[0]];
    Matrix tb{nb.scale, 0, -nb.scale * nb.x, 0, nb.scale, -nb.scale * nb.y, 0, 0, 1};
    Matrix inverse_a{1 / na.scale, 0, na.x, 0, 1 / na.scale, na.y, 0, 0, 1};
    h = multiply(multiply(inverse_a, h), tb);
    if (std::abs(h[8]) < 1e-10)
        throw std::runtime_error("Invalid homography scale");
    double scale = h[8];
    for (auto &v : h) {
        v /= scale;
        if (!std::isfinite(v))
            throw std::runtime_error("Nonfinite homography");
    }
    return h;
}
HomographyResult score(Matrix h, const Features &a, const Features &b,
                       const std::vector<Match> &matches, const Params &p) {
    HomographyResult result;
    result.h = h;
    for (size_t i = 0; i < matches.size(); ++i) {
        auto u = a.points[matches[i].a], v = b.points[matches[i].b];
        try {
            auto q = project(h, v.x, v.y);
            double error = std::hypot(q[0] - u.x, q[1] - u.y);
            if (error <= p.reprojection) {
                result.inliers.push_back(int(i));
                result.error += error;
            }
        } catch (const std::runtime_error &) { /* Không dùng điểm có mẫu số chiếu gần 0. */
        }
    }
    return result;
}
} // namespace
HomographyResult estimate_homography_v0(const Features &a, const Features &b,
                                        const std::vector<Match> &matches, const Params &p) {
    if (matches.size() < 4)
        throw std::runtime_error("Need at least four matches");
    std::mt19937 rng(p.seed);
    std::uniform_int_distribution<int> choose(0, int(matches.size()) - 1);
    HomographyResult best;
    for (int iteration = 0; iteration < p.iterations; ++iteration) {
        std::vector<int> sample;
        while (sample.size() < 4) {
            int id = choose(rng);
            if (std::find(sample.begin(), sample.end(), id) == sample.end())
                sample.push_back(id);
        }
        try {
            auto candidate = score(fit(a, b, matches, sample), a, b, matches, p);
            if (candidate.inliers.size() > best.inliers.size() ||
                (candidate.inliers.size() == best.inliers.size() && candidate.error < best.error))
                best = std::move(candidate);
        } catch (const std::runtime_error &) { /* Bỏ mẫu suy biến và thử mẫu khác. */
        }
    }
    if (best.inliers.size() < 4)
        throw std::runtime_error("RANSAC found no valid model");
    auto refined = score(fit(a, b, matches, best.inliers), a, b, matches, p);
    if (refined.inliers.size() >= best.inliers.size())
        best = std::move(refined);
    best.error /= best.inliers.size();
    return best;
}
