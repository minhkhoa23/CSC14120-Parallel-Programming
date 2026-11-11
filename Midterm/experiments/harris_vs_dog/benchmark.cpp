#include "harris_detector.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

using Clock = std::chrono::steady_clock;
struct Attempt {
    PipelineResult result;
    std::array<double, 6> stages;
    std::vector<std::array<double, 4>> inlier_pairs;
    bool success = false;
    int completed = 0;
    std::string error;
    double elapsed_ms = 0;
    double diagnostics_ms = 0;
    Attempt() { stages.fill(std::numeric_limits<double>::quiet_NaN()); }
};

Attempt pipeline(const Image &a, const Image &b, const Params &p, bool harris) {
    Attempt attempt;
    auto begin = Clock::now();
    auto stamp = [&](int stage) {
        const auto end = Clock::now();
        attempt.stages[stage] = std::chrono::duration<double, std::milli>(end - begin).count();
        attempt.completed = stage + 1;
        begin = end;
    };
    try {
        auto pa = preprocess_v0(a, p), pb = preprocess_v0(b, p);
        stamp(0);
        auto ka = harris ? detect_harris(pa, p) : detect_v0(pa, p);
        auto kb = harris ? detect_harris(pb, p) : detect_v0(pb, p);
        stamp(1);
        auto fa = describe_v0(pa, ka, p), fb = describe_v0(pb, kb, p);
        stamp(2);
        attempt.result.points_a = int(fa.points.size());
        attempt.result.points_b = int(fb.points.size());
        auto matches = match_v0(fa, fb, p);
        stamp(3);
        attempt.result.matches = int(matches.size());
        attempt.result.model = estimate_homography_v0(fa, fb, matches, p);
        stamp(4);
        attempt.result.panorama = warp_blend_v0(a, b, attempt.result.model, p);
        stamp(5);
        attempt.success = true;
        // Diagnostics are collected after the six timed stages.
        const auto diagnostic_begin = Clock::now();
        for (int id : attempt.result.model.inliers) {
            const auto &u = fa.points[matches[id].a];
            const auto &v = fb.points[matches[id].b];
            attempt.inlier_pairs.push_back({u.x, u.y, v.x, v.y});
        }
        attempt.diagnostics_ms = std::chrono::duration<double, std::milli>(Clock::now() - diagnostic_begin).count();
    } catch (const std::exception &e) {
        attempt.error = e.what();
    }
    return attempt;
}

std::string quoted(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c >= 32) out += char(c);
    }
    return out + '"';
}
void number(std::ostream &out, double x) {
    if (std::isfinite(x)) out << x; else out << "null";
}
double mean(const std::vector<double> &v) {
    return v.empty() ? std::numeric_limits<double>::quiet_NaN()
                     : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

void save(const std::string &prefix, const std::string &detector, const Image &a, const Image &b,
          const std::vector<Attempt> &attempts, const Attempt &last, int first_detector) {
    const std::string root = prefix + "_" + detector;
    if (last.success) write_image(root + ".png", last.result.panorama);
    std::ofstream out(root + ".json");
    if (!out) throw std::runtime_error("Cannot write benchmark JSON");
    out << std::setprecision(15);
    int successes = 0;
    std::vector<double> totals;
    std::array<std::vector<double>, 6> stages;
    for (const auto &r : attempts) {
        if (r.success) { ++successes; totals.push_back(r.elapsed_ms); }
        for (int s = 0; s < r.completed; ++s) stages[s].push_back(r.stages[s]);
    }
    const auto &r = last.result;
    out << "{\n\"detector\":" << quoted(detector)
        << ",\"status\":" << quoted(successes == int(attempts.size()) ? "ok" : successes == 0 ? "error" : "mixed")
        << ",\"error\":" << quoted(last.error) << ",\"runs\":" << attempts.size()
        << ",\"successful_runs\":" << successes << ",\"warmup_runs\":1,\"seed\":21,"
        << "\"input_a\":[" << a.w << ',' << a.h << "],\"input_b\":[" << b.w << ',' << b.h << "],"
        << "\"harris_params\":{\"k\":0.04,\"integration_sigma\":1.5,\"quality_level\":0.01,\"nms_window\":3},"
        << "\"shared_params\":{\"max_keypoints\":4000,\"dog_contrast\":0.03,\"ratio\":0.8,"
        << "\"ransac_iterations\":" << Params{}.iterations << ",\"reprojection_px\":3,\"brief_bits\":256,\"patch_size\":31},\n"
        << "\"mean_ms\":";
    number(out, mean(totals));
    out << ",\"min_ms\":";
    number(out, totals.empty() ? std::numeric_limits<double>::quiet_NaN() : *std::min_element(totals.begin(), totals.end()));
    double variance = 0;
    for (double value : totals) variance += (value - mean(totals)) * (value - mean(totals)) / totals.size();
    out << ",\"stddev_ms\":"; number(out, totals.empty() ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(variance));
    out << ",\"stage_mean_ms\":[";
    for (int s = 0; s < 6; ++s) { if (s) out << ','; number(out, mean(stages[s])); }
    out << "],\"runs_data\":[";
    for (size_t i = 0; i < attempts.size(); ++i) {
        if (i) out << ',';
        const auto &row = attempts[i];
        out << "{\"success\":" << (row.success ? "true" : "false") << ",\"elapsed_attempt_ms\":" << row.elapsed_ms
            << ",\"completed_stages\":" << row.completed << ",\"error\":" << quoted(row.error)
            << ",\"ran_first\":" << ((int(i) + first_detector) % 2 == (detector == "dog" ? 0 : 1) ? "true" : "false")
            << ",\"stage_ms\":[";
        for (int s = 0; s < 6; ++s) { if (s) out << ','; number(out, row.stages[s]); }
        out << "]}";
    }
    out << "],\"keypoints\":[" << r.points_a << ',' << r.points_b << "],\"matches\":" << r.matches
        << ",\"inliers\":" << r.model.inliers.size() << ",\"inlier_error_px\":";
    number(out, last.completed >= 5 ? r.model.error : std::numeric_limits<double>::quiet_NaN());
    out << ",\"H_B_to_A\":[";
    for (int i = 0; i < 9; ++i) { if (i) out << ','; number(out, last.completed >= 5 ? r.model.h[i] : std::numeric_limits<double>::quiet_NaN()); }
    out << "],\"panorama_size\":[" << r.panorama.w << ',' << r.panorama.h << "],\"inlier_pairs\":[";
    for (size_t i = 0; i < last.inlier_pairs.size(); ++i) {
        if (i) out << ',';
        out << '[';
        for (int j = 0; j < 4; ++j) { if (j) out << ','; out << last.inlier_pairs[i][j]; }
        out << ']';
    }
    out << "]\n}\n";
    std::cout << detector << ": " << successes << '/' << attempts.size() << " successful; stage2="
              << mean(stages[1]) << " ms; e2e=" << mean(totals) << " ms; points=" << r.points_a
              << ',' << r.points_b << "; matches=" << r.matches << "; inliers=" << r.model.inliers.size()
              << "; error=" << last.error << '\n';
}

void self_test() {
    Params p;
    auto make = [](int type, int sx = 0, int sy = 0) {
        Image im{96, 96, std::vector<unsigned char>(96 * 96 * 3)};
        for (int y = 0; y < 96; ++y)
            for (int x = 0; x < 96; ++x) {
                bool white = type == 1 ? x >= 48 : type == 2 && x >= 24 + sx && x < 72 + sx && y >= 24 + sy && y < 72 + sy;
                for (int c = 0; c < 3; ++c) im.rgb[3 * (y * 96 + x) + c] = white ? 255 : 0;
            }
        return im;
    };
    for (int type : {0, 1})
        if (!detect_harris(preprocess_v0(make(type), p), p).empty())
            throw std::runtime_error("Harris incorrectly detects a flat image or straight edge");
    const auto original = detect_harris(preprocess_v0(make(2), p), p);
    const auto shifted = detect_harris(preprocess_v0(make(2, 5, 7), p), p);
    if (original.size() != 4 || shifted.size() != 4)
        throw std::runtime_error("Harris did not find the four square corners");
    for (const auto &q : original) {
        bool found = false;
        for (const auto &r : shifted) found |= std::hypot(r.x - q.x - 5, r.y - q.y - 7) < 0.1;
        if (!found) throw std::runtime_error("Harris translation check failed");
    }
    p.max_keypoints = 2;
    if (detect_harris(preprocess_v0(make(2), p), p).size() != 2)
        throw std::runtime_error("Harris point budget failed");
    std::cout << "Harris self-test passed: flat image, edge rejection, four corners, translation, point budget.\n";
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        if (argc < 4 || argc > 6) throw std::runtime_error("Usage: benchmark left right output_prefix [runs=5] [first=dog|harris]");
        const int runs = argc >= 5 ? std::stoi(argv[4]) : 5;
        const int first = argc >= 6 && std::string(argv[5]) == "harris" ? 1 : 0;
        if (runs < 1) throw std::runtime_error("runs must be positive");
        const auto a = read_image(argv[1]), b = read_image(argv[2]);
        const Params p;
        std::array<std::vector<Attempt>, 2> timings;
        std::array<Attempt, 2> last;
        for (int i = 0; i < 2; ++i) { auto warmup = pipeline(a, b, p, (i + first) % 2 == 1); }
        for (int run = 0; run < runs; ++run)
            for (int i = 0; i < 2; ++i) {
                const int method = (run + i + first) % 2;
                const auto begin = Clock::now();
                auto current = pipeline(a, b, p, method == 1);
                current.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count()
                                     - current.diagnostics_ms;
                // Images and coordinates are retained only for the last timed run.
                Attempt compact;
                compact.success = current.success;
                compact.completed = current.completed;
                compact.error = current.error;
                compact.elapsed_ms = current.elapsed_ms;
                compact.stages = current.stages;
                timings[method].push_back(std::move(compact));
                last[method] = std::move(current);
            }
        for (int method = 0; method < 2; ++method)
            save(argv[3], method == 0 ? "dog" : "harris", a, b, timings[method], last[method], first);
    } catch (const std::exception &e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
