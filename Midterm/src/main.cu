#include "versions.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace std;


int main(int argc, char **argv) {
    try {
        if (argc < 4)
            throw runtime_error(
                "Usage: stitch left right output --version 0 [--runs N] [--metrics result.json]");
        int runs = 5, version = 0;
        string metrics;
        for (int i = 4; i < argc; ++i) {
            string arg = argv[i];
            if (i + 1 >= argc)
                throw runtime_error("Missing option value");
            string value = argv[++i];
            if (arg == "--metrics")
                metrics = value;
            else {
                size_t used = 0;
                int number = stoi(value, &used);
                if (used != value.size())
                    throw runtime_error("Invalid integer option");
                if (arg == "--runs")
                    runs = number;
                else if (arg == "--version")
                    version = number;
                else
                    throw runtime_error("Unknown option: " + arg);
            }
        }
        if (version != 0 || runs < 1)
            throw runtime_error("Only V0 is available; runs must be positive");
        auto a = read_image(argv[1]), b = read_image(argv[2]);
        Params params;
        {
            auto warmup = run_v0(a, b, params);
        }
        vector<array<double, 6>> stages;
        vector<double> totals;
        PipelineResult result;
        for (int run = 0; run < runs; ++run) {
            auto begin = chrono::high_resolution_clock::now();
            auto current = run_v0(a, b, params);
            auto end = chrono::high_resolution_clock::now();
            totals.push_back(chrono::duration<double, milli>(end - begin).count());
            stages.push_back(current.ms);
            result = move(current);
        }
        write_image(argv[3], result.panorama);
        array<double, 6> means{};
        for (const auto &row : stages)
            for (int s = 0; s < 6; ++s)
                means[s] += row[s] / runs;
        double mean = accumulate(totals.begin(), totals.end(), 0.0) / runs, variance = 0;
        for (double t : totals)
            variance += (t - mean) * (t - mean) / runs;
        cout << "V0 CPU, " << runs << " timed runs after warm-up\n";
        for (int s = 0; s < 6; ++s)
            cout << "Stage " << s + 1 << ": " << means[s] << " ms\n";
        cout << "End-to-end excluding disk I/O: " << mean << " ms\n";
        if (!metrics.empty()) {
            ofstream out(metrics);
            if (!out)
                throw runtime_error("Cannot write metrics");
            out << setprecision(12) << "{\n\"version\":0,\"runs\":" << runs
                << ",\"seed\":21,\"input_a\":[" << a.w << ',' << a.h << "],\"input_b\":[" << b.w
                << ',' << b.h << "],\n";
            out << "\"params\":{\"max_keypoints\":4000,\"contrast\":0.03,\"ratio\":0.8,\"ransac_"
                   "iterations\":" << params.iterations << ",\"reprojection_px\":3,\"brief_bits\":256,\"patch_size\":31},"
                   "\n";
            out << "\"stage_mean_ms\":[";
            for (int s = 0; s < 6; ++s)
                out << (s ? "," : "") << means[s];
            out << "],\"runs_stage_ms\":[";
            for (int r = 0; r < runs; ++r) {
                out << (r ? ",[" : "[");
                for (int s = 0; s < 6; ++s)
                    out << (s ? "," : "") << stages[r][s];
                out << ']';
            }
            out << "],\"runs_e2e_ms\":[";
            for (int r = 0; r < runs; ++r)
                out << (r ? "," : "") << totals[r];
            out << "],\"mean_ms\":" << mean
                << ",\"min_ms\":" << *min_element(totals.begin(), totals.end())
                << ",\"stddev_ms\":" << sqrt(variance);
            out << ",\"keypoints\":[" << result.points_a << ',' << result.points_b
                << "],\"matches\":" << result.matches
                << ",\"inliers\":" << result.model.inliers.size()
                << ",\"inlier_error_px\":" << result.model.error << ",\"H_B_to_A\":[";
            for (int i = 0; i < 9; ++i)
                out << (i ? "," : "") << result.model.h[i];
            out << "],\"gpu_kernel_ms\":null,\"transfer_ms\":null}\n";
        }
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
