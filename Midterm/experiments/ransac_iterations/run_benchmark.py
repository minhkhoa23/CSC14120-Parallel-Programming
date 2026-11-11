"""Compare original V0 at 2000 vs 10000 RANSAC iterations, including its real CLI.

Run with Python + numpy + matplotlib + Pillow and g++ on PATH:
    python run_benchmark.py --build --runs 5
    python run_benchmark.py --summarize-only
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
MIDTERM = HERE.parents[1]
REPO = MIDTERM.parent
SOURCE = MIDTERM / "src"
BUILD = MIDTERM / "build" / "ransac_iterations"
OUTPUT = MIDTERM / "outputs"
REPORT = OUTPUT / "ransac_iterations"
BASELINE = REPORT / "2000"
SIZES = ["1280x720", "1920x1080", "3840x2160"]
STAGES = ["preprocess", "detection", "description", "matching", "homography", "warp"]
EXES = {2000: BUILD / ("stitch_2000.exe" if os.name == "nt" else "stitch_2000"),
        10000: MIDTERM / ("stitch.exe" if os.name == "nt" else "stitch")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_versions():
    BUILD.mkdir(parents=True, exist_ok=True)
    snapshot = BUILD / "baseline_src"
    shutil.copytree(SOURCE, snapshot, dirs_exist_ok=True)
    header = snapshot / "common.h"
    text = header.read_text(encoding="utf-8")
    assert text.count("iterations = 10000") == 1
    header.write_text(text.replace("iterations = 10000", "iterations = 2000"), encoding="utf-8")
    commands = {}
    for iterations, source in [(2000, snapshot), (10000, SOURCE)]:
        command = ["g++", "-O2", "-std=c++17", "-x", "c++", str(source / "main.cu"),
                   str(source / "stb_impl.cpp"), *map(str, sorted((source / "v0").glob("*.cpp"))),
                   "-o", str(EXES[iterations])]
        print(f"BUILD {iterations}", flush=True)
        subprocess.run(command, check=True)
        commands[str(iterations)] = command
    return commands


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    environment = json.loads((REPORT / "environment.json").read_text(encoding="utf-8"))
    records = []
    for size in environment["sizes"]:
        for site in environment["sites"]:
            name = f"image{site}_{size}"
            for iterations, directory in [(2000, BASELINE), (10000, OUTPUT)]:
                status = (directory / f"{name}.status").read_text().strip()
                data = json.loads((directory / f"{name}.json").read_text()) if status == "ok" else None
                row = {"case": name, "iterations": iterations, "status": status,
                       "mean_ms": data["mean_ms"] if data else None,
                       "min_ms": data["min_ms"] if data else None,
                       "stddev_ms": data["stddev_ms"] if data else None,
                       "keypoints_a": data["keypoints"][0] if data else None,
                       "keypoints_b": data["keypoints"][1] if data else None,
                       "matches": data["matches"] if data else None,
                       "inliers": data["inliers"] if data else None,
                       "inlier_error_px": data["inlier_error_px"] if data else None,
                       "error": "" if data else (directory / f"{name}.log").read_text().strip()}
                for i, stage in enumerate(STAGES):
                    row[stage + "_ms"] = data["stage_mean_ms"][i] if data else None
                records.append(row)
    with (REPORT / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    cases = [f"image{site}_{size}" for size in environment["sizes"] for site in environment["sites"]]
    by_key = {(row["case"], row["iterations"]): row for row in records}
    common = [case for case in cases if all(by_key[case, n]["status"] == "ok" for n in (2000, 10000))]
    sums = {str(n): sum(by_key[c, n]["mean_ms"] for c in common) for n in (2000, 10000)}
    stage5 = {str(n): sum(by_key[c, n]["homography_ms"] for c in common) for n in (2000, 10000)}
    summary = {"cases": len(cases), "runs": environment["runs"], "common_success_cases": len(common),
               "successful_cases": {str(n): sum(by_key[c, n]["status"] == "ok" for c in cases) for n in (2000, 10000)},
               "sum_e2e_common_ms": sums, "sum_homography_common_ms": stage5,
               "full_pipeline_increase_percent": (sums["10000"] / sums["2000"] - 1) * 100 if common else None,
               "homography_time_ratio": stage5["10000"] / stage5["2000"] if common else None,
               "failed_cases": {str(n): [c for c in cases if by_key[c, n]["status"] != "ok"] for n in (2000, 10000)}}
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    x = np.arange(len(cases))
    for offset, n, color in [(-0.19, 2000, "#315fa6"), (0.19, 10000, "#c96528")]:
        values = [by_key[c, n] for c in cases]
        axes[0].bar(x + offset, [v["homography_ms"] if v["status"] == "ok" else np.nan for v in values],
                    0.38, color=color, label=f"RANSAC {n:,}")
        times = [v["mean_ms"] if v["status"] == "ok" else np.nan for v in values]
        axes[1].bar(x + offset, times, 0.38, color=color,
                    yerr=[v["stddev_ms"] if v["status"] == "ok" else 0 for v in values], capsize=2)
        for i, value in enumerate(times):
            if np.isnan(value):
                for axis in axes:
                    axis.text(i + offset, 0, "FAIL", color=color, rotation=90, ha="center", va="bottom")
    for axis in axes:
        axis.set_xticks(x, [c.replace("_", "\n") for c in cases], fontsize=8)
        axis.set_ylabel("Mean time (ms)")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_title("Stage 5: homography estimation")
    axes[0].legend()
    axes[1].set_title("Full V0 pipeline excluding disk I/O; error bars = SD")
    fig.suptitle(f"DoG V0 | RANSAC 2,000 vs 10,000 iterations\nCPU, g++ -O2; one warm-up + {environment['runs']} timed runs; method order alternates between cases")
    fig.savefig(REPORT / "benchmark.png", dpi=150)
    plt.close(fig)

    for size, short in zip(SIZES, ("720p", "1080p", "4k")):
        fig, axes = plt.subplots(4, 2, figsize=(14, 14))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.97, hspace=0.16, wspace=0.06)
        for i in range(4):
            case = f"image{i + 1}_{size}"
            for j, (n, directory) in enumerate([(2000, BASELINE), (10000, OUTPUT)]):
                record = by_key.get((case, n))
                if record and record["status"] == "ok":
                    with Image.open(directory / f"{case}.png") as im:
                        im.thumbnail((1400, 900))
                        axes[i, j].imshow(im)
                else:
                    axes[i, j].text(0.5, 0.5, "Pipeline failed / no result", ha="center")
                axes[i, j].set_title(f"image{i + 1}, {size} | RANSAC {n:,}", fontsize=11, pad=8)
                axes[i, j].axis("off")
        fig.savefig(REPORT / f"panorama_comparison_{short}.png", dpi=130)
        plt.close(fig)

    def fmt(value, digits=1):
        return "FAIL" if value is None else f"{value:.{digits}f}"
    lines = ["# Benchmark V0: RANSAC 2.000 và 10.000 lần", "",
             "Benchmark chạy trên Windows, CPU i7-11370H, g++ -O2 -std=c++17. Chạy lại cả hai cấu hình bằng executable thật, cùng mã nguồn và cùng dữ liệu; chỉ khác Params.iterations.", "",
             f"Mỗi cấu hình chạy khởi động một lần và đo {environment['runs']} lần nếu khởi động thành công. Cấu hình thất bại ở warm-up không có thời gian toàn pipeline; không dùng thời gian thất bại để tính mức tăng hiệu năng.",
             "Đảo thứ tự cấu hình giữa các cặp ảnh. Thời gian đo không gồm đọc/ghi file ảnh. Không dùng số đo Colab hay số đo lịch sử để tính chênh lệch.", "",
             f"Xuất được panorama: 2.000 lần = {summary['successful_cases']['2000']}/{len(cases)}; 10.000 lần = {summary['successful_cases']['10000']}/{len(cases)}.",
             f"Trên {len(common)} trường hợp cùng thành công, thời gian toàn pipeline tăng {summary['full_pipeline_increase_percent']:.1f}%; thời gian homography tăng {summary['homography_time_ratio']:.2f} lần (tỷ lệ tổng các giá trị trung bình)." if common else "Không có trường hợp cùng thành công để so sánh thời gian.", "",
             "| Trường hợp | Tổng 2.000 (ms) | Tổng 10.000 (ms) | Homography 2.000/10.000 (ms) | Inlier 2.000/10.000 | Sai số inlier 2.000/10.000 (px) |",
             "|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        a, b = by_key[case, 2000], by_key[case, 10000]
        lines.append(f"| {case} | {fmt(a['mean_ms'])} | {fmt(b['mean_ms'])} | {fmt(a['homography_ms'])} / {fmt(b['homography_ms'])} | {a['inliers'] if a['inliers'] is not None else 'FAIL'} / {b['inliers'] if b['inliers'] is not None else 'FAIL'} | {fmt(a['inlier_error_px'],2)} / {fmt(b['inlier_error_px'],2)} |")
    lines.extend(["", "Các số inlier và sai số là độ đồng thuận với homography của từng cấu hình; không phải độ chính xác đo bằng ground truth. Tăng số lần RANSAC không bảo đảm hết bóng đôi do sai lệch phối cảnh/parallax.", "",
                  "Kết quả 10.000 lần được lưu trực tiếp trong Midterm/outputs để notebook đọc lại. Kết quả 2.000 lần nằm trong thư mục 2000 của báo cáo này.", "",
                  "[Biểu đồ](benchmark.png) · [CSV](benchmark.csv) · [720p](panorama_comparison_720p.png) · [1080p](panorama_comparison_1080p.png) · [4K](panorama_comparison_4k.png)", ""])
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--sites", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    BASELINE.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize()
        return
    commands = compile_versions() if args.build else None
    previous = json.loads((REPORT / "environment.json").read_text()) if (REPORT / "environment.json").exists() else {}
    environment = {"platform": platform.platform(), "runs": args.runs, "sites": args.sites,
                   "sizes": args.sizes, "build_commands": commands or previous.get("build_commands"),
                   "compiler": subprocess.check_output(["g++", "--version"], text=True),
                   "source_sha256": {p.relative_to(SOURCE).as_posix(): sha(p) for p in sorted(SOURCE.rglob("*")) if p.is_file()},
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (REPORT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    for i, (size, site) in enumerate((size, site) for size in args.sizes for site in args.sites):
        name = f"image{site}_{size}"
        partner = "mid" if site == 2 else "right"
        for n in ([2000, 10000] if i % 2 == 0 else [10000, 2000]):
            directory = BASELINE if n == 2000 else OUTPUT
            print(f"START {name} iterations={n}", flush=True)
            command = [str(EXES[n]), str(REPO / "Dataset" / f"image{site}_left_{size}.jpg"),
                       str(REPO / "Dataset" / f"image{site}_{partner}_{size}.jpg"), str(directory / f"{name}.png"),
                       "--version", "0", "--runs", str(args.runs), "--metrics", str(directory / f"{name}.json")]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (directory / f"{name}.log").write_text(result.stdout, encoding="utf-8")
            status = "ok" if result.returncode == 0 else "error"
            (directory / f"{name}.status").write_text(status, encoding="utf-8")
            if status == "ok":
                data = json.loads((directory / f"{name}.json").read_text())
                assert data["params"]["ransac_iterations"] == n
                assert data["runs"] == args.runs
                print(f"DONE {name} iterations={n}: mean={data['mean_ms']:.2f} ms, stage5={data['stage_mean_ms'][4]:.2f} ms, inliers={data['inliers']}", flush=True)
            else:
                print(result.stdout, end="", flush=True)
    assert all(sha(SOURCE / p) == value for p, value in environment["source_sha256"].items())
    summarize()


if __name__ == "__main__":
    main()
