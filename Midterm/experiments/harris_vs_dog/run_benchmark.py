"""Build and compare the original DoG detector against an isolated Harris detector.

Examples (Windows or Linux with g++):
    python run_benchmark.py --build --sites 1 --sizes 1280x720 --runs 1
    python run_benchmark.py --build --runs 5
    python run_benchmark.py --summarize-only
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

HERE = Path(__file__).resolve().parent
MIDTERM = HERE.parents[1]
REPO = MIDTERM.parent
SOURCE = MIDTERM / "src"
BUILD = MIDTERM / "build" / "harris_vs_dog"
OUTPUT = MIDTERM / "outputs" / "harris_vs_dog"
EXE = BUILD / ("benchmark.exe" if os.name == "nt" else "benchmark")
SIZES = ["1280x720", "1920x1080", "3840x2160"]
STAGES = ["preprocess", "detection", "description", "matching", "homography", "warp"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(REPO)).replace("\\", "/"): digest(path)
            for path in sorted(SOURCE.rglob("*")) if path.is_file()}


def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor()


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    sources = [HERE / "benchmark.cpp", HERE / "harris_detector.cpp",
               SOURCE / "stb_impl.cpp", *sorted((SOURCE / "v0").glob("*.cpp"))]
    command = ["g++", "-O2", "-std=c++17", *map(str, sources), "-o", str(EXE)]
    subprocess.run(command, check=True)
    subprocess.run([str(EXE), "--self-test"], check=True)
    return command


def hull_fraction(pairs, side, width, height):
    points = sorted(set(tuple(p[side:side + 2]) for p in pairs))
    if len(points) < 3:
        return None
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = abs(sum(p[0] * q[1] - p[1] * q[0]
                   for p, q in zip(hull, hull[1:] + hull[:1]))) / 2
    return area / ((width - 1) * (height - 1))


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    rows, records = [], {}
    for path in sorted(OUTPUT.glob("image[1-4]_*_*.json")):
        site, size, detector = path.stem.split("_")
        data = json.loads(path.read_text(encoding="utf-8"))
        key = (site, size)
        records.setdefault(key, {})[detector] = data
        row = {"site": site, "size": size, "detector": detector,
               "status": data["status"], "successful_runs": data["successful_runs"],
               "runs": data["runs"], "mean_ms": data["mean_ms"],
               "stddev_ms": data["stddev_ms"], "min_ms": data["min_ms"],
               "keypoints_a": data["keypoints"][0], "keypoints_b": data["keypoints"][1],
               "matches": data["matches"], "inliers": data["inliers"],
               "inlier_ratio": data["inliers"] / data["matches"] if data["matches"] else None,
               "inlier_error_px": data["inlier_error_px"], "error": data["error"],
               "panorama_width": data["panorama_size"][0],
               "panorama_height": data["panorama_size"][1]}
        for i, stage in enumerate(STAGES):
            row[stage + "_ms"] = data["stage_mean_ms"][i]
        for side, name in [(0, "a"), (2, "b")]:
            row["inlier_hull_fraction_" + name] = hull_fraction(
                data["inlier_pairs"], side, *data["input_" + name])
        rows.append(row)
    if not rows:
        raise RuntimeError("No results to summarize")
    with (OUTPUT / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    keys = sorted(records, key=lambda key: (SIZES.index(key[1]), key[0]))
    paired = [key for key in keys if set(records[key]) == {"dog", "harris"}]
    common = [key for key in paired if all(records[key][d]["status"] == "ok" for d in ("dog", "harris"))]
    metrics = {"cases": len(paired), "common_success_cases": len(common),
               "successes": {d: sum(records[k][d]["status"] == "ok" for k in paired) for d in ("dog", "harris")},
               "total_detection_ms": {d: sum(records[k][d]["stage_mean_ms"][1] for k in paired) for d in ("dog", "harris")},
               "total_e2e_common_ms": {d: sum(records[k][d]["mean_ms"] for k in common) for d in ("dog", "harris")}}
    metrics["detection_speedup_dog_over_harris"] = metrics["total_detection_ms"]["dog"] / metrics["total_detection_ms"]["harris"]
    metrics["e2e_speedup_dog_over_harris"] = metrics["total_e2e_common_ms"]["dog"] / metrics["total_e2e_common_ms"]["harris"]
    metrics["geometric_mean_e2e_speedup"] = math.exp(sum(math.log(records[k]["dog"]["mean_ms"] / records[k]["harris"]["mean_ms"]) for k in common) / len(common)) if common else None
    (OUTPUT / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    colors = {"dog": "#315fa6", "harris": "#c96528"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), layout="constrained")
    labels = [f"{k[0]}\n{k[1]}" for k in paired]
    x = np.arange(len(paired))
    for offset, detector in [(-0.19, "dog"), (0.19, "harris")]:
        values = [records[k][detector] for k in paired]
        stage_errors = [np.std([r["stage_ms"][1] for r in v["runs_data"]]) for v in values]
        axes[0].bar(x + offset, [v["stage_mean_ms"][1] for v in values], 0.38, label=detector.upper(), color=colors[detector], yerr=stage_errors, capsize=2)
        times = [v["mean_ms"] if v["status"] == "ok" else np.nan for v in values]
        time_errors = [v["stddev_ms"] if v["status"] == "ok" else 0 for v in values]
        axes[1].bar(x + offset, times, 0.38, label=detector.upper(), color=colors[detector], yerr=time_errors, capsize=2)
        for i, value in enumerate(times):
            if np.isnan(value): axes[1].text(i + offset, 0, "FAIL", ha="center", va="bottom", color=colors[detector], rotation=90)
        axes[2].bar(x + offset, [100 * v["inliers"] / v["matches"] if v["matches"] else 0 for v in values], 0.38, color=colors[detector])
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set(title="Stage 2: detection of both input images (lower is faster)", ylabel="Mean time (ms)")
    axes[1].set(title="Full pipeline, excluding disk I/O (failed cases have no complete time)", ylabel="Mean time (ms)")
    axes[2].set(title="RANSAC inlier ratio (not ground-truth accuracy)", ylabel="Inliers / matches (%)", ylim=(0, 100))
    axes[0].legend()
    run_counts = sorted(set(row["runs"] for row in rows))
    run_label = str(run_counts[0]) if len(run_counts) == 1 else str(run_counts)
    fig.suptitle(f"DoG vs Harris | same five Gaussian levels + BRIEF + matching + RANSAC + warp\n1 warm-up and {run_label} alternating timed runs per detector; CPU, g++ -O2; error bars = SD", fontsize=11)
    fig.savefig(OUTPUT / "benchmark.png", dpi=150)
    plt.close(fig)

    for size, label in zip(SIZES, ("720p", "1080p", "4k")):
        fig, axes = plt.subplots(4, 2, figsize=(14, 14))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.97, hspace=0.16, wspace=0.06)
        for i in range(4):
            for j, detector in enumerate(("dog", "harris")):
                path = OUTPUT / f"image{i + 1}_{size}_{detector}.png"
                record = records.get((f"image{i + 1}", size), {}).get(detector)
                if record and record["status"] == "ok" and path.exists():
                    with Image.open(path) as im:
                        im.thumbnail((1400, 900))
                        axes[i, j].imshow(im)
                else:
                    axes[i, j].text(0.5, 0.5, "Pipeline failed / no result", ha="center")
                axes[i, j].set_title(f"image{i + 1}, {size} | {detector.upper()}", fontsize=11, pad=8)
                axes[i, j].axis("off")
        fig.savefig(OUTPUT / f"panorama_comparison_{label}.png", dpi=130)
        plt.close(fig)

    def fmt(value, digits=1):
        return "FAIL" if value is None else f"{value:.{digits}f}"
    iteration_counts = sorted({record["shared_params"]["ransac_iterations"] for pair in records.values() for record in pair.values()})
    lines = ["# DoG vs Harris: isolated CPU experiment", "",
             "Original files under `src/` and both notebooks are unchanged. This experiment replaces only detection; all five Gaussian levels are retained for both methods.", "",
             "- Same image pairs as Midterm.ipynb: left/right for image1, image3, image4; left/mid for image2; three resolutions.",
             "- Same compiler and process, -O2 -std=c++17. One warm-up per method; five timed runs, alternating order within each case and alternating the first method between cases.",
             f"- Shared parameters: at most 4000 points/image; BRIEF 256 bits, 31x31; ratio 0.8; RANSAC iterations {iteration_counts}, seed 21, inlier threshold 3 px.",
             "- Harris: central differences on Gaussian[0] (sigma=1); Gaussian tensor integration sigma=1.5, radius=5; R=det(M)-0.04*trace(M)^2; threshold=1% of maximum valid response; 3x3 NMS; border=15; top-K.",
             "- Harris thresholds were fixed before looking at dataset outcomes; they are not equivalent to DoG's contrast threshold. Same point cap does not imply equal actual point counts.",
             "- Stage 2 includes detection of both images. Full times include local allocation/deallocation but exclude disk I/O and diagnostic coordinate copying. Failed pipelines have no full-pipeline mean; their completed stage times remain available.",
             "- Inlier count/ratio/error are consistency with each method's fitted homography, not independently measured accuracy. No ground-truth homographies are provided.", "",
             f"Successful cases: DoG {metrics['successes']['dog']}/{len(paired)}, Harris {metrics['successes']['harris']}/{len(paired)}.",
             f"Ratio of summed detection times (DoG/Harris): {metrics['detection_speedup_dog_over_harris']:.3f}x.",
             f"Ratio of summed complete times over {len(common)} common successful cases (DoG/Harris): {metrics['e2e_speedup_dog_over_harris']:.3f}x.", "",
             "| Case | Detection DoG/Harris ms | Full DoG/Harris ms | Points A/B DoG | Points A/B Harris | Inliers DoG/Harris | Inlier error DoG/Harris px |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key in paired:
        d, h = records[key]["dog"], records[key]["harris"]
        lines.append(f"| {' '.join(key)} | {fmt(d['stage_mean_ms'][1])} / {fmt(h['stage_mean_ms'][1])} | {fmt(d['mean_ms'])} / {fmt(h['mean_ms'])} | {d['keypoints'][0]}/{d['keypoints'][1]} | {h['keypoints'][0]}/{h['keypoints'][1]} | {d['inliers']} / {h['inliers']} | {fmt(d['inlier_error_px'], 2)} / {fmt(h['inlier_error_px'], 2)} |")
    lines.extend(["", "## Failures", ""])
    for key in paired:
        for detector in ("dog", "harris"):
            value = records[key][detector]
            if value["status"] != "ok": lines.append(f"- {' '.join(key)} {detector}: {value['error']}")
    metadata = json.loads((OUTPUT / "environment.json").read_text(encoding="utf-8"))
    lines.extend(["", "## Interpretation", "",
                  f"For these implementations, Harris detection takes {1 / metrics['detection_speedup_dog_over_harris']:.2f}x the DoG time (ratio of summed means across all cases).",
                  f"Across the {len(common)} common successful cases, summed full-pipeline time increases by {(1 / metrics['e2e_speedup_dog_over_harris'] - 1) * 100:.1f}% with Harris.",
                  "DoG reuses Gaussian levels computed in stage 1; Harris additionally computes gradients and smooths the second-moment matrix during stage 2. This is a stage-replacement experiment, not a claim that every Harris implementation is slower.",
                  "Different point counts also affect quadratic descriptor matching cost; faster or slower full-pipeline results cannot be attributed solely to detection.",
                  f"Completed panoramas: DoG {metrics['successes']['dog']}/{len(paired)}, Harris {metrics['successes']['harris']}/{len(paired)}. Writing a panorama does not establish correct alignment. Inspect the comparison sheets for remaining overlap/ghosting artifacts.",
                  "", "Figures: [timing and inlier ratios](benchmark.png), [720p](panorama_comparison_720p.png), [1080p](panorama_comparison_1080p.png), [4K](panorama_comparison_4k.png).", ""])
    lines.extend(["", "## Environment and validation", "", f"CPU: {metadata['cpu']}; OS: {metadata['platform']}; compiler: {metadata['compiler'].splitlines()[0]}.",
                  "", f"Source SHA-256 check after benchmark: {source_hashes() == metadata['source_sha256']}.",
                  "", "Harris self-test covers constant image, straight edge rejection, four square corners, translation covariance, and top-K budget. DoG reference parity is checked separately against the original executable output.",
                  "", "Sources: [Harris & Stephens (1988)](https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.pdf); [Lowe (2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).", ""])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--sites", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize()
        return
    command = build() if args.build else None
    if not EXE.is_file(): raise RuntimeError("Build missing; run with --build")
    previous = json.loads((OUTPUT / "environment.json").read_text(encoding="utf-8")) if (OUTPUT / "environment.json").is_file() else {}
    metadata = {"cpu": cpu_name(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
                "python": platform.python_version(), "compiler": subprocess.check_output(["g++", "--version"], text=True),
                "build_command": command or previous.get("build_command"), "source_sha256": source_hashes(),
                "notebook_sha256": {p.name: digest(p) for p in MIDTERM.glob("*.ipynb")},
                "experiment_sha256": {p.name: digest(p) for p in sorted(HERE.glob("*")) if p.is_file()},
                "runs": args.runs, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUTPUT / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for case, (site, size) in enumerate((site, size) for size in args.sizes for site in args.sites):
        partner = "mid" if site == 2 else "right"
        name = f"image{site}_{size}"
        print(f"START {name}", flush=True)
        invocation = [str(EXE), str(REPO / "Dataset" / f"image{site}_left_{size}.jpg"),
                      str(REPO / "Dataset" / f"image{site}_{partner}_{size}.jpg"),
                      str(OUTPUT / name), str(args.runs), "dog" if case % 2 == 0 else "harris"]
        result = subprocess.run(invocation, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (OUTPUT / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        if result.returncode: raise RuntimeError(f"Benchmark executable failed: {name}")
    if source_hashes() != metadata["source_sha256"]: raise RuntimeError("Original sources changed during benchmark")
    summarize()


if __name__ == "__main__":
    main()
