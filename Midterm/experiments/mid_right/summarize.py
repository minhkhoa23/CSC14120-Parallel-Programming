"""Validate measured mid/right results, write a report, and package Colab inputs."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

import nbformat
from PIL import Image

MIDTERM = Path(__file__).resolve().parents[2]
REPO = MIDTERM.parent
OUTPUT = MIDTERM / "outputs" / "mid_right"


def text(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair"] == ["mid", "right"] and len(manifest["cases"]) == 12
    assert all(hashlib.sha256((MIDTERM / "src" / p).read_bytes()).hexdigest() == sha
               for p, sha in manifest["source_sha256"].items())
    rows, inputs = [], set()
    for case in manifest["cases"]:
        assert case["input_a"] == f"image{case['group']}_mid_{case['size']}.jpg"
        assert case["input_b"] == f"image{case['group']}_right_{case['size']}.jpg"
        for name, sha in zip((case["input_a"], case["input_b"]), case["input_sha256"]):
            assert "_left_" not in name
            assert hashlib.sha256((REPO / "Dataset" / name).read_bytes()).hexdigest() == sha
            inputs.add(name)
        if case["status"] != "ok":
            rows.append((case, None))
            continue
        data = json.loads((OUTPUT / f"{case['case']}.json").read_text(encoding="utf-8"))
        assert data["params"]["ransac_iterations"] == 10000 and data["runs"] == 5
        assert len(data["runs_stage_ms"]) == len(data["runs_e2e_ms"]) == 5
        assert all(len(row) == 6 and all(v >= 0 for v in row) for row in data["runs_stage_ms"])
        assert abs(statistics.mean(data["runs_e2e_ms"]) - data["mean_ms"]) < 1e-6
        assert abs(statistics.pstdev(data["runs_e2e_ms"]) - data["stddev_ms"]) < 1e-6
        assert 4 <= data["inliers"] <= data["matches"] <= data["keypoints"][0] <= 4000
        assert data["inlier_error_px"] <= 3
        dims = list(map(int, case["size"].split("x")))
        assert data["input_a"] == data["input_b"] == dims
        H = data["H_B_to_A"]
        zs = [H[6] * x + H[7] * y + H[8] for x in (0, dims[0] - 1) for y in (0, dims[1] - 1)]
        assert min(zs) > 0 or max(zs) < 0
        with Image.open(OUTPUT / f"{case['case']}.png") as im: im.verify()
        rows.append((case, data))
    assert len(inputs) == 24
    raw = json.loads((MIDTERM / "Group21_Midterm.ipynb").read_text(encoding="utf-8"))
    original = json.loads((OUTPUT / "Group21_before_mid_right.ipynb").read_text(encoding="utf-8"))
    assert all(text(raw["cells"][i]) == text(original["cells"][i]) for i in range(4))
    nbformat.validate(nbformat.reads(json.dumps(raw, ensure_ascii=False), as_version=4))
    assert not any(o.get("output_type") == "error" for c in raw["cells"] if c["cell_type"] == "code" for o in c["outputs"])
    # Use the notebook's final, tightly cropped render for the exported previews.
    for size in ("1280x720", "1920x1080", "3840x2160"):
        entry = next(c for c in raw["cells"] if text(c).strip() == f'show_panoramas("{size}")')
        rich = next(o for o in entry["outputs"] if "image/png" in o.get("data", {}))
        encoded = rich["data"]["image/png"]
        (OUTPUT / f"panoramas_{size}.png").write_bytes(base64.b64decode("".join(encoded) if isinstance(encoded, list) else encoded))
    successes = sum(data is not None for _, data in rows)
    summary = {"input_a": "mid", "input_b": "right", "attempted_cases": 12,
               "successful_cases": successes, "runs": 5, "warmup_runs": 1, "ransac_iterations": 10000,
               "failed_cases": [case["case"] for case, data in rows if data is None],
               "mean_ms_by_resolution": {size: statistics.mean(data["mean_ms"] for case, data in rows if case["size"] == size and data)
                                          for size in ("1280x720", "1920x1080", "3840x2160")}}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Kết quả ghép mid–right trên 12 trường hợp", "",
             "Ảnh A là mid, ảnh B là right cho cả bốn bộ ảnh và ba độ phân giải. Mã nguồn C++ được giữ nguyên: V0 CPU, DoG, BRIEF 256 bit, ratio 0.8, tối đa 4000 keypoint/ảnh, RANSAC 10000 lần, seed 21 và ngưỡng inlier 3 pixel.", "",
             f"Mỗi trường hợp chạy một lần khởi động và năm lần đo. Thời gian không gồm đọc/ghi file ảnh. Xuất được panorama: **{successes}/12**.",
             f"Môi trường: {manifest['platform']}; {manifest['compiler'].splitlines()[0]}.", "",
             "| Trường hợp | Trung bình (ms) | Độ lệch chuẩn (ms) | Keypoint A/B | Matches | Inlier | Tỷ lệ inlier | Sai số inlier (px) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, data in rows:
        if data:
            lines.append(f"| {case['case']} | {data['mean_ms']:.2f} | {data['stddev_ms']:.2f} | {data['keypoints'][0]}/{data['keypoints'][1]} | {data['matches']} | {data['inliers']} | {data['inliers']/data['matches']:.1%} | {data['inlier_error_px']:.2f} |")
        else:
            lines.append(f"| {case['case']} | ERROR | | | | | | |")
    lines.extend(["", "Inlier và sai số phản ánh độ đồng thuận với homography được ước lượng, chưa phải độ chính xác độc lập so với ground truth. Cần xem vùng chồng lấp để kiểm tra bóng đôi hoặc sai lệch do vật thể ở các độ sâu khác nhau.", "",
                  "Ảnh xem trước được thu nhỏ; mỗi file imageN_SIZE.png giữ nguyên canvas panorama do chương trình tạo từ ảnh nguồn tương ứng.", "",
                  "[CSV đầy đủ](timing.csv) · [Phân bố thời gian](stage_times.png) · [720p](panoramas_1280x720.png) · [1080p](panoramas_1920x1080.png) · [4K](panoramas_3840x2160.png)", ""])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    colab = OUTPUT / "colab"
    colab.mkdir(exist_ok=True)
    shutil.copy2(MIDTERM / "Group21_Midterm.ipynb", colab / "Group21_Midterm.ipynb")
    archive_path = colab / "src.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted((MIDTERM / "src").rglob("*")):
            if path.is_file(): archive.write(path, "src/" + path.relative_to(MIDTERM / "src").as_posix())
        for name in sorted(inputs): archive.write(REPO / "Dataset" / name, "Dataset/" + name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any("_left_" in name for name in names)
        assert len([name for name in names if name.startswith("Dataset/")]) == 24
        assert "iterations = 10000" in archive.read("src/common.h").decode("utf-8")
    validation = {"cases_attempted": 12, "successful_cases": successes, "input_images": 24,
                  "pair_roles_verified": True, "no_left_images_used": True,
                  "source_cpp_unchanged": True, "source_and_input_hashes_verified": True,
                  "runs_and_ransac_metadata_verified": True, "timing_statistics_verified": True,
                  "notebook_valid": True, "original_report_prefix_preserved": True,
                  "colab_zip_crc_valid": True, "colab_zip_has_24_mid_right_images": True}
    (OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Validated notebook, metrics, images, input roles, source hashes and Colab package.")


if __name__ == "__main__": main()
