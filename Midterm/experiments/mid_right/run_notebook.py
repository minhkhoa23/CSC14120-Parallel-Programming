"""Prepare and execute Group21_Midterm.ipynb for all 12 mid/right pairs.

Runs the notebook's Python cells headlessly with IPython and captures their real
outputs. The CPU C++ source and all original Dataset files are left untouched.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid
import zipfile

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

HERE = Path(__file__).resolve().parent
MIDTERM = HERE.parents[1]
REPO = MIDTERM.parent
NOTEBOOK = MIDTERM / "Group21_Midterm.ipynb"
OUTPUT = MIDTERM / "outputs" / "mid_right"


def source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def cell(kind, text):
    result = {"cell_type": kind, "id": uuid.uuid4().hex[:8],
              "metadata": {"tags": ["mid-right-benchmark"]},
              "source": text.strip().splitlines(keepends=True)}
    if kind == "code": result.update(execution_count=None, outputs=[])
    return result


def prepare(raw):
    if raw.get("metadata", {}).get("mid_right_benchmark"):
        return raw
    headings = {source(c).strip(): i for i, c in enumerate(raw["cells"]) if c["cell_type"] == "markdown"}
    seq = headings["# Sequential version (V0)"]
    results = headings["# V0 results and profiling"]
    dataset = headings["# Dataset"]
    gpu = headings["# GPU plan"]
    cells = raw["cells"][:seq + 1]
    cells.append(cell("markdown", """
## Chuẩn bị và biên dịch

Benchmark này dùng ảnh **mid làm A**, ảnh **right làm B**, cho cả bốn bộ ảnh ở 720p, 1080p và 4K. Mỗi trường hợp chạy khởi động một lần rồi đo năm lần, với RANSAC 10000 lần. Kết quả lưu riêng tại `outputs/mid_right/`.

Trên Colab, upload gói `src.zip` đi kèm trước khi chạy. Cell chuẩn bị chỉ giải nén khi chưa tìm thấy mã nguồn, để tránh ghi đè mã nguồn đang có. V0 được biên dịch bằng g++ vì file `main.cu` hiện chứa mã C++ chạy trên CPU.
"""))
    cells.append(cell("code", r'''
from pathlib import Path
import os
import zipfile

start = Path.cwd()
candidates = [start, start / "Midterm", Path("/content/CSC14120-Parallel-Programming/Midterm")]
base = next((p for p in candidates if (p / "src/common.h").is_file()), None)
if base is None:
    archive = start / "src.zip"
    assert archive.is_file(), "Upload src.zip chứa src/ và Dataset/ trước khi chạy"
    with zipfile.ZipFile(archive) as package:
        for name in package.namelist():
            target = (start / name.replace("\\", "/")).resolve()
            assert target.is_relative_to(start.resolve()), "Đường dẫn ZIP không hợp lệ"
        package.extractall(start)
    base = next((p for p in candidates if (p / "src/common.h").is_file()), None)
assert base is not None, "Chưa tìm thấy src/common.h"
base = base.resolve()
os.chdir(base)
data_candidates = ([Path(os.environ["PANORAMA_DATASET"])] if os.environ.get("PANORAMA_DATASET") else [])
data_candidates += [base / "Dataset", base.parent / "Dataset", Path("/content/Dataset")]
data = next((p.resolve() for p in data_candidates if p.is_dir()), None)
assert data is not None, "Chưa tìm thấy Dataset"
os.environ["PANORAMA_DATASET"] = str(data)
out = base / "outputs" / "mid_right"
build = base / "build" / "mid_right"
out.mkdir(parents=True, exist_ok=True)
build.mkdir(parents=True, exist_ok=True)
sizes = ("1280x720", "1920x1080", "3840x2160")
runs = 5
for group in range(1, 5):
    for size in sizes:
        for role in ("mid", "right"):
            assert (data / f"image{group}_{role}_{size}.jpg").is_file()
print("Mã nguồn:", base / "src")
print("Dataset:", data)
print("Đã kiểm tra đủ 24 ảnh mid/right cho 12 trường hợp.")
'''))
    cells.append(cell("code", '''
import hashlib
import json
import platform
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

compiler = shutil.which("g++")
assert compiler, "Cần cài g++ và thêm vào PATH"
compiler_version = subprocess.check_output([compiler, "--version"], text=True)
print(compiler_version.splitlines()[0])
'''))
    cells.append(cell("code", '''
cpp_sources = [base / "src/stb_impl.cpp", *sorted((base / "src/v0").glob("*.cpp"))]
objects = []
for file in cpp_sources:
    name = ("v0_" if file.parent.name == "v0" else "") + file.stem
    target = build / f"{name}.o"
    subprocess.run([compiler, "-O2", "-std=c++17", "-c", str(file), "-o", str(target)], check=True)
    objects.append(target)
print("Đã biên dịch", len(objects), "module C++.")
'''))
    cells.append(cell("code", '''
main_object = build / "main.o"
subprocess.run([compiler, "-O2", "-std=c++17", "-x", "c++", "-c",
                str(base / "src/main.cu"), "-o", str(main_object)], check=True)
exe = build / ("stitch.exe" if os.name == "nt" else "stitch")
subprocess.run([compiler, str(main_object), *map(str, objects), "-o", str(exe)], check=True)
print("Executable:", exe)
'''))
    cells.append(cell("code", '''
tests_source = base / "src/tests.cpp"
if tests_source.is_file():
    test_exe = build / ("tests.exe" if os.name == "nt" else "tests")
    v0_objects = [p for p in objects if p.name.startswith("v0_")]
    subprocess.run([compiler, "-O2", "-std=c++17", str(tests_source),
                    *map(str, v0_objects), "-o", str(test_exe)], check=True)
    print("Đã biên dịch chương trình kiểm thử:", test_exe)
else:
    print("Chưa có src/tests.cpp; bỏ qua phần biên dịch kiểm thử.")
'''))
    cells.append(raw["cells"][results])
    cells.append(cell("markdown", """
## Chạy 12 cặp mid–right

Ảnh A luôn là `imageN_mid_SIZE.jpg`; ảnh B luôn là `imageN_right_SIZE.jpg`. Thời gian do chương trình C++ đo không gồm đọc/ghi ảnh. File `.status` và `.log` ghi nhận riêng các trường hợp không xuất được panorama; các trường hợp đó vẫn được giữ trong danh sách kết quả.
"""))
    cells.append(cell("code", '''
records = []
source_sha256 = {p.relative_to(base / "src").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((base / "src").rglob("*")) if p.is_file()}

def run_group(group):
    for size in sizes:
        name = f"image{group}_{size}"
        image_a = data / f"image{group}_mid_{size}.jpg"
        image_b = data / f"image{group}_right_{size}.jpg"
        command = [str(exe), str(image_a), str(image_b), str(out / f"{name}.png"),
                   "--version", "0", "--runs", str(runs), "--metrics", str(out / f"{name}.json")]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        status = "ok" if completed.returncode == 0 else "error"
        (out / f"{name}.log").write_text(completed.stdout, encoding="utf-8")
        (out / f"{name}.status").write_text(status, encoding="utf-8")
        record = {"case": name, "group": group, "size": size, "status": status,
                  "input_a": image_a.name, "input_b": image_b.name,
                  "input_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in (image_a, image_b)]}
        records.append(record)
        if status == "ok":
            result = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
            assert result["params"]["ransac_iterations"] == 10000
            print(f"{name}: OK | {result['mean_ms']:.2f} ms | {result['matches']} matches | {result['inliers']} inlier", flush=True)
        else:
            print(f"{name}: ERROR | {completed.stdout.strip()}", flush=True)
        manifest = {"pair": ["mid", "right"], "platform": platform.platform(),
                    "compiler": compiler_version, "runs": runs, "warmup_runs": 1,
                    "ransac_iterations": 10000, "source_sha256": source_sha256, "cases": records}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
'''))
    for group in range(1, 5):
        cells.append(cell("markdown", f"### Bộ ảnh {group}: mid–right"))
        cells.append(cell("code", f"run_group({group})"))
    cells.append(cell("markdown", "## Bảng kết quả và thời gian"))
    cells.append(cell("code", '''
assert len(records) == 12
rows, failures = [], []
stage_names = ["Preprocessing", "Feature Detection", "Description", "Matching", "Homography", "Warp and Blending"]
for record in records:
    name = record["case"]
    if record["status"] != "ok":
        failures.append({"Bộ ảnh": record["group"], "Kích thước": record["size"],
                         "Lỗi": (out / f"{name}.log").read_text(encoding="utf-8").strip()})
        continue
    result = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    row = {"Trường hợp": name, "Bộ ảnh": record["group"], "Kích thước": record["size"],
           "Ảnh A": record["input_a"], "Ảnh B": record["input_b"],
           "Trung bình (ms)": result["mean_ms"], "Nhỏ nhất (ms)": result["min_ms"],
           "Độ lệch chuẩn (ms)": result["stddev_ms"],
           "Keypoint A": result["keypoints"][0], "Keypoint B": result["keypoints"][1],
           "Matches": result["matches"], "Inlier": result["inliers"],
           "Tỷ lệ inlier": result["inliers"] / result["matches"] if result["matches"] else 0,
           "Sai số inlier (px)": result["inlier_error_px"]}
    for i, value in enumerate(result["stage_mean_ms"], 1): row[f"Bước {i} (ms)"] = value
    rows.append(row)
timing = pd.DataFrame(rows)
failure_table = pd.DataFrame(failures, columns=["Bộ ảnh", "Kích thước", "Lỗi"])
print(f"Xuất được panorama: {len(rows)}/12 trường hợp.")
display(failure_table)
if not timing.empty:
    display(timing[["Bộ ảnh", "Kích thước", "Trung bình (ms)", "Độ lệch chuẩn (ms)", "Matches", "Inlier", "Sai số inlier (px)"]].round(2))
    display(timing[["Bộ ảnh", "Kích thước", *[f"Bước {i} (ms)" for i in range(1, 7)]]].round(2))
timing.to_csv(out / "timing.csv", index=False, encoding="utf-8-sig")
failure_table.to_csv(out / "failures.csv", index=False, encoding="utf-8-sig")
'''))
    cells.append(cell("code", '''
if not timing.empty:
    stage_times = timing.set_index("Trường hợp")[[f"Bước {i} (ms)" for i in range(1, 7)]]
    stage_times.columns = stage_names
    stage_times.plot.bar(stacked=True, figsize=(13, 5))
    plt.ylabel("Thời gian (ms)")
    plt.title("V0 DoG + BRIEF, RANSAC 10000 — cặp mid/right")
    plt.tight_layout()
    plt.savefig(out / "stage_times.png", dpi=150)
    plt.show()
'''))
    cells.append(cell("markdown", """
## Panorama của tất cả trường hợp

Các hình xem trước dưới đây được thu nhỏ để hiển thị. Ảnh PNG đầy đủ của mỗi trường hợp được lưu trong `outputs/mid_right/`. Việc xuất được ảnh và số inlier cao chưa đủ để khẳng định căn chỉnh hoàn toàn chính xác; cần xem thêm vùng chồng lấp và hiện tượng bóng đôi.
"""))
    cells.append(cell("code", '''
def show_panoramas(size):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, hspace=0.17, wspace=0.06)
    for group, axis in enumerate(axes.ravel(), 1):
        name = f"image{group}_{size}"
        status = (out / f"{name}.status").read_text(encoding="utf-8").strip()
        if status == "ok":
            with Image.open(out / f"{name}.png") as image:
                image.thumbnail((1600, 1000))
                axis.imshow(image)
        else:
            axis.text(0.5, 0.5, "Không xuất được panorama", ha="center")
        axis.set_title(f"Bộ {group}, {size} — mid/right")
        axis.axis("off")
    fig.savefig(out / f"panoramas_{size}.png", dpi=130, bbox_inches="tight")
    plt.show()
'''))
    for size in ("1280x720", "1920x1080", "3840x2160"):
        cells.append(cell("markdown", f"### Kết quả {size}"))
        cells.append(cell("code", f'show_panoramas("{size}")'))
    cells.append(raw["cells"][dataset])
    cells.append(cell("markdown", """
Benchmark sử dụng **24 ảnh**, tương ứng bốn cặp `mid/right` ở ba độ phân giải, tạo thành 12 trường hợp thử:

| Bộ ảnh | Ảnh A | Ảnh B | Các kích thước |
|---|---|---|---|
| 1 | `image1_mid_SIZE.jpg` | `image1_right_SIZE.jpg` | 1280×720, 1920×1080, 3840×2160 |
| 2 | `image2_mid_SIZE.jpg` | `image2_right_SIZE.jpg` | 1280×720, 1920×1080, 3840×2160 |
| 3 | `image3_mid_SIZE.jpg` | `image3_right_SIZE.jpg` | 1280×720, 1920×1080, 3840×2160 |
| 4 | `image4_mid_SIZE.jpg` | `image4_right_SIZE.jpg` | 1280×720, 1920×1080, 3840×2160 |

Trong đó `SIZE` lần lượt là `1280x720`, `1920x1080`, `3840x2160`. Ảnh mid được chọn làm hệ tọa độ tham chiếu A. Danh sách tệp đầu vào, mã SHA-256 và trạng thái từng trường hợp nằm trong `outputs/mid_right/manifest.json`.
"""))
    cells.extend(raw["cells"][gpu:])
    raw["cells"] = cells
    raw["metadata"]["mid_right_benchmark"] = {"input_a": "mid", "input_b": "right", "runs": 5,
                                               "ransac_iterations": 10000, "output_directory": "outputs/mid_right"}
    return raw


class Tee:
    def __init__(self, captured, live): self.captured, self.live = captured, live
    def write(self, text):
        self.captured.write(text)
        self.live.write(text)
        self.live.flush()
        return len(text)
    def flush(self): self.live.flush()
    def isatty(self): return False


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    backup = OUTPUT / "Group21_before_mid_right.ipynb"
    if not backup.exists(): shutil.copy2(NOTEBOOK, backup)
    raw = prepare(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    raw["nbformat"] = 4
    raw["nbformat_minor"] = max(5, raw.get("nbformat_minor", 0))
    for entry in raw["cells"]: entry.setdefault("id", uuid.uuid4().hex[:8])
    nbformat.validate(nbformat.reads(json.dumps(raw, ensure_ascii=False), as_version=4))
    reuse = "--reuse-measurements" in sys.argv
    saved_manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8")) if reuse else None
    if reuse:
        assert len(saved_manifest["cases"]) == 12
        assert all(hashlib.sha256((MIDTERM / "src" / name).read_bytes()).hexdigest() == value
                   for name, value in saved_manifest["source_sha256"].items())
    shell = InteractiveShell.instance(config=Config({"HistoryManager": {"hist_file": ":memory:"}}))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from IPython.display import display

    def show(*args, **kwargs):
        for number in plt.get_fignums():
            fig = plt.figure(number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            display({"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                     "text/plain": repr(fig)}, raw=True)
        plt.close("all")
    plt.show = show
    os.chdir(MIDTERM)
    if reuse: shell.user_ns["_saved_mid_right_cases"] = saved_manifest["cases"]
    count = 0
    for index, entry in enumerate(raw["cells"]):
        if entry["cell_type"] != "code": continue
        count += 1
        print(f"EXECUTE cell {index}", flush=True)
        entry["outputs"] = []
        entry["execution_count"] = None
        code = source(entry)
        group_call = re.fullmatch(r"run_group\(([1-4])\)", code.strip())
        if reuse and group_call:
            group = int(group_call.group(1))
            code = f'''\
for record in _saved_mid_right_cases:
    if record["group"] != {group}: continue
    records.append(record)
    name = record["case"]
    if record["status"] == "ok":
        result = json.loads((out / f"{{name}}.json").read_text(encoding="utf-8"))
        assert result["runs"] == runs and result["params"]["ransac_iterations"] == 10000
        print(f"{{name}}: kết quả đã đo | {{result['mean_ms']:.2f}} ms | {{result['matches']}} matches | {{result['inliers']}} inlier")
    else:
        print(name, "ERROR", (out / f"{{name}}.log").read_text(encoding="utf-8").strip())
'''
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            sys.stdout = Tee(sys.stdout, sys.__stdout__)
            sys.stderr = Tee(sys.stderr, sys.__stderr__)
            result = shell.run_cell(code, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Notebook cell {index} failed: {result.error_before_exec or result.error_in_exec}")
        if captured.stdout: entry["outputs"].append({"output_type": "stream", "name": "stdout", "text": captured.stdout.splitlines(keepends=True)})
        if captured.stderr: entry["outputs"].append({"output_type": "stream", "name": "stderr", "text": captured.stderr.splitlines(keepends=True)})
        for rich in captured.outputs:
            data = {mime: base64.b64encode(value).decode("ascii") if isinstance(value, (bytes, bytearray)) else value
                    for mime, value in rich.data.items()}
            entry["outputs"].append({"output_type": "display_data", "data": data, "metadata": rich.metadata or {}})
        entry["execution_count"] = count
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 12
    assert all("_mid_" in c["input_a"] and "_right_" in c["input_b"] for c in manifest["cases"])
    assert all(hashlib.sha256((MIDTERM / "src" / name).read_bytes()).hexdigest() == value for name, value in manifest["source_sha256"].items())
    raw["metadata"]["mid_right_benchmark"].update(platform=manifest["platform"], compiler=manifest["compiler"].splitlines()[0])
    nbformat.validate(nbformat.reads(json.dumps(raw, ensure_ascii=False), as_version=4))
    NOTEBOOK.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved executed Group21_Midterm.ipynb; all 12 mid/right cases attempted.", flush=True)


if __name__ == "__main__": main()
