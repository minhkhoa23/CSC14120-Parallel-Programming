"""Refresh notebook analysis from the measured 10000-iteration CLI outputs.

The original Colab commands remain intact. Only the method description, execution
outputs, and benchmark provenance change. Requires nbformat, IPython, matplotlib.
"""
import base64
import io
import json
import os
from pathlib import Path
import re

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

HERE = Path(__file__).resolve().parent
MIDTERM = HERE.parents[1]
PATH = MIDTERM / "Midterm.ipynb"
OUTPUT = MIDTERM / "outputs"


def main():
    original = json.loads(PATH.read_text(encoding="utf-8"))
    for cell in original["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            for key in ("outputId", "executionInfo"):
                cell.get("metadata", {}).pop(key, None)
        else:
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            source = source.replace("2000 lần lấy mẫu", "10000 lần lấy mẫu")
            if source.startswith("# Method"):
                marker = "Kết quả benchmark dưới đây"
                if marker not in source:
                    source += ("\n\nKết quả benchmark dưới đây được đo lại trên CPU Intel i7-11370H, "
                               "Windows, g++ 15.2.0 (`-O2 -std=c++17`), RANSAC 10000 lần, "
                               "một lượt khởi động và năm lượt đo cho mỗi trường hợp. "
                               "Các lệnh thực thi Colab được giữ nguyên.\n")
            cell["source"] = source.splitlines(keepends=True)

    selected = [4, 5, 22, 23, 25, 26, 27, 30, 32, 34, 36]
    analysis_cells = []
    for index in selected:
        source = original["cells"][index]["source"]
        source = "".join(source) if isinstance(source, list) else source
        assert not source.lstrip().startswith(("!", "%%")), (index, source)
        analysis_cells.append(source)
    os.environ["PANORAMA_DATASET"] = str(MIDTERM.parent / "Dataset")
    os.chdir(MIDTERM)
    print("Executing original Python analysis cells against the new outputs...", flush=True)
    shell = InteractiveShell.instance(config=Config({"HistoryManager": {"hist_file": ":memory:"}}))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from IPython.display import display

    # Capture plt.show without a GUI or a local kernel server.
    def show(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight")
            display({"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                     "text/plain": repr(figure)}, raw=True)
        plt.close("all")
    plt.show = show
    for count, (index, source) in enumerate(zip(selected, analysis_cells), 1):
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Notebook cell {index} failed: {captured.stderr or captured.stdout}")
        outputs = []
        if captured.stdout:
            outputs.append({"output_type": "stream", "name": "stdout", "text": captured.stdout.splitlines(keepends=True)})
        if captured.stderr:
            outputs.append({"output_type": "stream", "name": "stderr", "text": captured.stderr.splitlines(keepends=True)})
        for rich in captured.outputs:
            data = {mime: base64.b64encode(value).decode("ascii") if isinstance(value, (bytes, bytearray)) else value
                    for mime, value in rich.data.items()}
            outputs.append({"output_type": "display_data", "data": data, "metadata": rich.metadata or {}})
        original["cells"][index]["outputs"] = outputs
        original["cells"][index]["execution_count"] = count
        print(f"Updated analysis cell {index}: {len(outputs)} output(s)", flush=True)

    for cell in original["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        site = re.search(r'name="image([1-4])_\$\{size\}"', source)
        if not site or "./stitch" not in source:
            continue
        chunks = []
        for size in ("1280x720", "1920x1080", "3840x2160"):
            name = f"image{site.group(1)}_{size}"
            chunks.append(name + "\n" + (OUTPUT / f"{name}.log").read_text(encoding="utf-8"))
        cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": "".join(chunks).splitlines(keepends=True)}]

    original["metadata"]["v0_benchmark"] = {
        "ransac_iterations": 10000, "warmup_runs": 1, "timed_runs": 5,
        "platform": "Windows CPU, Intel i7-11370H, g++ 15.2.0 -O2",
        "results": "outputs/", "comparison": "outputs/ransac_iterations/",
    }
    normalized = nbformat.reads(json.dumps(original, ensure_ascii=False), as_version=4)
    nbformat.validate(normalized)
    assert not any(o.get("output_type") == "error" for c in original["cells"] if c["cell_type"] == "code" for o in c["outputs"])
    assert sum(len(c["outputs"]) for c in original["cells"] if c["cell_type"] == "code") > 10
    PATH.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Updated and validated Midterm.ipynb; Colab code cells preserved.", flush=True)


if __name__ == "__main__":
    main()
