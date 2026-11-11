# Isolated Harris vs DoG benchmark

This directory adds an experimental Harris detector and a paired benchmark driver.
The existing `Midterm/src/` files and notebooks are not edited. The driver calls
the original preprocessing, DoG, BRIEF, matching, homography, and blending functions
directly; only the detector changes for the Harris variant.

From this directory, with Python (numpy, matplotlib, Pillow) and g++ available:

```sh
python run_benchmark.py --build --runs 5
```

On this Windows machine the complete Python environment is `E:\Anaconda\python.exe`.
The script also works on Linux/Colab when the repository, including this experiment,
is present with `Dataset/` at the repository root.

Outputs are in `Midterm/outputs/harris_vs_dog/`: per-case JSON and panorama PNGs,
`benchmark.csv`, `environment.json`, `summary.json`, charts, and `report.md`.
The binary is in `Midterm/build/harris_vs_dog/`.

Method: single-scale Harris with central image derivatives, Gaussian integration
(sigma 1.5), k=0.04, positive response threshold at 1% of maximum, 3x3 NMS, a
15-pixel border, and at most 4000 points. Both variants retain all five Gaussian
levels so the experiment measures replacing stage 2, not redesigning stage 1.
Actual point counts may differ. Parameters are fixed before dataset evaluation.

The same executable alternates detector order on every timed repeat; the script
alternates the first detector between cases. Each method gets one warm-up and
five timed repeats by default. Failed cases keep completed stage timings and are
excluded from complete-pipeline comparisons. Timings are CPU timings, not GPU.

The self-test validates flat/edge rejection, four synthetic corners, translation,
and the point limit. Inlier statistics are internal model consistency metrics;
they are not ground-truth panorama accuracy.

References: [Harris & Stephens](https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.pdf),
[Lowe](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).
