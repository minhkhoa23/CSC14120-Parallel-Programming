# V0 benchmark: RANSAC 2000 vs 10000

```powershell
& 'E:\Anaconda\python.exe' 'D:\CSC14120-Parallel-Programming\Midterm\experiments\ransac_iterations\run_benchmark.py' --build --runs 5
```

The script builds the real updated application at 10000 iterations, and an isolated
copy of the same source with only the default changed back to 2000. Both use
`g++ -O2 -std=c++17 -x c++`. No Harris detector or extra homography filtering is used.

It runs the same 12 cases as the notebook: image1/3/4 left-right, image2 left-mid,
at 720p, 1080p, and 4K. Each executable performs one warm-up followed by five timed
runs. Method order alternates between cases. Disk I/O is excluded by the original
application timer. A warm-up failure produces no full-pipeline timing or metrics.

Updated 10000-iteration results are in `Midterm/outputs/`, compatible with the
existing notebook analysis cells. Fresh 2000-iteration results, the comparison CSV,
figures, environment, and report are in `Midterm/outputs/ransac_iterations/`.

The old baseline copy and binary are isolated in `Midterm/build/ransac_iterations/`.
Summarize saved results without rerunning measurements with `--summarize-only`.
