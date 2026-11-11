# Run Group21 notebook on mid/right pairs

The notebook now runs all 4 image groups at 720p, 1080p and 4K, using mid as A and
right as B. Its setup and CPU build work on Windows and Colab. Existing C++ source
is not changed; RANSAC remains at 10000 iterations.

```powershell
& 'E:\Anaconda\python.exe' 'D:\CSC14120-Parallel-Programming\Midterm\experiments\mid_right\run_notebook.py'
& 'E:\Anaconda\python.exe' 'D:\CSC14120-Parallel-Programming\Midterm\experiments\mid_right\summarize.py'
```

The first command executes notebook code cells and saves their real outputs.
If all cases were already measured but only notebook rendering needs to be
refreshed, use `--reuse-measurements`; it checks source hashes and reads existing
measurements instead of running the 12 pipelines again.

Results, logs, the input manifest, CSV, figures and report are under
`Midterm/outputs/mid_right/`. The original notebook snapshot is stored there as
`Group21_before_mid_right.ipynb`. A self-contained Colab notebook and `src.zip`
with the source plus only the 24 mid/right images are in the `colab/` subfolder.
The original Dataset directory is not pruned.
