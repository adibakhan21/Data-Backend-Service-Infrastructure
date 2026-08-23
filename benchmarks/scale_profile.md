# Scale profile

Measured, not extrapolated. Peak resident memory and wall clock for generating
a dataset and building the in-memory snapshot over it.

| Rows | Generate | Snapshot build | Peak RSS | Bytes/row | Anomalies |
|---|---|---|---|---|---|
| 200,000 | 0.12s | 0.19s | 228 MB | 1197 B | 1,174 |
| 1,000,000 | 0.54s | 0.99s | 628 MB | 659 B | 5,982 |
| 2,000,000 | 1.09s | 2.08s | 1,031 MB | 541 B | 11,957 |
| 5,000,000 | 2.84s | 6.68s | 1,872 MB | 393 B | 29,874 |

## What this establishes

Snapshot build time grows close to linearly, so time is not what breaks first.
**Memory is.** At roughly the bytes-per-row shown above, a single worker holding
10M rows needs several GB, and each additional uvicorn worker holds its own copy -
so memory, not CPU, is what caps worker count.

That is the measured basis for two decisions in `docs/system_design.md`: pushing
filtering into SQL (§7.1), and moving the snapshot out of process before scaling
horizontally (§7.3). Both were reasoned about before this profile existed; the
profile is what turns them from plausible into supported.
