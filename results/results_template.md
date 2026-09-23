# Results Table

Copy this template into `results_table.md` and add one row per benchmark run. Keep the JSON result files; the table is the summary.

| Date | Policy | Checkpoint | Device | Runs | chunk_p50 | chunk_p95 | chunks/s | actions/s | Peak mem | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-23 | smolvla | lerobot/smolvla_base | cpu | 3 | 812.4ms | 901.2ms | 1.21 | 60.5 | 2140 MiB (rss) | first smoke run, values are placeholders |

Rules for rows:

- One row per JSON file in this directory. The numbers must match the file.
- Peak mem says `(cuda)` or `(rss)` so readers know what was measured.
- Never edit a published row. Rerun and add a new one.
