# First Numbers Checklist

The goal of the first session is one honest table row: SmolVLA on your machine, real weights, real timing. Nothing needs to be fast. It needs to be reproducible.

## 1. Environment

- [ ] Python 3.10+ installed (`python --version`)
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python bench.py --dry-run` runs cleanly and writes a JSON file to `results/`

## 2. Real weights

- [ ] `python bench.py --policy smolvla --runs 3 --warmup 1` downloads the checkpoint and completes
- [ ] Confirm the output row shows your actual device (`cpu`, `mps`, or `cuda`)
- [ ] Open the JSON in `results/` and confirm `dry_run` is `false`

## 3. The real benchmark

- [ ] `python bench.py --policy smolvla --runs 20 --warmup 3`
- [ ] Copy the terminal row into `results/results_table.md` (create it from the template)
- [ ] Commit the JSON result file

## 4. Sanity checks before publishing numbers

- [ ] chunk_p95 is not wildly larger than chunk_p50 (a large gap means thermal throttling or background load; rerun)
- [ ] Peak memory is stable across runs, not growing (growth means a leak in the loop)
- [ ] The hardware string in the JSON names the actual machine, not a guess

Done when `results_table.md` has one row you would defend in a job interview.
