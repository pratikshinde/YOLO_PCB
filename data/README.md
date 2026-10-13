# data/

Real captures only. No synthetic images, no downloaded datasets.

## Current: clean GOOD / FAULT collection (since 2026-10-06)

Written only by `python scripts/collect.py`. Guide: [docs/clean_data_collection.md](../docs/clean_data_collection.md).

| Folder | Content |
|---|---|
| `collection/originals/<good\|fault>/<board_id>/<session_id>/` | `session.json` (session + every image record), `images/<image_id>.png` (originals: saved first, write-once, SHA-256), `pcb/<image_id>_pcb.png` (canonical PCB image: board only, 116 x 83 mm at 20 px/mm) |
| `collection/good/<GOOD_###>/<session_id>/` | KEEP of a GOOD capture: the PCB image only, a verified copy with no background and no watermark. The first 6 images, kept before 2026-10-06 15:30, also have their full frame here |
| `collection/fault/<FAULT_###>/<session_id>/` | KEEP of a FAULT capture: the same |
| `collection/quarantine/<good\|fault>/<board_id>/<session_id>/` | REJECT: verified copies of the full frame and the PCB image (evidence), reason in `session.json` |

**Rules:**
- **Types.** Only GOOD and FAULT are collected now. TEST comes later, and there is no `test/` folder. FAULT boards never enter `good/` or the GOOD reference-bank candidates.
- **Board IDs.** One ID per physical board, for life: `GOOD_###` or `FAULT_###`, required for every session. Images stay grouped by board and session.
- **Saving.** Every capture is stored before any decision. Nothing is deleted: originals stay in `originals/`, and KEEP / REJECT only copy.
- **RECAPTURE.** A capture whose check says RECAPTURE (e.g. watermark over the PCB) cannot be kept.
- **Checking.** `python scripts/collection_summary.py` gives counts and an audit; `python scripts/pcb.py validate` covers the store too.

## Archived (not used)

Every image and record collected before 2026-10-06 is in `../archive/2026-10-06_pre_collection_reset/`:
- Experiment 0, Experiment 1B acquisition / curated / quarantine, station captures, live captures;
- inspection results, the old board registry, derived caches;
- the watermarked golden reference.

`manifest.csv` lists every file with its SHA-256. Do not copy anything back into `data/` as dataset or reference data.

## Other folders

| Folder | Content |
|---|---|
| `metadata/` | Production-track metadata templates (`boards.csv`, `sessions.csv`, `captures.csv`, `defects.csv`), header only |
| `raw/good/`, `raw/defective/`, `raw/calibration/` | Production-representative captures (later, production camera); empty |
| `annotations/` | Labels (later); empty |
| `processed/` | Derived data only, safe to delete (e.g. `processed/collection/reference_check/` verification sheets) |
| `captures/`, `products/` | Created by the inspection station (`scripts/station.py`) when it is used; not part of this collection |
