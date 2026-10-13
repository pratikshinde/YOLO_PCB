# Experiment 1B: Controlled Capture / Repeatability

| | |
|---|---|
| Status | **Analysis tooling ready and tested. No controlled photos captured yet, so there are NO Experiment 1B results.** |
| Date | 2026-10-02 |
| Baseline | [Experiment 1](../experiment_001/README.md): handheld phone, room light |
| Pipeline / parameters | Exactly the Experiment 1 pipeline (`src/pcb_inspection/comparison/pipeline.py`) and config ([`../experiment_001/config.yaml`](../experiment_001/config.yaml)); otherwise the comparison with the baseline would not be fair |
| Command | `python scripts/run_experiment_001b.py --sessions <controlled session ID>` |

## Objective
Does **controlled** image capture reduce false differences enough for reference-based inspection to become viable as a baseline? And can it separate normal variation from a **verified** real defect?

No training of any kind, no synthetic defects, no labels inferred from images.

## 1. Capture setup (mobile phone, prototype)
| Item | Requirement |
|---|---|
| Mount | Phone fixed on a stand or tripod directly above the board; nothing touches the phone between shots |
| Angle | Camera as perpendicular to the board as practical (screen parallel to the board) |
| Distance | Measure lens-to-board in mm once; it does not change during the session |
| Background | Plain, matte, non-reflective, white or light grey; an L-shaped tape corner guide |
| Lighting | Diffuse: no lamp directly above the board, no window behind or beside it. E.g. ceiling light through a white sheet or paper, or a simple light tent. Same light for the whole session. |
| Phone settings | Highest resolution, 1× main camera, JPEG, no flash / HDR / filters. **Pro / manual mode if available:** fixed focus on the board, fixed ISO and shutter, fixed white balance. Otherwise AE/AF lock (long press on the board). |
| Shutter | 2-second self-timer or a remote / volume-button shutter, so tapping does not shake the phone |
| Orientation | Same as the reference photos: board portrait, switch SW1 at the top right |

## 2. What to photograph (one session, one setup)
| Group | Boards | Photos | Purpose |
|---|---|---|---|
| A/B | **GOOD_001** | **10**; lift the board and push it back into the corner guide before each photo | Placement and capture repeatability. A and B coincide because every photo follows a re-placement. |
| C | **5–10 other verified-good boards** (GOOD_002 …), other colours if available | 1–3 each | Board-to-board variation |
| D | **At least 1 real defective / rejected board**, defect verified by the operator | 1–3 | Does a real defect stand out above the noise floor? |

Interleave the boards in the same session (same light, same phone settings).

## 3. Capture and ingest (camera-first, nothing typed per image)

**Preferred since 2026-10-05: live camera.** Use the phone as a live camera through DroidCam with `python scripts/acquire.py`. It captures, names, checksums, quality-checks and reviews every image (KEEP / REJECT) directly into `data/acquisition/`, `data/curated/` and `data/quarantine/`. Guide: [docs/experiment_1b_camera_acquisition.md](../../docs/experiment_1b_camera_acquisition.md).

**Not connected yet.** `run_experiment_001b.py` (section 5) still reads the Experiment 0 capture log below. It does not read the live-camera data yet.

**Photo route (Experiment 0 ingest), as before:**

1. Copy the original photos into board folders: `data/experiment_0/inbox/GOOD_001/`, `inbox/GOOD_002/`, …, `inbox/DEFECT_001/`.
2. Run **one** ingest for the whole session, with the measured distance and actual stage:

   ```bash
   python scripts/e0_ingest.py --stage after_tht --distance-mm <MEASURED_DISTANCE> --lighting "diffuse, <describe>" --notes "E1B controlled"
   ```

   The command prints the new session ID (e.g. `S20261012-01`). The colour is estimated automatically.

## 4. Ground truth (the only manual records)
- **`data/experiment_0/metadata/boards.csv`:** one row per board.
  - `condition` (`good` / `defective`), `verified_by`, `ground_truth_source`.
  - Group C needs verified-good boards; unverified boards are skipped, with the reason reported.
- **`data/experiment_0/metadata/defects.csv`:** one row per verified defect.
  - `defect_type`, `defect_location` (words), `defect_description`, `ground_truth_source`, `verified_by`.
  - **`defect_box_mm`** = `x0,y0,x1,y1` in mm.
- **How to fill `defect_box_mm`:**
  - The origin is the top-left board corner in the reference orientation; x points right (0–83), y points down (0–116).
  - Read it off `experiments/experiment_001b/reference/<reference ID>/reference_mm_grid.jpg`, which is written on the first run, or measure on the board with a ruler.
  - **Without `defect_box_mm`, the run reports that defect-overlap ground truth is missing and does not evaluate it.** Nothing is ever inferred from the difference images.

## 5. Run
```bash
python scripts/run_experiment_001b.py --sessions <SESSION_ID>
```
The reference defaults to the first GOOD_001 photo of the session; `--reference <image ID>` overrides it.

**Outputs:**
- `results.csv`: one row per pair, with its group.
- `runs/<reference>__<test>/`: reference, test, aligned image, difference map, candidate regions, verified defect box (blue), overlays, `metrics.json`.
- `summary.md` / `summary.json`, containing:
  - per group (A/B, C, D): registration error, SSIM, colour difference, edge mismatch, number of candidate regions, % of the board flagged, largest region, time per pair;
  - **handheld (Experiment 1) vs controlled**: medians, absolute and % change, for same-board and other-board pairs;
  - **verified defects**: overlapped or not; the overlapping region's size and rank among all candidates on that board; whether it is larger than every false candidate on the good boards (descriptive, not a decision rule).

## 6. Verification done so far (no new data)
- **Unit tests** (142 passing) cover: grouping from human records, summaries, the baseline comparison, defect findings, `defect_box_mm` validation, and an end-to-end pipeline run on a generated board scene.
- **Plumbing check:** the 1B script was run on the existing **handheld** photos (output in a scratch folder, not here). It reproduced Experiment 1 exactly: 0.0 % change on every metric. The two experiments are therefore measured identically.

## 7. Questions this experiment must answer (open until the data exist)
1. Is controlled reference comparison viable as a baseline? (Target: few small false candidates on re-placed GOOD_001 photos.)
2. Can it distinguish normal variation from an obvious, verified defect?
3. Which false-candidate sources remain (tall-part parallax, solder glare, board shade)?
4. Which defects still need another approach (e.g. subtle defects below the noise floor, wrong-value look-alike parts, hidden joints)?
5. What data is needed next?

The decision between YOLO, anomaly detection, segmentation or a hybrid system is taken **after** these answers, not before.
