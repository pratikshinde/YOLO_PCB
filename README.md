# PCB Pre-Test Visual Inspection

An AI-assisted visual inspection station that photographs each assembled PCB *before* electrical/functional test. It detects visible assembly and board defects and returns **PASS / FAIL / REVIEW** with an explanation.

```
assembled PCB → camera → capture → image quality check → alignment → inspection
→ defect detection + localization → decision engine → PASS / FAIL / REVIEW → electrical test
```

## Status: Phase 1 (foundation), clean real-data collection (2026-10-06)

**Current phase: CLEAN REAL DATA COLLECTION, GOOD PCB / FAULT PCB only.** Guide: [docs/clean_data_collection.md](docs/clean_data_collection.md).

- **Reset.** All earlier image data was **archived** to `archive/2026-10-06_pre_collection_reset/`: 4,852 files, 2,538 images, each listed with its SHA-256 in `manifest.csv`. In 94 of the 96 old kept / station frames the DroidCam watermark lay over the PCB, and the old golden reference hid LED4.
- **Collection.** Data is collected fresh with `python scripts/collect.py` into `data/collection/`. Each image goes through: live camera, 3 s countdown, capture, **original saved first**, **PCB crop** (canonical board image without background or watermark), **capture check** (complete PCB, inside the ROI, watermark outside the PCB, focus, exposure: OK / REVIEW / RECAPTURE), then operator KEEP / REJECT. Every image belongs to a physical board ID (`GOOD_###`, `FAULT_###`).
- **Not in this phase:** TEST collection (later), synthetic data, model training, production PASS / FAIL. The golden bank is NOT_CONFIGURED until a clean reference is chosen.
- **Tested so far** with generated frames and a fake camera only; not yet with the live phone stream. No new image exists yet.

**Current scope:**
- **Board:** LG Indicator, 1 variant, 1 revision, 83 × 116 mm. The board colour varies (blue / red / green) on the same PCB.
- **Inspection:** top side, at two stages (after SMT/reflow, after THT), with manual loading.

These facts come from two rounds of verified production answers (2026-10-02). See [docs/pcb_inspection_spec.md](docs/pcb_inspection_spec.md).

**Camera-first.** The product is `camera → AI inspection → PASS / FAIL / REVIEW`; the operator types nothing per inspection (inspection mode, not built yet). Metadata collection belongs to development / dataset mode and is automated ([docs/requirements.md](docs/requirements.md), section 2.1).

**Earlier tracks.** Their data is archived; the code and docs are kept.
- **Experiment 0** (handheld phone photos, `scripts/e0_ingest.py`) is closed. Docs: [docs/experiment_0_camera_visibility.md](docs/experiment_0_camera_visibility.md).
- **Experiment 1B acquisition** (`scripts/acquire.py`: GOOD / DEFECT / TEST, old `data/acquisition` layout) is superseded by `scripts/collect.py`. It now runs only with `--legacy`. Guide: [docs/experiment_1b_camera_acquisition.md](docs/experiment_1b_camera_acquisition.md).

**Product-profile inspection architecture (2026-10-05).** Products are configuration (`configs/products/<product_id>/`: LG Indicator; Pump Controller identity only).
- Each product has a golden reference **bank** of GOOD boards, automatic product identification, and the generic `InspectionEngine.inspect(image, product_profile, reference_bank)` with GOOD / DEFECT / REVIEW / ERROR.
- A capture-first live station (`python scripts/station.py`) saves every capture before deciding; human review is stored separately.
- Validation covers every image of every product (`python scripts/pcb.py validate`); for LG that is now `data/collection/`.
- The engine is the EXPERIMENTAL `REFERENCE_BASELINE`.
- **Since the 2026-10-06 reset the LG golden bank is NOT_CONFIGURED:** the watermarked reference is archived, so every station inspection is REVIEW.

See [docs/inspection_architecture.md](docs/inspection_architecture.md).

| Exists and is tested | Does not exist yet |
|---|---|
| Requirements, data-collection protocol, model strategy, experiment plan, production questionnaire | Any images of the clean collection (the phase starts now; earlier images archived, not used) |
| Clean GOOD / FAULT collection: PCB crop / canonical board image, capture check (complete PCB, ROI, watermark outside the PCB, focus, exposure), KEEP blocked on RECAPTURE, summary, audit, reference check (tested with generated frames and a fake camera) | A run with the real DroidCam stream; a verified clean golden reference; golden bank built from the GOOD PCB images |
| Project config + LG Indicator inspection profile; unknowns `null`, ambiguous answers recorded verbatim | Any trained model or model weights |
| Camera-first Experiment 0 ingest: automatic IDs, EXIF metadata, checksum, filing; capture-log validation | Inspection mode (operator workflow), production camera / lens / lighting |
| Live-camera acquisition + review application (Experiment 1B): camera by name, timed batch capture, session metadata, quality check, KEEP / REJECT, TEST isolation, board split groups (tested with a mock camera) | Run with the real DroidCam stream; Experiment 1B analysis reading acquisition data; annotation tool |
| Experiment 2 prototype: TEST image → alignment to GOOD images → per-component comparison → "MISSING COMPONENT: R37" ([report](experiments/experiment_002/README.md)); component-box tool | A **verified** component map (55-part draft only, no BOM / centroid file); a reliable detector (≈0.03 margin; needs more GOOD boards with board IDs) |
| Simple image-quality checks (readable, resolution, sharpness, exposure, board in frame, colour estimate, orientation, px/mm) | Inspection modules / defect detection, annotations |
| Experiment 1 baseline: board alignment (SIFT + homography, ≈0.05 mm residual) + reference comparison; run once on real photos ([report](experiments/experiment_001/README.md)) | Any trained model |
| Dataset metadata schema + validator; board-level (leakage-free) splitting, not yet applied | Camera / PLC / MES integration, operator UI |
| Decision-engine contract with fail-safe rules | |

The inspection model is **not chosen yet**. YOLO is one candidate among several (see [docs/model_strategy.md](docs/model_strategy.md)). No model is trained until real images have been collected and inspected.

## Documents

| Document | Content |
|---|---|
| [docs/clean_data_collection.md](docs/clean_data_collection.md) | **Current phase:** reset / archive, watermark handling, PCB crop, capture check, folders, metadata, exact GOOD / FAULT capture steps, reference verification |
| [docs/inspection_architecture.md](docs/inspection_architecture.md) | Product-agnostic architecture: product profiles, golden bank, identification, engine, decision states, storage, review, validation, commands, limitations |
| [docs/pcb_inspection_spec.md](docs/pcb_inspection_spec.md) | Current scope (LG Indicator), profiles / colour variation / stages, imaging implications |
| [docs/production_profile.md](docs/production_profile.md) | What is KNOWN / UNKNOWN / NEEDS CLARIFICATION from production |
| [docs/production_questions.md](docs/production_questions.md) | All questions with answers, status and source; the next questions |
| [docs/experiment_0_camera_visibility.md](docs/experiment_0_camera_visibility.md) | Experiment 0 protocol (mobile phone): capture, metadata, folders, quality check, visibility review |
| [docs/experiment_0_capture_checklist.md](docs/experiment_0_capture_checklist.md) | Printable checklist for the first capture session |
| [docs/experiment_1b_camera_acquisition.md](docs/experiment_1b_camera_acquisition.md) | Live-camera acquisition + review system (DroidCam): workflow, modes, folders, metadata, first 10-image session |
| [docs/data_collection_plan.md](docs/data_collection_plan.md) | Both collection tracks; production-representative setup, good / defective boards, metadata, identity, splitting, annotation |
| [docs/requirements.md](docs/requirements.md) | Business problem, functional / non-functional / defect / input / output / hardware / data / evaluation requirements |
| [docs/model_strategy.md](docs/model_strategy.md) | Comparison of reference comparison, registration, detection, segmentation, anomaly detection, hybrid; colour / stage strategy |
| [docs/experiment_plan.md](docs/experiment_plan.md) | Experiments E0 to E6, metrics, record template |

## Layout

```
configs/
  project.yaml                  project-wide config (line, station, protocol); null = UNKNOWN
  profiles/lg_indicator/
    variant_1.yaml              inspection profile: board facts, colour variation, stages, defect rules
  capture/mobile_phone_e0.yaml  phone prototype capture settings + quality-check thresholds
  capture/droidcam_collection.yaml  CURRENT: clean GOOD / FAULT collection (camera, PCB check, review, fault types)
  capture/droidcam_e1b.yaml     phone as live camera (DroidCam), Experiment 1B acquisition + review settings (legacy)
archive/2026-10-06_pre_collection_reset/   all earlier image data (original paths) + manifest.csv; NOT used
data/
  collection/                   CURRENT clean dataset (scripts/collect.py)
    originals/<good|fault>/<board>/<session>/   session.json, images/ (originals), pcb/ (PCB images)
    good/<GOOD_###>/<session>/, fault/<FAULT_###>/<session>/   KEEP (original + PCB image)
    quarantine/<good|fault>/<board>/<session>/                 REJECT
  raw/good/, raw/defective/     production-representative captures (later)   raw/good/GOOD_001/<capture>.png
  raw/calibration/              per-session calibration target images
  processed/                    derived data only, never edited raw data
  annotations/                  labels, per dataset version (not started)
  metadata/                     boards.csv, sessions.csv, captures.csv, defects.csv
docs/                           project documentation (above)
experiments/TEMPLATE.yaml       record template for every experiment
experiments/experiment_001/     Experiment 1: config, report, reference, per-pair images + metrics, results.csv
experiments/experiment_001b/    Experiment 1B (controlled capture): protocol; results appear here after the controlled session
experiments/experiment_002/     Experiment 2: config, component map (draft), watermark mask, report, runs/ (one folder per image)
scripts/                        command-line tools (below)
src/pcb_inspection/
  config.py                     config + profile loading, unresolved-value report
  dataset/                      naming rules, metadata validation (both tracks), board-level splitting
  acquisition/                  camera-first ingest of photos; live camera, sessions, timed capture, review UI
  image_quality/                simple image checks (NumPy + Pillow): sharpness, exposure, board in frame, colour
  preprocessing/                board segmentation, canonical 83 x 116 mm board frame, illumination normalisation
  alignment/                    feature-based registration (board only) + independent residual measurement
  comparison/                   difference maps (colour dE, SSIM, edges) and candidate regions; per-component check (E2)
  visualization/                experiment images (heat maps, overlays, panels)
  decision_engine/              findings + stage results -> PASS / FAIL / REVIEW
tests/                          unit tests
```

Modules are added only when there is real work for them: acquisition, alignment, inspection methods, visualization, plus `models/`, `notebooks/` and `app/`.

## Commands

Python ≥ 3.10. Runtime dependencies: PyYAML, NumPy, Pillow, OpenCV (headless); the acquisition window uses Tkinter (included with Python on Windows). Tests need pytest.

Optional: create an isolated environment.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -e .[dev]
```

Run the tests:

```bash
python -m pytest
```

**Clean data collection (current phase).** Start the GOOD / FAULT collection window (live DroidCam camera; the DroidCam client must be streaming):

```bash
python scripts/collect.py
```

Counts (GOOD / FAULT boards and images, TEST = 0, archived old data), capture checks and audit:

```bash
python scripts/collection_summary.py
```

Verify one kept GOOD capture as a clean reference (full PCB, watermark outside, crop, quality; writes a sheet for the visual LED4 check):

```bash
python scripts/collection_summary.py --verify-reference GOOD_001_TOP_S20261006-01_C001
```

Archive old image data (dry run; `--apply` moves it to `archive/`, never deletes):

```bash
python scripts/archive_old_data.py
```

Live inspection station: automatic capture, product identification, GOOD / DEFECT / REVIEW. Every capture is stored first in `data/captures/`. Add `--headless --captures 5` to run without a window, or `--product lg_indicator` to select the product:

```bash
python scripts/station.py
```

Products, physical boards, dataset validation (every image of every product), inspecting stored images, and human review:

```bash
python scripts/pcb.py products
```

```bash
python scripts/pcb.py boards --product lg_indicator
```

```bash
python scripts/pcb.py validate
```

```bash
python scripts/pcb.py inspect <image.png>
```

```bash
python scripts/pcb.py review <inspection_id> --decision CONFIRM_GOOD --reviewer NAME
```

Legacy Experiment 1B acquisition (GOOD / DEFECT / TEST PCB sessions, old `data/acquisition` layout). Superseded by `scripts/collect.py`; it runs only with `--legacy`:

```bash
python scripts/acquire.py --legacy
```

Camera diagnostic: open → read frames → print frame size → release. It uses the `camera.index` / `backend` from `configs/capture/droidcam_e1b.yaml` (DirectShow); add `--list` to list devices or `--index N` for another camera:

```bash
python scripts/check_camera.py
```

Summary and consistency audit of the acquired, curated and quarantined images:

```bash
python scripts/acquisition_summary.py
```

Experiment 2: inspect every image in `data/testing/input/`. It names missing components; results go to `data/testing/output/`. Or pass images or folders as arguments:

```bash
python scripts/inspect_pcb.py
```

Live inspection station: the camera captures one image every 3 s and runs the unchanged Experiment 2 inspection on it. Results are shown live and saved to `data/testing/live/` and `data/testing/live_output/`. Keys: SPACE inspect now, R reset, Q / ESC quit:

```bash
python scripts/live_inspection.py --camera-index 1 --interval 3
```

Experiment 2: draw, name and verify the component boxes on the GOOD reference (the current map is an unverified draft):

```bash
python scripts/define_components.py
```

Ingest new Experiment 0 phone photos from `data/experiment_0/inbox/` (automatic IDs, EXIF metadata, filing, quality check):

```bash
python scripts/e0_ingest.py
```

Experiment 1 (alignment + reference comparison; image IDs or paths):

```bash
python scripts/run_experiment_001.py --reference S20261002-01_C003 --test S20261002-01_C001 S20261002-03_C001
```

Experiment 1B (controlled capture vs handheld baseline; groups A/B, C, D from boards.csv / defects.csv):

```bash
python scripts/run_experiment_001b.py --sessions <SESSION_ID>
```

Re-check all ingested Experiment 0 photos and their log (e.g. after changing thresholds):

```bash
python scripts/check_e0_images.py
```

Show readiness gates with their exact blockers, open clarifications and unknown values (add `--list-unknowns` for every path):

```bash
python scripts/check_readiness.py
```

Validate the production-representative dataset (`data/raw/`, later):

```bash
python scripts/validate_dataset.py
```
