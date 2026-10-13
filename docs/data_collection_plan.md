# Data Collection Plan: LG Indicator

| | |
|---|---|
| Status | Draft 0.3. Specific to the LG Indicator. **No images captured yet.** Experiment 0 (mobile phone) is ready to run. Production-representative capture is blocked (section 2). |
| Date | 2026-10-02 |
| Board facts | Production/PCB mentor, 2026-10-02, rounds 1 and 2 (see [production_profile.md](production_profile.md)) |
| Related | [experiment_0_camera_visibility.md](experiment_0_camera_visibility.md) · [experiment_0_capture_checklist.md](experiment_0_capture_checklist.md) · [pcb_inspection_spec.md](pcb_inspection_spec.md) · [production_questions.md](production_questions.md) · [requirements.md](requirements.md) |

Numbers marked *proposal* are starting points to agree with the production team and the mentor, not requirements. Facts about the board are quoted from the verified answers; nothing else about the board is assumed.

---

## 1. Scope and the two collection tracks

| Item | Value |
|---|---|
| Board | **LG Indicator**, real assembled boards (profile `lg_indicator/variant_1`) |
| Variant / revision | Variant `1`; single revision reported (PCB-02, PCB-03) |
| Board colour | Varies with availability: blue / red / green reported. Same PCB, same function (PCB-18). Recorded per board as a **visual variation**. |
| Side | **Top side only** (PCB-07) |
| Inspection stages | **After SMT/reflow** and **after THT** (PRD-06). Recorded per capture. A third stage is not confirmed. |
| Good boards | From the ~500 known-good boards reported available (DATA-01) |
| Defective boards | From the rejected boards reported available (DATA-02). **No artificial defects.** |
| Ground truth | Operator (DATA-06), recorded by name |
| Storage | Local (PRD-25, DATA-08) |

Data are collected in **two separate tracks**:

| Track | Camera | Purpose | Where | Protocol |
|---|---|---|---|---|
| **Experiment 0** (now) | Mobile phone, prototype only | Can a simple camera see what must be detected? | `data/experiment_0/` (photos dropped in `inbox/`, ingested automatically) | [experiment_0_camera_visibility.md](experiment_0_camera_visibility.md) |
| **Production-representative dataset** (later) | Production camera, not decided yet | Data for the model experiments E1 to E6 | `data/raw/`, `data/metadata/` | This document, sections 4 to 12 |

Phone images are **never mixed** into the production-representative dataset. Board IDs are shared between the two tracks: a physical board keeps the same ID in both.

## 2. Readiness

| Prerequisite | Experiment 0 (phone) | Production-representative dataset |
|---|---|---|
| Storage permission | ✔ Local storage allowed | ✔ (location and retention UNKNOWN, PRD-20) |
| Boards | ✔ A good and a defective board are available | ✔ Pools available; how many may be photographed UNKNOWN (DATA-01, DATA-02) |
| Operator for ground truth | ✔ Available | ✔ Identity / qualification UNKNOWN (DATA-06) |
| Camera | ✔ Mobile phone | ✘ Production camera not decided (IMG-01) |
| Lens, lighting, mount, distance, enclosure | Recorded as used | ✘ Not decided (IMG-02 to IMG-04) |
| Calibration | ✔ Ruler as scale reference | ✘ Targets UNKNOWN (IMG-07) |
| Inspection stage | ✔ Recorded per capture | ✔ Recorded per capture |
| Board colour | ✔ Recorded per board | ✔ Recorded per board |
| Fixture reference (holes) | Not needed (tape corner guide) | ✘ NEEDS CLARIFICATION (PCB-17) |
| Smallest feature dimensions | Not needed (E0 looks at the small resistors directly) | ✘ UNKNOWN (PCB-29) |
| Basis of "known-good" | Operator verification recorded | ✘ UNKNOWN (DATA-01) |
| ESD | No special restrictions reported; careful handling used | Company policy UNKNOWN unless documented (PRD-27) |

**Conclusion:**
- **Experiment 0: ready.**
- **Production-representative dataset: not ready.** It waits for the production imaging decision and the open items above.

## 3. 500 physical boards are not 500 image samples

~500 known-good boards is a pool of **physical boards**. A board becomes a usable sample only after **all** of the following:

1. It is confirmed as one of the boards that may be photographed (DATA-01).
2. It has a unique board ID and a row in the board register (section 10).
3. A named operator has verified its top side as good, against the agreed basis for "known-good" (section 5.1).
4. It has been captured under the **frozen** production setup.
5. Its captures pass the image quality check and the metadata validator.

Statistically:
- **One board is one independent sample**, however many times it is photographed. Repeated captures measure repeatability; they do not add independent samples.
- Boards of the same colour or from the same production lot may resemble each other more than other boards. Colour and lot are therefore recorded per board.

## 4. CALIBRATION / CAMERA SETUP (production-representative track)

The phone setup for Experiment 0 is defined in [experiment_0_camera_visibility.md](experiment_0_camera_visibility.md), sections 1 to 6. This section defines the **stricter** setup for the production-representative dataset, once production hardware is chosen.

### 4.1 Station requirements derived from the board
| Board fact | Station requirement |
|---|---|
| 83 × 116 mm, individual board (PCB-04, PCB-05) | Field of view ≥ 116 × 83 mm plus a margin on all sides. *Recommendation:* align the board's long side with the sensor's long side. |
| Up to ~12 mm component height (PCB-06) | Depth of field and parallax must be checked ([pcb_inspection_spec.md](pcb_inspection_spec.md), section 5.4). Working distance and lens type are chosen with this in mind. |
| Top side only (PCB-07) | One top-down camera view. No flip station. |
| Two inspection stages (PRD-06) | The station must work for boards after SMT/reflow and after THT (possibly different component heights at each stage). |
| Manual loading and positioning (PRD-09, PRD-13) | *Recommendation:* a simple locating fixture (e.g. L-shaped corner stops against two board edges). It must not cover the top side. Whether the holes (PCB-17) can be used as locating features depends on their clarification. |
| Board colour varies: blue / red / green (PCB-18) | A **neutral, matte** background (white / grey / black), never one of the board colours. Lighting and white balance must work for every colour that occurs. |
| No conformal coating (PCB-19) | No UV lighting needed. |
| Small resistors are the smallest objects (PCB-29); exact size UNKNOWN | The resolution requirement is set once their dimensions are known, using the px/mm measured in Experiment 0 as evidence ([pcb_inspection_spec.md](pcb_inspection_spec.md), section 5.3). |
| Ambient light UNKNOWN (PRD-11); enclosure not decided (IMG-03) | *Recommendation:* enclose the camera area, whatever the ambient conditions turn out to be. |

### 4.2 Capture parameters (fixed per session)
Each parameter has a requirement, a check, and the column(s) where it is recorded in `sessions.csv`.

**Camera resolution**
- **Requirement:** the full native sensor resolution. No binning, no on-camera resizing.
- **Check:** measure mm/px on the geometric calibration target.
- **Recorded:** `camera_model`, `camera_serial`; mm/px in `notes`.

**Lens**
- **Requirement:** fixed focal length, low distortion. Focus and aperture rings locked. No zoom lens.
- **Check:** distortion on the calibration grid.
- **Recorded:** `lens`.

**Distance**
- **Requirement:** a fixed working distance on a rigid mount. Never hand-held.
- **Check:** mm/px unchanged between sessions.
- **Recorded:** `working_distance_mm`.

**Angle**
- **Requirement:** optical axis perpendicular to the board.
- **Check:** the grid looks square, with equal mm/px in all four corners.
- **Recorded:** `notes`.

**Lighting**
- **Requirement:** stabilised LEDs, warmed up before capture, ambient light blocked. Candidate setups are compared on the same boards.
- **Check:** grey-reference brightness is stable between session start and end.
- **Recorded:** `lighting_setup` (an ID that points to a documented configuration with a photo).

**Focus**
- **Requirement:** manual focus, then locked. The focus plane (board surface vs. component tops) is chosen from the Experiment 0 findings.
- **Check:** sharpness at the centre and in the corners.
- **Recorded:** `aperture_f`; `notes`.

**Exposure**
- **Requirement:** manual. Auto-exposure, auto-gain and auto-white-balance off. Lowest practical gain. No large clipped areas.
- **Check:** histogram and clipped-pixel fraction.
- **Recorded:** `exposure_us`, `gain_db`, `white_balance`.

**Background**
- **Requirement:** matte, neutral, uniform, identical in every session.
- **Check:** visual.
- **Recorded:** `notes`.

**Fixture**
- **Requirement:** the same locating fixture for good and defective boards.
- **Check:** re-seat repeatability.
- **Recorded:** `station_id`.

**Orientation**
- **Requirement:** one canonical top-side orientation, documented with a reference photo (the same one as Experiment 0, if it is confirmed).
- **Check:** the operator compares each placement with the reference photo.
- **Recorded:** side `TOP` in the filename.

**Field of view**
- **Requirement:** the whole board plus margin.
- **Check:** all four board edges are visible.
- **Recorded:** `notes`.

**Image format**
- **Requirement:** lossless PNG or TIFF only, at the highest bit depth available. No in-camera sharpening, denoising, HDR, cropping or resizing.
- **Check:** `scripts/validate_dataset.py` rejects lossy files.
- **Recorded:** `image_format`, `bit_depth`.

### 4.3 Calibration captures (every session, at start and end)
1. **Geometric target** (checkerboard or dot grid of known pitch, covering at least 116 × 83 mm) at the board plane. Gives mm/px, distortion and a perpendicularity check.
2. **Grey / colour reference card.** Gives exposure and colour stability.
3. **Control board.** One designated good LG Indicator board, captured in every session, to track drift. Mark it "control board" in `boards.csv` `notes`.

Files go to `data/raw/calibration/<session_id>/<session_id>_<GEOM|GREY>_<NN>.png` and are listed in `sessions.csv` `calibration_files`.

### 4.4 Gate
Production station settings are fixed only after the production hardware has been chosen, informed by the Experiment 0 results, and tested on real boards. They are then entered into the profile's `imaging` section. The production-representative dataset is captured only with that frozen setup.

## 5. GOOD BOARDS

### 5.1 When a board counts as good
The top side of a board counts as **good** only if:
- the board comes from the known-good pool, **and**
- a named operator has inspected the top side and found no defect. The operator is recorded in `verified_by` (DATA-06), and the source of that status in the board's ground-truth source.

How "known-good" was originally established for the ~500 boards is still UNKNOWN (DATA-01). If electrical test results per board become available (DATA-03), they are recorded. A board that fails either condition is not used as a good sample.

### 5.2 Intake procedure (per board)
1. Assign the next free ID from the good pool (`GOOD_001`, `GOOD_002`, …). Put the board in an individually labelled bag or tray marked with that ID.
2. Record in the board register:
   - `pcb_model` = `LG Indicator`, `pcb_variant` = `1`.
   - `pcb_revision` = `UNKNOWN` (only one revision is reported; its identifier is not known).
   - `board_colour`, as observed.
   - `production_lot` / `manufacture_date`, if known.
   - `source` = `production`; `verified_by`.
3. Ask the operator which **inspection stage** the board is in (after SMT/reflow, or after THT). Record it on every capture.
4. The operator verifies the top side (section 5.1).
5. Capture per section 5.3.

### 5.3 Captures per board
*Proposal:* 3 top-side captures, with the board removed and re-seated between captures, repeated in at least 2 sessions on different days.
- Re-seating measures the variation from manual positioning.
- Different days measure station drift.

These repeats are *not* extra independent samples (section 3).

### 5.4 Variation to include deliberately
- **Every board colour** that occurs (PCB-18), with the number of boards per colour recorded. Colour is a visual variation of the same PCB, not a separate variant.
- **Both inspection stages** (after SMT/reflow and after THT), with the stage recorded per capture.
- Boards from as many production lots / dates as are available (DATA-01).
- The normal range of manual placement within the fixture.

### 5.5 Storage
```
data/raw/good/GOOD_001/GOOD_001_TOP_S<date>-01_C001.png
```

## 6. DEFECTIVE BOARDS

### 6.1 Sources for the LG Indicator
1. **Rejected boards: the primary source.** They are reported available, and a defective board is available now (DATA-02). The count, defect categories and records are UNKNOWN.
2. **Other failed boards** (repair queue, archive), if they exist. Confirm the defect is still present and has not been reworked.
3. **Controlled defects: NOT permitted at present.** DATA-05 is UNKNOWN and permission is not assumed. No artificial defects are created.
4. **Synthetic defects: not part of this phase.** If they are ever justified, they stay separate (`data/processed/synthetic/`), are never used in validation or test, and are never reported as real-world performance.

### 6.2 Intake procedure (per rejected board)
1. Assign the next free ID from the defective pool (`DEFECT_001`, `DEFECT_002`, …) and record the board (`source` = `production_reject`) as for good boards, including colour and inspection stage.
2. The operator examines the board and records **each** defect in `defects.csv`:
   - `defect_type`, taken from the taxonomy (all categories are in scope, PCB-24).
   - `side` (`TOP` or `BOT`).
   - `component_refdes`, if known. If the board carries printed reference designators, use them. Without a BOM (PCB-11 UNKNOWN), describe the position instead.
   - `location_description`.
   - `visible_in_images` (`yes` / `no` / `uncertain`), once the board has been imaged.
   - `verified_by`.
   - Any existing reject record or electrical test result (DATA-02, DATA-03, DATA-04).
3. A defect on the bottom side is still recorded, with `side` = `BOT`. It is not captured, because the scope is top only.
4. Capture the top side under the **same** setup as the good boards, interleaved in the same sessions. The model must not be able to tell good from defective by session, colour or stage. Record colour and stage so that this can be checked.

### 6.3 Labelling rule
`condition` describes the **captured top side**:
- `defective` if at least one recorded defect is on the top side (even if it is not visible: `visible_in_images` = `no`).
- `good` if all recorded defects are on the bottom side.

The ID prefix never changes. A `GOOD_` board found to be defective keeps its ID, with `condition` = `defective`.

## 7. Collection phases

| Phase | Purpose | Boards (*proposal*) | Gate to start |
|---|---|---|---|
| 0: Experiment 0 (**phone**) | Camera visibility and repeatability, not model accuracy | Minimum: `GOOD_001` + `DEFECT_001`. Full: 5 good boards (each colour available) + every available defective board with known defect information ([experiment_0_camera_visibility.md](experiment_0_camera_visibility.md), section 7) | **Ready now** |
| 1: Dataset v1 (production setup) | First dataset for experiments E1, E2, E4 | e.g. 100 distinct good boards (all colours and both stages) plus every available rejected board with a top-side defect | Production imaging chosen and frozen; PCB-17 and PCB-29 clarified; basis of "known-good" confirmed |
| 2: Extension | More boards if the experiments show a benefit, prioritising new lots and colours | Remaining photographable boards | Results of E1 to E4 |

The Phase 1 size is a proposal. It is revised once the capture time per board and the operator time available are known.

## 8. Metadata (production-representative track)

Experiment 0 uses its own capture log, filled automatically by `scripts/e0_ingest.py` ([experiment_0_camera_visibility.md](experiment_0_camera_visibility.md)). The tables below are development / dataset-mode metadata for the later production-representative dataset. Here too, IDs, capture time and camera data are to be generated by the capture software, not typed by an operator.

The tables below live in `data/metadata/`, CSV UTF-8, editable in Excel / LibreOffice (save as "CSV UTF-8"). Required columns are marked ✔. A required value that is genuinely unknown is written as `UNKNOWN`, never left blank. The column definitions are enforced by `src/pcb_inspection/dataset/manifest.py`.

**`boards.csv`**: one row per physical board.

| Column | ✔ | LG Indicator guidance |
|---|---|---|
| `board_id` | ✔ | `GOOD_001` … / `DEFECT_001` … (the same ID as in Experiment 0) |
| `serial_number` | | Empty: no serial / barcode in the current scope (PCB-21) |
| `pcb_model` | ✔ | `LG Indicator` |
| `pcb_variant` | ✔ | `1` |
| `pcb_revision` | ✔ | `UNKNOWN` until the revision identifier is known |
| `board_colour` | | Board colour as observed: `blue` / `red` / `green` / other |
| `production_lot`, `manufacture_date` | | If known |
| `source` | ✔ | `production` (good pool), `production_reject` (rejected), `failed_archive`, `engineering` |
| `electrical_test_result` | | If available (DATA-03) |
| `verified_by` | ✔ | Operator who verified the board |
| `notes` | | e.g. "control board", "returned to production on …" |

**`sessions.csv`**: one row per capture session (fixed settings).
Columns: `session_id` ✔, `date` ✔, `station_id` ✔, `camera_model` ✔, `camera_serial` ✔, `lens` ✔, `working_distance_mm` ✔, `lighting_setup` ✔, `exposure_us` ✔, `gain_db` ✔, `aperture_f`, `white_balance` ✔, `image_format` ✔, `bit_depth` ✔, `calibration_files` ✔, `operator_id` ✔, `notes`.

**`captures.csv`**: one row per image.
Columns:
- `capture_file` ✔: path relative to `data/`.
- `condition` ✔: `good` / `defective` for the captured side.
- `inspection_stage` ✔: `after_smt_reflow` / `after_tht`.
- `captured_at` ✔: ISO 8601.
- `defect_ids`: required for `defective`.
- `reseated_before_capture`, `sha256`, `notes`.

**`defects.csv`**: one row per physical defect.
Columns: `defect_id` ✔, `board_id` ✔, `side` ✔, `defect_type` ✔, `component_refdes`, `location_description` ✔, `severity`, `origin` ✔ (`production`; `controlled` only with authorization), `verified_by` ✔, `visible_in_images`, `electrical_test_result`, `notes`.

## 9. Naming conventions

| Identifier | Format | Example |
|---|---|---|
| Board ID | `GOOD_###` / `DEFECT_###` (prefix = intake pool, never changed) | `GOOD_001`, `DEFECT_001` |
| Side | `TOP` (current scope) / `BOT` | `TOP` |
| Session ID | `SYYYYMMDD-NN` | `S20261015-01` |
| Capture ID | `C###` | `C001` |
| Defect ID | `DEF_###` (the same ID in both tracks) | `DEF_001` |
| Filename | `<board_id>_<side>_<session_id>_<capture_id>.<ext>` | `GOOD_001_TOP_S20261015-01_C001.png` |
| Path | `raw/<condition>/<board_id>/<filename>` (relative to `data/`) | `raw/good/GOOD_001/…` |

ASCII only, no spaces, uppercase IDs, lowercase `png` / `tif` / `tiff`. Never rename a file after capture. Run the validator after every session:

```bash
python scripts/validate_dataset.py
```

## 10. Physical board identity
- The LG Indicator has **no serial number or barcode** in the current scope (PCB-21). Board identity is therefore maintained **by us**:
  - Every board gets an ID at intake, `GOOD_###` or `DEFECT_###` according to the pool it came from. The ID is kept for life, in Experiment 0 and in every later dataset, and is never reused or renamed.
  - Each board is stored in an individually labelled bag or tray carrying its ID.
  - Boards are never written on or stickered. Any mark becomes a visible feature that a model can learn.
  - Only one board is out of its bag / tray at the capture station at any time, so that IDs cannot be swapped.
  - **Image file names are never the board identity.** Each image's board ID comes from the board register.
- If boards must be returned to production, record this in `notes`. A board that comes back keeps its ID, but only if its identity can be confirmed; otherwise it gets a new ID with a note.
- **Handling:** the mentor reported no special restrictions; a formal company ESD policy is UNKNOWN unless documented (PRD-27). The project uses careful handling anyway (hold by the edges, gloves or clean hands, no touching the top side), because fingerprints and dust appear in images and the boards must be protected.

## 11. Dataset splitting
- **No splits are created now.** Raw data are still being collected (`data.split.created: false`).
- When splits are made: **split by `board_id`**. All captures of one physical board (all sessions, stages and conditions) go to the same split. This is implemented and unit-tested in `src/pcb_inspection/dataset/splitting.py`.
- **Stratify** by condition, board colour and inspection stage, and by lot where lots are known, so that every split contains each colour and stage.
- **Ratios:** *proposal* 60 / 20 / 20 (train / validation / test), confirmed once the dataset size is known.
- **Roles:**
  - train: fitting and building references (golden boards come from train only).
  - validation: thresholds and decision rules.
  - test: final reporting only.
- **Special cases:** the control board goes to train. Each dataset version stores its split file, and test boards never change within a version.

## 12. Annotation strategy

**Gate: image annotation does not start until the real image-acquisition setup is defined.** That means: production imaging chosen and frozen, and dataset v1 captures existing. Annotations made on images from a setup that later changes would have to be redone. **Experiment 0 phone images are not annotated.** Their defects are only described in `defects.csv` and rated for visibility.

Before that point, only **defect records** of the physical boards are made (what is wrong and where on the board). These are not image annotations.

Once annotation starts, the annotation type follows the defect and the approach:

| Defect | Primary annotation | Notes for the LG Indicator |
|---|---|---|
| Missing component | Per-component region label (present / missing) | Regions from CAD need BOM + pick-and-place file (PCB-11, PCB-12, UNKNOWN). Otherwise regions are drawn once on a reference image (per stage), or boxes are used. |
| Damaged component | Region label + polygon around the damage | |
| Misaligned component | Rotated box, or measured offset / rotation vs. reference | Tolerance UNKNOWN (PCB-23) |
| Wrong component | Region label (expected / other) | Look-alike parts UNKNOWN (PCB-15) |
| Reversed polarity | Orientation label per polarized part | Polarized parts UNKNOWN (PCB-13) |
| Missing / damaged lead | Box or keypoints per lead / component | Top-visible leads only |
| Broken / missing trace | Mask or polyline | Only if Experiment 0 shows traces are visible |
| Damaged PCB | Polygon / mask | |
| Solder defect | Region label or box per joint | Top-visible joints only (see PCB-08) |
| Foreign object | Box or mask | |
| Other anomaly | Image-level label + coarse mask | Masks are needed to evaluate localization, not to train |

- **Format:** COCO JSON master annotations in `data/annotations/<dataset_version>/`, in raw pixel coordinates. Model-specific formats are generated by scripts.
- **Tool:** a self-hosted tool (e.g. CVAT, Label Studio); local storage is allowed.
- **Consistency:** a written guideline with example images per defect type. The operator(s) who label are recorded, and a subset is double-labelled to check consistency.
