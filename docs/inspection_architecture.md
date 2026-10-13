# Inspection architecture: product-agnostic, profile-driven

> **Reset 2026-10-06** ([clean_data_collection.md](clean_data_collection.md)):
> - All LG image data described below (bank of 5 boards, Experiment 0 / 1B / 2 images, station captures) and the watermarked geometric reference were **archived** to `archive/2026-10-06_pre_collection_reset/`.
> - The LG golden bank is **NOT_CONFIGURED** (`geometric_reference: null`), so every station inspection is REVIEW.
> - LG data collection now uses `data/collection/` (`storage.layout`), GOOD / FAULT only, with a capture-time PCB check.
> - The measurements in sections 5, 12 and 13 describe the archived data.

Status 2026-10-05. Current product: **LG Indicator**. Declared future product: **Pump Controller** (identity
only). The inspection engine in use is the **REFERENCE_BASELINE**. It is an **EXPERIMENTAL BASELINE**, not
production inspection (section 13).

## 1. Principles (non-negotiable)

- There is no single golden image. Each product has a **golden reference BANK** with every verified GOOD
  physical board. A new GOOD board is judged against the bank (all other GOOD boards), never against
  GOOD_001 alone.
- **Every capture is stored first**, with SHA-256 verification, before identification or inspection, and
  also when the result is REVIEW or ERROR.
- **Products differ only by configuration.** Code never contains `if product == ...`. Adding a product
  means adding `configs/products/<product_id>/` and its data.
- An image is compared **only with the references of the product it was identified (or selected) as**. An
  unknown or ambiguous product gives REVIEW.
- Alignment failure, image-quality failure, too few golden boards and uncertain measurements give
  **REVIEW**. They are never reported as a PCB defect.
- **Confidence is never invented.** No calibrated confidence exists yet, so `confidence` is `null` and the
  measured values are reported instead.
- **No silent data loss.** Nothing is deleted, overwritten, merged or ignored. Images that cannot be
  classified are reported as UNRESOLVED.

## 2. Workflows

**Live inspection** (`scripts/station.py`) needs no operator input:

```
PCB placed -> camera preview: PCB in view and still -> ONE automatic capture
  -> 1 original saved + checksum + image record      data/captures/<session>/<image_id>.png / .json
  -> 2 image quality (station thresholds)            blocking flag -> REVIEW (image quality)
  -> 3 product: AUTO identification or SELECT PRODUCT  unknown / ambiguous -> REVIEW
  -> 4 product quality check (product thresholds, board size)
  -> 5 InspectionEngine.inspect(image, product_profile, reference_bank)
  -> 6 result.json + evidence (product folder); image record updated
  -> screen: PRODUCT (AUTO DETECTING... / name), RESULT GOOD / DEFECT (type, location) / REVIEW (reason)
```

The next capture happens only after the board is removed or replaced. SPACE captures at once.

**Data collection** (`scripts/acquire.py --product <id>`): the operator chooses GOOD / DEFECT / TEST,
there is a 3-second countdown, a quality check, then KEEP (curated/) or REJECT (quarantine/). Each session
records product, variant, board, stage and session. This is the only place where the operator declares
GOOD or DEFECT.

**Human review** (station window buttons, or `scripts/pcb.py review`): CONFIRM DEFECT / CONFIRM GOOD /
REJECT RESULT. It is stored separately and never changes the AI result.

## 3. Components

| Layer | Module | Knows products? |
|---|---|---|
| Camera | `pcb_inspection.camera`: `Camera` with `open / read / capture / get_info / close`; `LiveCamera` (OpenCV / DroidCam), `MockCamera`, `ReplayCamera` | no |
| Profiles | `pcb_inspection.products`: `load_product_profile(product_id)`, `load_all_profiles` | reads `configs/products/` |
| Golden bank | `pcb_inspection.inspection.bank.ReferenceBank`, `load_reference_bank(profile)` | data from the profile |
| Identification | `pcb_inspection.inspection.identification.ProductIdentifier` | profiles + banks only |
| Engine | `pcb_inspection.inspection.engine.InspectionEngine.inspect(image, product_profile, reference_bank)` | no |
| Backends | `BACKENDS = {"REFERENCE_BASELINE": ...}`; a YOLO or anomaly backend is registered the same way | no |
| Station | `pcb_inspection.station`: `InspectionStation`, `CaptureStore`, results / reviews, `AutoTrigger` | no |
| Validation | `pcb_inspection.dataset.product_discovery.validate_all` | data from the profile |

A test (`test_core_code_names_no_product`) fails if the product-agnostic packages mention a product name
or the LG board size.

## 4. Product profiles: `configs/products/<product_id>/`

```
product.yaml              product_id, name, revision, variants, board (length_mm, width_mm, thickness_mm,
                          inspection_side, orientation, allowed_rotation_deg), stages, appearance
                          (colour_variants, colour_is_variant), storage, identification
variants/<variant>.yaml   variant_id, label, revision, engine (type, status, config), alignment limits,
                          quality thresholds, golden / regions / rules files
golden/golden_bank.yaml   geometric_reference, reference_split, min_boards, calibration, max_rotation_deg (bank)
regions/regions.yaml      component / LED / connector / trace / hole / ignore regions (null = not defined)
inspection_rules.yaml     defect types (enabled, severity), decision thresholds
```

`null` means UNKNOWN. The loader never fills in a fact. The only defaults are paths: without `storage.root`,
a product stores under `data/products/<product_id>/`. Two products may not share a storage folder (reported
as a profile problem).

- **LG Indicator.** Facts are copied from the verified answers in
  `configs/profiles/lg_indicator/variant_1.yaml`; a test keeps the two equal. Data collection stays in the
  existing store (`storage.acquisition_root: data`, registry `data/dataset/`), so Experiment 0 / 1B / 2
  data and tools stay valid. Station inspections and reviews go to `data/products/lg_indicator/`.
- **Pump Controller.** Name only; everything else is null. Its golden bank is NOT_CONFIGURED, so it cannot
  be identified automatically. If the operator selects it, every inspection is REVIEW
  (`NO_GOLDEN_REFERENCES`), and it is never compared with LG references. Data collection refuses to start
  until `board.inspection_side` is known, because the side is part of every image ID.

### Adding a product (e.g. making Pump Controller inspectable): configuration and data only

1. Fill in `product.yaml`: board size, side, stages and allowed rotation.
2. Collect GOOD boards with `python scripts/acquire.py --product pump_controller`. Give each physical board
   an ID (GOOD_001, …) and KEEP good captures.
3. Register the boards in `data/products/pump_controller/dataset/boards.csv` (condition `good`, split
   `reference`), and add DEFECT / TEST boards with their own splits.
4. Choose one curated GOOD image as `golden_bank.yaml geometric_reference`, draw its regions
   (`python scripts/define_components.py`, or any tool writing the same YAML), and set `regions.yaml` and
   `min_boards`.
5. Enable the defect types and set decision thresholds in `inspection_rules.yaml`.
6. Run `python scripts/pcb.py products` and `python scripts/pcb.py validate --product pump_controller`.

No code changes. `test_future_product_needs_configuration_only` adds a product this way and inspects it.

## 5. Golden reference bank

- **Membership.** Every GOOD board that the product's registry (`boards.csv`) puts in the reference split,
  with all its KEPT images (acquisition sessions; for LG also Experiment 0). TEST, DEFECT and validation
  boards never enter it.
- **Geometric reference.** One curated GOOD image defines the canonical frame (mm, origin top-left) and the
  region map is drawn on it, checked by SHA-256. It is the coordinate frame only. Appearance comes from the
  whole bank.
- **Leave-one-board-out.** When an image of a bank board is inspected (board ID known), all images of that
  board are left out.
- **Status.** READY (≥ `min_boards` boards); INSUFFICIENT (identification possible, every inspection
  REVIEW); NOT_CONFIGURED; BROKEN (gives ERROR).
- **Bank orientation** (`max_rotation_deg`, LG: 180). Registration recovers any rotation, so a GOOD image
  laid down another way round is still a valid GOOD example. The inspection placement limit
  (`allowed_rotation_deg`, LG: 30) is unchanged.
- **Current LG bank.** 5 boards, 25 images: GOOD_001, GOOD_002, GOOD_003 (operator-verified) plus
  GOOD_901 and GOOD_902 (inferred by image matching, unverified; they may be the same physical boards as
  GOOD_001 / GOOD_002).

## 6. Product identification

- **AUTO.** The capture is registered (SIFT/RANSAC) against each product's own geometric reference. A
  product is IDENTIFIED if:
  - it registers: alignment check OK, or POSITION_INVALID (recognised but misplaced);
  - inliers ≥ `min_inliers` and inlier ratio ≥ `min_inlier_ratio`;
  - no other product is within `min_ratio_margin`.
- **Results.**
  - Otherwise UNKNOWN or AMBIGUOUS, which gives REVIEW, stored under `data/products/_unresolved/`.
  - NO_PCB and INVALID_FRAME are reported as such.
  - The reported `score` is the measured inlier ratio, not a probability.
- **Thresholds** live in `configs/inspection.yaml` and are PROVISIONAL. None has been validated against a
  second real product: no Pump Controller images exist.
- **SELECT PRODUCT.** The operator's choice is used, and the product's own registration is reused. A
  wrong selection fails the alignment gate (REVIEW); it is never compared with another product's
  references.

## 7. Gates

| Gate | Source of limits | Failure |
|---|---|---|
| Image quality: station thresholds, then product thresholds | `configs/inspection.yaml quality.blocking_flags`, capture config `quality_check` | REVIEW `IMAGE_QUALITY` |
| PCB present | engine config `pcb_presence` | REVIEW `NO_PCB` |
| Alignment: translation / rotation / scale / perspective / residual (inlier ratio, fit error, rotation, corner angles, board in frame) | variant `alignment`, product `allowed_rotation_deg`, engine `alignment_validation` | REVIEW `ALIGNMENT` (+ measured values in `alignment_error`) |
| Golden bank sufficient | `golden_bank.yaml min_boards` | REVIEW `INSUFFICIENT_GOLDEN_REFERENCES` / `NO_GOLDEN_REFERENCES` |

## 8. REFERENCE_BASELINE decision rules (`inspection_rules.yaml decision`)

For every region of the region map, the engine measures the best similarity to any GOOD board of the bank
and compares it with the limit:

- **Partly hidden region** (visible < `min_region_visible_fraction`: watermark, ignored pixels, outside the
  view): not evidence for either answer, so REVIEW `NOT_CHECKABLE`.
- **similarity ≤ limit − `uncertainty_band`:** missing → DEFECT, if `missing_component` is enabled.
- **|similarity − limit| < band:** UNCERTAIN → REVIEW.
- **More than `max_plausible_defects` regions differ:** implausible → REVIEW.
- **Every region ≥ limit + band:** GOOD. With unverified (draft) regions this requires
  `allow_good_with_unverified_regions`, and the limitation is listed in each result.

## 9. Output (`InspectionOutcome`, stored as `result.json`)

- **Identity:** `inspection_id`, `product_id`, `variant_id`, `image_id`, `board_id`.
- **Gates:** `alignment_status` (OK / FAILED / NOT_RUN), `alignment_error` (measured values),
  `quality_status` (PASS / FLAGGED / FAIL / NOT_CHECKED).
- **Decision:** `overall_decision` (GOOD / DEFECT / REVIEW / ERROR), `confidence` (null unless
  calibrated), `reasons` [{code, text}].
- **`defects[]`:** `defect_id`, `defect_type`, `confidence`, `location` (canonical mm box and centre),
  `region`, `evidence` (similarity, limit, margin, region verified?), `severity` (from the rules; null =
  undefined), `status` (unconfirmed).
- **Context:** `uncertain_regions`, `limitations`, `identification`, `reference_bank` summary, `engine`,
  `engine_status`, `evidence_files`, `image` (path, sha256, timestamp, camera).

## 10. Storage

```
data/captures/<session_id>/session.json               station session (mode, product selection, camera, failures)
data/captures/<session_id>/<image_id>.png              original, write-once, fsync, SHA-256 verified
data/captures/<session_id>/<image_id>.json             image record
data/products/<product_id>/inspections/<session_id>/<inspection_id>/
        result.json  overview.jpg  aligned.jpg  annotated.jpg  evidence.jpg  report.txt
data/products/<product_id>/inspections/results.jsonl   one line per inspection (append-only)
data/products/<product_id>/reviews/reviews.jsonl        human reviews (append-only)
data/products/_unresolved/...                           inspections whose product was not determined
<acquisition_root>/acquisition|curated|quarantine/...   data collection (LG: data/, unchanged layout)
```

- **Image record:** `image_id`, `inspection_id` (+ `inspection_ids`), `product_id`, `variant_id`,
  `physical_board_id`, `session_id`, `capture_timestamp`, `camera_id`, `resolution`, `stage`, `file_path`,
  `sha256`, `quality_status`, `inspection_status`, `decision`, `decision_confidence`, `synthetic`, …
- **Generated automatically.**
- **Mock and replay images.** A mock-camera capture is marked `synthetic: true` and never counts as data.
  Evidence images are derived; the lossless original is the stored capture.
- **IDs.**
  - `session_id` is `SYYYYMMDD-NN`, unique across all stores.
  - `image_id` is `<session_id>_C###`.
  - `inspection_id` is `<image_id>_I##`. Re-inspecting a stored image adds `_I02`; nothing is overwritten.
- **`scripts/pcb.py inspect <file>`** records files already under `data/` in place (no copy) and copies
  any other file into the capture store first.

## 11. Discovery and validation (`python scripts/pcb.py validate`)

**Sources scanned for every product:**
- its acquisition store, in all review states, including curated/ and quarantine/ copies;
- its legacy sources: Experiment 0, production raw folders, unlabelled folders such as
  `data/testing/live`, and old operator sessions;
- station captures identified as the product.

**Global checks:**
- station captures without a product;
- every image file under `data/` that belongs to no known source and is not derived data.

**Per image:** RESOLVED / UNRESOLVED / PENDING / REJECTED / SYNTHETIC / INVALID. Duplicates by SHA-256
are counted once and all paths are kept.

**Reported:**
- GOOD / DEFECT / TEST boards and images;
- duplicates, missing metadata (no physical board), invalid images, unregistered boards;
- quality flags, orphaned records and files;
- GOOD reference boards that are not in the bank.

Checksums are cached in `data/processed/checksums.json` (derived).

**Splits** are by physical board: the registry has exactly one row, and so one split, per board. The bank
uses only reference-split GOOD boards, and leave-one-board-out removes every image of a board.

## 12. Root cause: new GOOD images were not detected or used (found 2026-10-05)

1. **Rotation limit applied to the bank.** All three operator-verified GOOD boards were left out of the
   golden bank:
   - GOOD_001 and GOOD_002 were photographed at −91° relative to the reference, GOOD_003 at +179°;
   - the 30° placement limit applied while building the bank, and each image was only listed in a "skipped"
     list;
   - the bank therefore held only the two inferred, unverified boards.

   Fixed by the bank's own `max_rotation_deg`.
2. **Unreviewed GOOD captures were dropped silently.** A pending capture (S20261005-10) never appeared
   anywhere. It is now reported as PENDING by both validations.
3. **No product dimension, and LG size hard-coded.** The GOOD discovery and the classical engine assumed
   116 × 83 mm, and live and operator-station captures were never scanned. Discovery is now per product and
   covers every store.
4. **Fabricated baseline.** The old operator script built its baseline from raw, unaligned photos and filled
   it with synthetic boards labelled GOOD_001/002/003. It also stored simulated frames when the camera
   failed (2 synthetic frames under GOOD_002 in `data/inspections/sessions/S20261005_161630`). The script
   was replaced. The frames are kept and reported as SYNTHETIC.
5. **Fake leave-one-board-out.** `GoodBaseline.without_board` returned the full baseline. It now rebuilds
   from the other boards, or raises if there are none.

## 13. EXPERIMENTAL BASELINE vs FUTURE PRODUCTION INSPECTION

| | Experimental baseline (now) | Future production inspection |
|---|---|---|
| Engine | REFERENCE_BASELINE: registration + per-region similarity to GOOD boards; only `missing_component` | trained / validated backend(s) (detector, anomaly model, hybrid) registered in `BACKENDS` |
| Regions | 55 DRAFT boxes, unverified, no BOM / centroid file | regions from BOM + centroid / CAD, verified |
| Golden bank | 5 LG boards, 2 of them with unverified identity, phone camera, no fixed lighting | many verified GOOD boards per product / colour / lot, production camera + lighting |
| Thresholds | PROVISIONAL, set from the same few images | set on reference boards, validated on held-out TEST boards (by physical board) |
| Confidence | none (null) | calibrated on held-out boards |
| Camera | phone through DroidCam (watermark in the image) | production camera, fixture, lighting (still UNKNOWN) |

**Measured on the existing LG images (2026-10-05, current code):**
- **GOOD images:** REVIEW, never DEFECT.
- **R37-missing board:** REVIEW. R37 sits 0.01 below the limit, inside the 0.03 band.
- **LED4-missing board:** REVIEW. LED4 is only 69–85 % visible because the DroidCam watermark burned into
  the geometric reference covers it. Without the visibility rule this board was reported **GOOD** in 10 of
  10 images: the current Experiment 2 code also PASSes them, unlike the earlier README figures.

So with the current geometric reference no LG board can be GOOD. LED4 must first be made checkable: choose a
geometric reference without the watermark over any region and redraw or verify the regions, or mask the
watermark per GOOD example instead of from the reference.

## 14. Commands

```
python scripts/station.py                          live inspection (window, automatic capture)
python scripts/station.py --headless --captures 5  live inspection without a window
python scripts/acquire.py --product lg_indicator   data collection (GOOD / DEFECT / TEST boards)
python scripts/pcb.py products                     products, golden banks, readiness
python scripts/pcb.py boards --product lg_indicator
python scripts/pcb.py validate                     every image of every product (+ --save)
python scripts/pcb.py inspect <image> [--product AUTO|id] [--board-id GOOD_003]
python scripts/pcb.py review <inspection_id> --decision CONFIRM_GOOD --reviewer NAME
python -m pytest                                   complete test suite
```
