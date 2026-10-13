# Clean real-data collection: GOOD PCB / FAULT PCB

| | |
|---|---|
| Status | **Implemented and tested** with generated frames and a fake camera (`tests/test_clean_collection.py`, a UI smoke run). The capture check was also run read-only on the 96 archived real frames. **Not yet run with a live PCB:** on 2026-10-06 the DroidCam device opened but sent only the "Start DroidCam" screen, because the phone was not streaming. **No new image exists yet.** |
| Started | 2026-10-06 |
| Start | `python scripts/collect.py` |
| Summary | `python scripts/collection_summary.py` |
| Config | [`configs/capture/droidcam_collection.yaml`](../configs/capture/droidcam_collection.yaml), product [`configs/products/lg_indicator/product.yaml`](../configs/products/lg_indicator/product.yaml) (`storage`) |
| Code | `src/pcb_inspection/acquisition/pcb_view.py` (PCB check + crop), `session.py` (store), `ui.py` (window) |

## 1. Purpose and scope

This phase does three things only:
- **clean real data collection** from the live DroidCam camera;
- **clean PCB cropping / alignment** (the PCB alone, without background or watermark);
- **creation of the GOOD / FAULT dataset**.

It does **not** include:
- TEST PCB collection (added later);
- synthetic images, old data or the old references;
- YOLO or any model training;
- production PASS / FAIL.

Nothing here claims inspection accuracy.

## 2. Why the reset, and what was archived

The old image data was moved, never deleted, by `python scripts/archive_old_data.py --apply`:

| | |
|---|---|
| Archived | **4,852 files, of which 2,538 are images (3.5 GB)** |
| Archive folder | `archive/2026-10-06_pre_collection_reset/`, with the original project-relative paths |
| Contents list | `manifest.csv`: original path, size, SHA-256 and kind of every file; `README.md` |
| What it was | Experiment 0 phone photos, Experiment 1B GOOD / DEFECT / TEST sessions, curated / quarantine copies, station captures, live captures, old inspection results, the 2 SYNTHETIC frames, the old board registry, derived caches, the Experiment 1 / 2 reference images, run outputs and the watermarked calibration |
| Kept | All source code, configs, tests and documentation. Also kept: the experiment reports / configs, the draft component map (marked as drawn on the archived reference) and the watermark glyph template, which the check needs. `data/metadata/*.csv` were re-created as empty templates. |

**Why.** The new capture check, run read-only on the 96 old kept / station frames, gives **RECAPTURE for all 96**. In **94** of them the DroidCam watermark lies over the PCB. The old golden reference itself had the watermark over LED4. The golden bank is therefore **NOT_CONFIGURED** (`golden/golden_bank.yaml: geometric_reference: null`) until a clean reference exists. No old image may be used as data or reference again.

`pcb.py validate` sees only the new store (LG `storage.acquisition_root: data/collection`). The tests that exercise the old discovery code read the archive, read-only.

## 3. The watermark: physical fix plus pipeline

**Physical.** Place the PCB so that the DroidCam watermark text ("using droidcam.app") falls **outside the PCB**. Its position is fixed within one DroidCam streaming session but **moves when DroidCam restarts**: it was measured at two different places on 2026-10-05. So check the live overlay every time DroidCam is started.

**Pipeline.** A capture is never used as a full frame:

```
camera frame -> PCB region (board colour) -> 4 board corners -> crop / warp ONLY the PCB
             -> canonical PCB image: 116 x 83 mm at 20 px/mm = 2320 x 1660 px, board only
```

The watermark is located in every frame by matching its letter shapes. The capture check requires its box to lie outside the board outline, at least `watermark_clearance_mm` (3 mm) away. The canonical PCB image therefore contains no watermark pixel and no background. That image is what a golden-reference comparison or inspection uses; the full frame is kept as the original.

**Capture check.** It runs on every **saved** original, in full resolution, and live on the preview as guidance.

| Code | Meaning | Result |
|---|---|---|
| `NO_PCB` | no board-coloured region of plausible size (also the "Start DroidCam" screen) | RECAPTURE |
| `PCB_INCOMPLETE` | board at / beyond the image edge | RECAPTURE |
| `PCB_OUTSIDE_ROI` | board not completely inside the dashed inspection frame (ROI = 80 % of the view, board-shaped) | RECAPTURE |
| `WATERMARK_OVER_PCB` | watermark over the board or closer than 2 mm (shown in the preview as the red WATERMARK ZONE). Until 2026-10-06 the limit was 3 mm; a board 2.97 mm away was refused although the crop excludes it. | RECAPTURE |
| `PCB_TOO_SMALL` | board < 8 px/mm in the frame: the camera is too far for the board's detail (real frames: 9.3–12.7) | RECAPTURE |
| `BAD_CROP` | outline is not a 4-sided board of 116 x 83 mm proportions. Measured as: outline aspect; board-region convex hull / outline area 0.93–1.10; raw board pixels / outline ≥ 0.75. This catches a second board merged into the outline, a tilt, or a large object over the board. Parts on the board edge do not trigger it. | RECAPTURE |
| `BLURRY` | PCB-image sharpness < 300, measured at a fixed 8 px/mm so that the camera distance does not change the score. Real frames: blurred 121–242, in focus 355–1,100. The earlier score at 20 px/mm called a sharp but farther camera BLURRY. | RECAPTURE |
| `OVEREXPOSED` / `UNDEREXPOSED` | > 2 % clipped white / > 5 % black pixels on the PCB | RECAPTURE |
| `WATERMARK_NOT_LOCATED` | watermark-like text **on the PCB area** that is not clearly located: confirm by eye | REVIEW |
| (watermark not visible) | It may be outside the view, cut off at the image edge, or faint on a light background. The PCB area is then searched on its own; no match there means no watermark on the PCB. Board silkscreen scores 0.32–0.35, the threshold is 0.45, and every one of the 94 archived over-PCB watermarks matched. | OK |
| `LESS_SHARP_THAN_SESSION` | < 50 % of the session's median sharpness | REVIEW |
| `PCB_IMAGE_ERROR` | PCB image could not be made / saved (the original is still saved) | REVIEW |
| none | | **OK** |

**Automatic flow** (fixed 2026-10-06 after the first real session):
- **Wait for the PCB.** The 3 s countdown runs only while the live check is not RECAPTURE. With the board misplaced the window shows **PLACE PCB** and the reason, and nothing is captured. As soon as the board is OK, the countdown starts and the capture happens automatically.
- **Automatic retake.** A capture whose check still says RECAPTURE (e.g. the board moved during the countdown) is put in quarantine automatically, with the reason and `review.decided_by: capture_check`. It is taken again and does not count as one of the requested images. After 3 such captures in a row the session stops and says why.
- **Operator decides.** OK and REVIEW captures wait for the operator's KEEP / REJECT; a KEEP is always the operator's.
- **Re-check at start.** Captures still waiting for review are re-checked with the current limits when the application starts; the earlier result is kept in `quality.capture_check_history`.

**A RECAPTURE image cannot be KEPT.** A REVIEW image may be kept after the operator has checked it.

**First real session (2026-10-06), two false RECAPTURE / REVIEW results, now fixed:**
- **Parts on the board's top edge** notched the board-colour region, so the raw pixel fill was 0.84–0.89 where the limit was 0.92: BAD_CROP although the outline was correct. The check now uses the convex outline.
- **The watermark sat half cut off at the top image edge,** on a light screen background, so it never matched and every capture was REVIEW. Now no watermark on the PCB area gives OK.

Re-run on the 10 real captures: the 9 correctly framed ones are OK. The one with the board half out of the view and blurred is still RECAPTURE. The 96 archived contaminated frames still give WATERMARK_OVER_PCB in 94.

All limits are **PROVISIONAL**:
- the sharpness floor 25 comes from the archived frames of the same phone: in focus 33–92, motion-blurred 18–21;
- the shape limits come from whole boards in those frames: aspect error 0.01–0.03, fill 0.94.

Re-check all limits after the first real sessions.

## 4. Automatic board sequence and workflow per image

**START AUTO CAPTURE once.** After that the operator only places and removes boards:

```
PLACE GOOD_001  -- board in view; live check OK (whole PCB in the frame, outside the red WATERMARK ZONE) -->
  3 s countdown -> CAPTURE -> ... until GOOD_001 has "images per board" (10) KEPT images
REMOVE GOOD_001 -- board lifted out (no PCB in view for 1 s) -->  PLACE GOOD_002  -> ... (same for FAULT_001, ...)
```

Each image:

```
LIVE CAMERA (preview: dashed ROI, board outline, red WATERMARK ZONE, live check OK / REVIEW / RECAPTURE)
 -> 3 s countdown (only while the live check is not RECAPTURE) -> CAPTURE (a frame that arrived after the countdown)
 -> SAVE ORIGINAL   originals/<type>/<board>/<session>/images/<image_id>.png   write-once, fsync, SHA-256 verified
 -> PCB crop        originals/<type>/<board>/<session>/pcb/<image_id>_pcb.png  canonical PCB image, SHA-256
 -> capture check   OK / REVIEW / RECAPTURE + reasons, stored in the record
 -> OK        -> KEPT automatically: the PCB image -> good/<GOOD_###>/<session>/ or fault/<FAULT_###>/<session>/
                 (REJECT LAST undoes it: to quarantine, the board gets another capture)
    REVIEW    -> waits for the operator's KEEP / REJECT
    RECAPTURE -> quarantine/<type>/<board>/<session>/ automatically (original + PCB image), taken again
```

- Every capture is stored before any decision. Nothing is deleted: the original always stays in `originals/`.
- The per-board target counts **kept images of that board over all its sessions**. A board already holding 6 kept images needs only 4 more. Starting at a complete board offers to start at the next incomplete one.
- One run covers "Boards" boards (0 = until STOP), in ID order. **Label the boards** and place them in that order. The board must be removed between two IDs, so one board standing still is never recorded under two IDs. That a newly placed board is really a different board is the operator's responsibility; it is not detected.
- "Keep OK images automatically" can be switched off; then every capture waits for KEEP / REJECT as before.

## 5. Folders: `data/collection/`

```
data/collection/
  originals/good/GOOD_001/S20261006-01/session.json          session + every image record
  originals/good/GOOD_001/S20261006-01/images/GOOD_001_TOP_S20261006-01_C001.png
  originals/good/GOOD_001/S20261006-01/pcb/GOOD_001_TOP_S20261006-01_C001_pcb.png
  good/GOOD_001/S20261006-01/   GOOD_001_TOP_S20261006-01_C001_pcb.png    KEEP: the PCB image only (no watermark)
  fault/FAULT_001/S20261006-03/ ...                                       KEEP
  quarantine/good/GOOD_001/S20261006-01/ ..._C002.png + ..._C002_pcb.png  REJECT: full frame + PCB image (evidence)
```

**The GOOD / FAULT dataset contains no watermark.** Since 2026-10-06 (`review.copy_original_on_keep: false`) a KEEP copies only the PCB image: the board alone, no background, no watermark. The full camera frame, which DroidCam marks with the watermark, stays in `originals/` and is referenced by `path` / `sha256`. The first 6 images kept before this change also have their full frame in `good/`.

The watermark itself is drawn into the video by the free DroidCam edition. Software cannot recover the pixels under it; it can only keep it off the PCB, which the check guarantees. To get frames without it at all, use DroidCam's paid edition or another phone-camera app without a watermark.

Rules:
- Only `good` and `fault` exist. A `test/` folder or any stray file is reported by the audit.
- GOOD images stay grouped by **physical board** and session.
- A board ID is **required**: `GOOD_###` in GOOD sessions, `FAULT_###` in FAULT sessions.
- A FAULT board ID is refused in a GOOD session, and the other way round.

## 6. Metadata (in `session.json`, one record per image)

| Requested | Field |
|---|---|
| image_id | `image_id` (`GOOD_001_TOP_S20261006-01_C001`) |
| product_id | `product_id` (`lg_indicator`) |
| physical_board_id | `physical_board_id` (= `board_id`) |
| session_id | `session_id` |
| stage | `stage` (`after_smt_reflow` / `after_tht` / `other` + note) |
| timestamp | `timestamp` (arrival time of the captured frame, ms) |
| camera | `camera` (device name); session: index, backend, measured FPS |
| resolution | `resolution` [width, height] as delivered |
| path | `path` (original), `pcb_file` (PCB image), `saved_file` / `saved_pcb_file` (KEEP / REJECT copies) |
| sha256 | `sha256` (original), `pcb_sha256` |
| quality | `quality` (full-frame measurements + `quality.capture_check`: status, codes, messages, board corners, ROI, watermark box / distance, PCB-image sharpness / exposure) |
| review_status | `review_status` (`pending` / `kept` / `rejected`) + `review` {decided_at, reason, note} |

FAULT sessions also hold `defect_report`:
- `defect_known` (yes / unknown), `defect_type`, `component_reference`, `location`, `notes`;
- `source: operator_at_capture`, `verified: false`.

Fault types:
- missing component, damaged component, wrong component;
- broken / missing trace, solder defect, misalignment, other.

**UNKNOWN is the default and is never replaced by a guess.**

Flat table: `python scripts/collection_summary.py --csv data/processed/collection/metadata.csv`.

## 7. Before the first session

1. Fix the phone on a stand above the PCB, as perpendicular as practical. Use a plain matte background that is **not** board-coloured (not red / green / blue) and diffuse light without flash. Keep camera and light fixed for the whole collection.
2. Start DroidCam on the phone. On the PC, start the DroidCam client and press **Start**. Set the highest resolution (2560 × 1440 was measured).
3. Close every other program that uses the camera, then test it. It must end with `OK`:
   ```bash
   python scripts/check_camera.py
   ```
4. Start the collection window:
   ```bash
   python scripts/collect.py
   ```
5. Wait for `Camera: LIVE`. Lay a GOOD PCB under the camera.
   - Move the board (or the phone) until the **live check line says OK**: the whole board is inside the yellow dashed frame, and the grey WATERMARK box is outside the green board outline.
   - Use the **same orientation for every board** (e.g. the connector side always left).
   - Tap the PCB in the DroidCam app to focus.

## 8. Exact capture steps

### A + B. 10 GOOD images of board 1, then 10 of board 2 (one run)

1. Label two physically good boards `GOOD_001` and `GOOD_002`.
2. Left panel: press **GOOD PCB**. Form:
   - **First board ID** `1` (becomes `GOOD_001`);
   - **Stage**: the boards' real stage;
   - **Images per board** `10`, **Boards** `2` (or 0 = until STOP), **Interval** `3`;
   - **Lighting**: e.g. `desk LED + diffuser`;
   - "Keep OK images automatically" ticked.
3. Press **START AUTO CAPTURE**. The window shows **PLACE GOOD_001**.
4. Place GOOD_001 inside the dashed frame, **outside the red WATERMARK ZONE**, always the same way round.
   - The live check turns green (OK), and `3 → 2 → 1 → CAPTURE` runs by itself.
   - OK images are kept automatically; the counter shows `GOOD_001: kept n / 10`.
   - To get small placement differences, nudge the board slightly between captures, keeping it inside the frame.
   - **ADJUST GOOD_001** means the live check is not OK; the reason is shown under the preview.
   - A **REVIEW** image (rare) waits for your KEEP / REJECT.
   - An image you don't want: choose a reject reason, then press **REJECT LAST**. It goes to quarantine and the board is captured once more.
5. At 10 / 10 the window shows **REMOVE GOOD_001**. Lift the board out of the view.
6. **PLACE GOOD_002**: place the second board. It is captured the same way, then **REMOVE GOOD_002**, and the run ends ("Complete boards: GOOD_001, GOOD_002").

More GOOD boards: run again with First board ID `3`, or set Boards `0` and keep placing boards until you press STOP. A golden bank needs several GOOD boards; no board is "the" golden board.

### C. FAULT images (same flow)

1. Label the faulty boards `FAULT_001`, `FAULT_002`, …
2. Press **FAULT PCB**. First board ID `1`, Images per board e.g. `10`, Boards as needed. Press **START AUTO CAPTURE**.
3. **Before placing each board**, while the window shows PLACE FAULT_00n, fill in **Fault known?**
   - **YES** only if you actually know the fault. Choose the type (missing / damaged / wrong component, broken or missing trace, solder defect, misalignment, other).
   - Optionally add the component (e.g. `LED4`), a location and notes.
   - Otherwise leave **UNKNOWN**.

   The form is locked while that board is being captured and resets to UNKNOWN for the next board, so a label never carries over. It is operator information (`verified: false`), not ground truth.
4. Place, capture, remove, next, exactly as for GOOD. **Never re-label a FAULT board as GOOD**: FAULT data never enters `good/` or the GOOD bank candidates.

### D. Check the dataset

```bash
python scripts/collection_summary.py
```

It prints:
- old data archived (count);
- GOOD boards / GOOD images, FAULT boards / FAULT images (kept, plus captured / pending / rejected);
- **TEST images: 0**;
- per-board capture-check counts;
- the GOOD reference-bank candidates;
- the audit (checksums, folders, records).

Exit code 1 means the audit found an issue. `python scripts/pcb.py validate` also covers the new store.

## 9. Verify a clean reference image

```bash
python scripts/collection_summary.py --verify-reference GOOD_001_TOP_S20261006-01_C001
```

It re-runs the check on the kept original in full resolution and reports PASS / FAIL for:
- GOOD and kept;
- original and PCB image intact (SHA-256);
- **full PCB visible**;
- PCB inside the ROI;
- **no watermark over the PCB / components** (distance in mm);
- **correct crop** (outline aspect, fill, PCB-image size);
- **acceptable quality** (sharpness, clipped pixels);
- **LED4 not covered by the watermark.**

It also writes a sheet to `data/processed/collection/reference_check/<image_id>.jpg`: the frame with ROI, board outline and watermark box, next to the PCB image with a 10 mm grid.

**LED4 presence is not detected automatically.** No verified region map for the new reference exists yet. The automatic checks guarantee that the whole board is in view and the watermark lies outside it, so nothing covers LED4. That LED4 is actually visible and present is confirmed **by eye** on the sheet.

**Next, not done in this phase:**
1. Choose a clean reference (it must pass the above).
2. Redraw / verify the region map on it (`define_components.py`).
3. Set `golden_bank.yaml geometric_reference` / `min_boards`.
4. Connect the golden bank to the KEPT GOOD PCB images (`AcquisitionStore.good_bank_candidates()`: GOOD boards only, grouped by board, never FAULT).

## 9b. Testing a PCB (live camera, EXPERIMENTAL)

`python scripts/station.py` opens the camera. When a board lies still in view it captures automatically, identifies the product and shows **GOOD / DEFECT (part, location) / REVIEW (reason)**. Every capture is stored first in `data/captures/`.

**Set up on 2026-10-06** (`python scripts/setup_reference.py --reference GOOD_001_TOP_S20261006-03_C007 --apply`):
- **Reference:** `GOOD_001_TOP_S20261006-03_C007` passed every automatic clean-reference check, and LED4 is visible on its sheet.
- **Registry:** `data/collection/registry/boards.csv` holds the capture declarations. GOOD_001 and GOOD_003 form the reference split. FAULT_001/002/003/005 form the validation split, with the reported faults (unverified).
- **Component map:** the 55 draft boxes are re-attached unchanged, in mm, to the new reference (`configs/products/lg_indicator/regions/components.yaml`); checked on the sheet `..._components.jpg`. They are still DRAFT / NOT VERIFIED.
- **Orientation check:** the collection window now refuses a board turned 180° (`PCB_ROTATED`).

**Same way round only.**
- A board turned 180° is registered correctly, but lit from the other side its parts look different. Upright GOOD_001 compared with rotated GOOD_003 gave 30 of 55 parts "different", and a missing LED4 matched a rotated GOOD example.
- So only GOOD images placed like the reference are comparison examples (`golden_bank.yaml max_rotation_deg: 30`). An inspected board turned more than 30° gets REVIEW "place it the same way round".
- **Status now:** GOOD_003 and GOOD_001 sessions 04–05 lie rotated, which leaves 1 upright GOOD board, so the bank is INSUFFICIENT and every result is REVIEW. **Capture one more GOOD board the same way round** to make it READY (e.g. GOOD_003 again: First board ID `3`, Images per board `20`).

**Measured on the collected FAULT boards** (read-only, with the rotated examples still in the bank). This is NOT a validation:

| Board (reported fault) | Results over 10 images |
|---|---|
| FAULT_003 (LED4 missing) | 2 DEFECT LED4, 8 REVIEW (LED4 uncertain) |
| FAULT_005 (LED4 missing) | 2 DEFECT LED4, 2 REVIEW, 6 GOOD |
| FAULT_001 (R37 replaced by a diode) | 1 DEFECT (Q9, wrong part), 4 REVIEW, 5 GOOD |
| FAULT_002 (LED10 missing) | 7 GOOD, 2 REVIEW, 1 DEFECT (Q9). LED10 is not in the draft map, so it cannot be found |

**Limitations:**
- The method detects only **missing** parts (`missing_component`). A wrong part with a similar outline, such as R37 → diode, passes (similarity 0.80 against limit 0.735).
- The limit 0.735 comes from the archived bank. Per-part limits need ≥ 5 GOOD boards.
- The component map is a draft with only 55 parts. It does not include LED10 and other LEDs, and nobody has verified it.
- Results must be validated on **TEST boards** (unseen, not used for any setting) before they mean anything.

## 10. Limitations and open points

- **Not yet run with the live phone stream**, and no new image exists: every count is 0 until the first session.
- All check limits are PROVISIONAL (section 3).
- Board detection relies on the board colour. A board-coloured background or object breaks it; the check then says NO_PCB / BAD_CROP.
- The 180° orientation of the PCB image follows the placement. Keep one placement orientation; a later registration step handles residual rotation.
- The watermark template was measured at 2560 × 1440. At other resolutions it is scaled, which is not verified. If the watermark is not found, the check says REVIEW and never OK.
- "Broken / missing trace" is one operator choice. The project taxonomy has `broken_trace` and `missing_trace` separately, so it is stored with `taxonomy_defect_type: null`; write which one in the notes.
