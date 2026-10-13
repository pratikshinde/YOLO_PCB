# PCB COMPONENT DEFECT DETECTION — MASTER PROMPT (state 2026-10-07)

You are working on an existing project: an industry-oriented visual inspection system for ONE fixed PCB design
(LG Indicator, 116 x 83 mm, top side, board colour may be red / green / blue). Project root:
D:\Projects\YOLO_PCB_Componet (no git; never create a new repo or move to another folder).

## 1. Ground rules (always)

- Real data only: never fabricate, generate or download PCB images, labels, predictions or model weights.
- Unknown facts stay null / UNKNOWN. Never guess labels; operator fault reports are "unverified".
- Never delete captures: originals are write-once with SHA-256; rejected images go to quarantine.
- Old pre-2026-10-06 data is archived in archive/2026-10-06_pre_collection_reset/ and must not be used
  (DroidCam watermark over the PCB, LED4 hidden).
- Claim only what was implemented AND tested. Do not claim inspection accuracy until it is measured on unseen boards.
- Keep a REVIEW result for uncertain cases; never force PASS / FAIL.
- Count data by PHYSICAL BOARDS and DEFECT INSTANCES, not by images. Split train / val / test by physical board.

## 2. What already exists (built and tested)

- Clean collection: `python scripts/collect.py` (live DroidCam camera, index 1, 2560 x 1440).
  - AUTO board sequence: START once; board N captured automatically until "images per board" are KEPT, then
    "REMOVE board", then "PLACE board N+1". GOOD / FAULT only (TEST later).
  - Every capture: 3 s countdown -> original saved first -> canonical PCB image (board only, 116 x 83 mm at
    20 px/mm, no background, no watermark) -> capture check -> OK kept automatically / REVIEW waits for the
    operator / RECAPTURE quarantined automatically and retaken. "REJECT LAST" undoes an automatic KEEP.
  - Capture check: complete PCB, inside ROI, watermark >= 2 mm from the board (red WATERMARK ZONE in the preview),
    clean board outline, focus (sharpness at fixed 8 px/mm >= 300), exposure, board scale >= 8 px/mm, and
    PCB_ROTATED (board turned 180 deg vs the reference).
  - Folders: data/collection/originals/<good|fault>/<board>/<session>/ (session.json, images/, pcb/),
    good/<GOOD_###>/, fault/<FAULT_###>/ (KEEP: PCB image only), quarantine/.
  - Metadata per image: image_id, product_id, physical_board_id, session_id, stage, timestamp, camera,
    resolution, path, sha256, quality (+ capture_check), review_status; FAULT sessions: defect_report.
- `python scripts/collection_summary.py` (counts, audit), `--verify-reference best|IMAGE_ID` (clean reference check).
- `python scripts/setup_reference.py --reference GOOD_001_TOP_S20261006-03_C007 --apply` (reference, board
  registry data/collection/registry/boards.csv, component map, configs).
- Live test: `python scripts/station.py` (auto capture when the board is still -> product identification ->
  EXPERIMENTAL REFERENCE_BASELINE engine: alignment (SIFT + homography, ~0.05 mm) + per-component similarity
  to GOOD boards -> GOOD / DEFECT / REVIEW). Detects only "missing" parts.
- Board-level splitting, leakage checks, dataset validation (`python scripts/pcb.py validate`), 349 passing tests.

## 3. Current data and status

- GOOD_001: 12 kept (9 placed correctly, 3 rotated 180 deg). GOOD_003: 10 kept, ALL rotated 180 deg.
- FAULT_001 (R37 replaced by a diode), FAULT_002 (LED10 missing), FAULT_003 (LED4 missing),
  FAULT_005 (LED4 missing): 10 kept each. FAULT_004 and GOOD_002: no images.
- Reference: GOOD_001_TOP_S20261006-03_C007 (passed all automatic checks; LED4 visible).
- Golden bank: INSUFFICIENT (only 1 board placed the same way round; rotated GOOD images are skipped as
  comparison examples because directional light makes them look different). Needs >= 2 correctly placed GOOD boards.
- Component map: 55 DRAFT boxes, NOT VERIFIED, missing many parts (e.g. LED10). No BOM / centroid file yet.
- Measured (not a validation): LED4-missing boards partly found (DEFECT / REVIEW), many misses; R37 -> diode
  not found (method detects only missing parts); LED10 not in the map.
- Open: 3 FAULT sessions recorded UNKNOWN after restarts (labels known per board) - fix only if the user confirms.

## 4. Target architecture (V1)

Camera -> board stable? -> PCB detection -> alignment / perspective correction to the canonical PCB frame (mm)
-> golden reference (expected layout) -> YOLO on the aligned PCB image -> actual layout -> compare -> defects
-> PASS / FAIL / REVIEW -> overlay + explanation + coordinates (e.g. "R12 MISALIGNED, expected (x, y) mm,
actual (x, y) mm, offset (+dx, +dy) mm").

Key design decisions (corrections to the original plan):
1. YOLO detects COMPONENT TYPES (resistor, capacitor, LED, diode, transistor, IC, switch, connector; optionally
   "damaged"), on every part of every image - not only defect boxes. The inspection engine derives defects:
   - missing: no detection at the expected position;
   - misaligned: centre offset / rotation beyond a per-part tolerance in MM (canonical frame, not raw pixels);
   - wrong component: detected type != expected type;
   - damaged: "damaged" class or an anomaly score.
2. Golden reference = BOM + centroid (pick-and-place) file: refdes, type, value, package, DNP, X / Y (mm),
   rotation, side, board origin. Replaces the draft map.
3. Resolution: board must fill ~80 % of the frame; YOLO runs on the aligned PCB image at imgsz >= 1280 or on tiles
   (0805 parts are only ~6 x 4 px at 640 px on the full frame).
4. Rotated parts need oriented boxes (YOLO-OBB). YOLO box precision allows only gross misalignment (~>= 0.3-0.5 mm).
5. Wrong-value limits: resistor value only via tiny printed code (borderline, OCR on high-res crops); ceramic
   capacitors have no marking -> visually impossible (electrical test). Type / colour swaps are visible.
6. Damage: anomaly detection trained on GOOD only (e.g. PatchCore per component ROI) is likely better than
   YOLO with few examples.
7. V2 (solder, orientation, pins, traces, cracks) only after V1 is validated; then ROI / high-res crops.

## 5. Data preparation (the most important step)

1. Fix the setup BEFORE any dataset image: phone on a fixed stand, straight above; board ~80 % of the frame;
   2560 x 1440; focus and exposure locked; diffuse fixed light (no window, no flash); matte background that is not
   red / green / blue; a corner stop so every board lies in the same place and THE SAME WAY ROUND as GOOD_001;
   watermark outside the board (or DroidCam paid). Never change the setup afterwards.
2. Documents: request BOM, centroid file (with board origin) and optionally Gerber (outline + silkscreen) from the
   PCB designer / assembler.
3. Boards: label each with a sticker ID (GOOD_###, FAULT_###), one ID per physical board for life. Photograph a
   board as GOOD before reworking it; the reworked board gets a FAULT ID ("reworked from GOOD_00x").
   Keep 2-3 GOOD and 2-3 FAULT boards as an untouched TEST set.
4. Rework log (ground truth, one row per defect): board, part, defect type, detail (e.g. "1 mm left"), who, date.
5. Capture with collect.py: 5-10 images per board / state, nudge the board slightly between captures, same
   orientation, real stage, fault info (or "Other" + parts in Notes).
6. After each capture day: collection_summary.py (audit consistent, counts complete), spot-check PCB images.
7. Don't: mix pre-setup images, put one board in train and test, take 50 photos of one board, guess labels,
   use synthetic or downloaded images.

## 6. How many images

- Pilot: ~10 GOOD boards (50-100 images); 15-20 instances each of missing / misaligned / wrong / damaged at
  different parts and locations, 3-5 images each (~60 per class); ~300 images from ~10 GOOD boards and
  60-80 defects (several defects per board allowed).
- Final V1: 20-30 GOOD boards (150-300 images); 50-100 instances per defect type (150-400 images each);
  test set 3-5 GOOD + 3-5 FAULT unseen boards.

## 7. One-board pilot (allowed for building / proving the pipeline, not for the final dataset)

- One sacrificial board, one FAULT ID (e.g. FAULT_010). State 0: complete board, 10 images.
- Cycles (12-15): remove 2-3 parts of different types / places -> 5 images; solder them back (some shifted or
  rotated = misaligned, some swapped = wrong) and remove new ones -> 5 images; repeat. Keep 1-3 defects per state
  (realistic). Clean flux (IPA) after every rework so the model does not learn "flux = defect".
- Log every state (state, session, missing, misaligned, wrong, notes).
- Train on the rework board + GOOD_001; TEST on boards never seen: GOOD_003 (recaptured correctly),
  FAULT_001 / 002 / 003 / 005. Results on the training board itself are not evidence.

## 8. Labelling

1. Golden map from BOM + centroid (or every part drawn once in CVAT on the reference image).
2. Auto-label: align each image to the reference, transfer all golden boxes -> YOLO .txt per image
   (class cx cy w h, normalised 0-1). For rework states: golden map minus removed parts, misaligned / wrong parts
   from the log.
3. Correct only defect boxes in CVAT (delete missing, move misaligned, change class of wrong, add damaged).
4. Export YOLO format: images/{train,val,test}, labels/{train,val,test}, split by physical board.

## 9. Next tasks (in order)

1. Fix the setup; request BOM + centroid file; prepare boards and the rework log.
2. Recapture GOOD_003 correctly placed (or another GOOD board) -> golden bank READY.
3. Build: golden map import (BOM + centroid -> canonical mm), auto-label transfer, CVAT export / import,
   YOLO dataset export with board-level split.
4. Pilot capture (one-board rework cycles + GOOD boards); label; train YOLO (component types, imgsz >= 1280 / tiles).
5. Build the comparison engine (missing / misaligned / wrong / damaged, mm tolerances, REVIEW) and the live UI
   (overlay, component ID, defect type, confidence, expected / actual / offset, explanation).
6. Evaluate on unseen boards; analyse failures; expand data; only then V2.
