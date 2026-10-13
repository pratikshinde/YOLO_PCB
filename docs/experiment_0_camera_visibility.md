# Experiment 0: Camera Visibility / Image Acquisition (LG Indicator)

| | |
|---|---|
| Status | **Ready to run** with a mobile phone. The ingest workflow has been tested on copies of the first 4 real photos (see section 8). |
| Date | 2026-10-02 |
| Camera | **Mobile phone: prototype only** (mentor, 2026-10-02). Not the production camera. |
| Mode | **Development / dataset mode.** This is not the operator's inspection workflow ([requirements.md](requirements.md), section 2). |
| Configuration | [`configs/capture/mobile_phone_e0.yaml`](../configs/capture/mobile_phone_e0.yaml) |
| Checklist | [experiment_0_capture_checklist.md](experiment_0_capture_checklist.md) |

## Objective

**The objective is NOT model accuracy.** No model is trained.

> Can the camera capture enough visual information from the real LG Indicator PCB to support automated visual inspection?

This uses real boards only. No artificial defects, no synthetic images, no annotations. The results show what a simple camera can and cannot see; they are not production-ready imaging.

## Workflow: camera-first

```
take photo  →  copy original into data/experiment_0/inbox/  →  python scripts/e0_ingest.py  →  review flagged photos at full resolution
```

There is **no manual CSV entry and no typing per image.** The ingest does the bookkeeping:

| Done automatically | How |
|---|---|
| Session ID | One session per capture date and ingest run, e.g. `S20261002-01` |
| Image ID | Per photo, in capture order, e.g. `S20261002-01_C001` |
| File naming and filing | A **byte-identical copy** (checksum verified, timestamps kept) goes to `images/<board_id or unassigned>/<session_id>/<image_id>.jpg`. The staged file is then parked in `inbox/_ingested/<session_id>/`; nothing is deleted. |
| Capture time | From EXIF (`DateTimeOriginal`); the file time if EXIF is missing, which is recorded |
| Camera / phone model | From EXIF, e.g. `Xiaomi Redmi 12 5G` |
| Image size | Width × height, after the phone's rotation flag |
| Digital zoom, flash, exposure time, ISO, focal length | From EXIF, where the phone records them |
| Original file name, checksum | Recorded; the checksum later proves the image was not edited |
| PCB model / variant | From the active inspection profile (`LG Indicator`, `1`) |
| Board colour | **Estimated from the image** (red / green / blue / other); nobody selects it |
| Image quality | Checked immediately (section 6) |

**Optional context, given once per ingest run, never per image.** Anything omitted is recorded as unknown:
```bash
python scripts/e0_ingest.py --stage after_tht --distance-mm 300 --lighting "ceiling LED"
```
- `--stage`: `after_smt_reflow` or `after_tht`. In the final system the stage will come from the recipe, the station or the PLC / MES; nobody types it per board.
- `--distance-mm`: lens-to-board distance, if measured.
- `--lighting`: a short description.
- `--board GOOD_001`: board ID for photos directly in the inbox, if one physical board is photographed in this run. Board sub-folders (`inbox/GOOD_002/`) take precedence (see section 4).
- `--dry-run`: show what would happen without moving anything.

## 1. Taking the photo

| Item | Do |
|---|---|
| Camera | Main 1× rear camera, highest resolution, normal photo mode |
| Avoid | Digital zoom (move the phone instead), portrait / background blur, filters, flash, "AI scene" modes |
| Format | JPEG (on an iPhone: "Most Compatible", not HEIC) |
| Position | As top-down as practical; phone held still (a stand or fixed support is best); the **whole board plus a margin** visible |
| Background | Plain, matte, **white or light grey**, never blue, red, green or wood. In the first photos, a wooden plank confused the earlier detector. |
| Focus | Tap on the area with the **small resistors**; check them at 100 % zoom after taking the photo |
| Lighting | Use what is there, keep it constant during a session, and avoid strong glare spots. Production lighting is **not** solved by this experiment. |
| Board | Top side up, held by its edges, no fingerprints on the top side |

## 2. Getting the photo onto the computer

- Copy the **original** file by USB cable, or another method that keeps the original. **Never** use messaging apps (WhatsApp, Telegram and similar): they recompress and shrink photos.
- Put it into `data/experiment_0/inbox/`. **Do not rename or edit it.**

## 3. What to photograph

| Step | Boards | Photos |
|---|---|---|
| First capture | One verified-good LG Indicator board | 1 photo, ingested and reviewed (section 7) |
| Repeatability | The same board | About 10 photos, lifting and re-placing the board between them (one run, `--board GOOD_001`) |
| Variation | A few more good boards, including each available colour and each available stage | 2–3 photos each |
| Defects | Every available real defective board **with known defect information** | 2–3 photos each, plus the defect record (section 5) |

These numbers are proposals for a visibility study, not a dataset size. **No artificial defects.**

## 4. Board identity: optional, not typed per image

- For **visibility checks**, no board ID is needed. Photos dropped straight into `inbox/` simply have none.
- A board ID is needed **only** when the same physical board is photographed several times, or for a defective board with ground truth. Tag it without typing per image:
  - **Either** put that board's photos into a sub-folder named after it: `inbox/GOOD_001/`, `inbox/DEFECT_001/`;
  - **or** pass `--board GOOD_001` when only that board's photos are in the inbox.
- **Prefixes:** `GOOD_###` = from the known-good pool, `DEFECT_###` = from the rejected boards. The ID stays with the physical board for life.
- **Never use the phone's file names as board identity.**
- Image IDs do not contain the board ID, so a board ID can be added or corrected later in `captures.csv` without renaming any file.
- In the final system, board identity comes from an automatic inspection ID (or a barcode / serial, if one is added later), never from operator typing.

## 5. Ground truth (only where it is needed)

| File | When | Content |
|---|---|---|
| `metadata/boards.csv` | Optional; needed for boards with a board ID | `board_id`, `condition` (good / defective), `colour`, `verified_by`, `ground_truth_source`, `notes` |
| `metadata/defects.csv` | **For every defective board** | `defect_id` (DEF_001 …), `board_id`, `defect_type` (taxonomy), `defect_location`, `defect_description`, `ground_truth_source`, `verified_by`, `visible_in_images` (after review: yes / partial / no), optional `defect_box_mm` (`x0,y0,x1,y1` in board mm from the top-left corner in the reference orientation; read off the experiment's `reference_mm_grid.jpg`; needed for defect-overlap ground truth in Experiment 1B) |

This is the only manual data entry, and it describes the physical board, not the photos. Defect types: `missing_component`, `damaged_component`, `misaligned_component`, `wrong_component`, `wrong_polarity`, `missing_or_damaged_lead`, `broken_trace`, `missing_trace`, `damaged_pcb`, `solder_defect`, `foreign_object`, `other_anomaly`.

## 6. What the software checks

At ingest, and again with `python scripts/check_e0_images.py` (which re-checks everything, e.g. after thresholds change):

| Check | Flag / note |
|---|---|
| Image exists and is readable | `missing`, `unreadable` |
| Supported format (JPEG / PNG / TIFF) | HEIC etc. are skipped and left in the inbox |
| Duplicate photo (same bytes as one already ingested) | Skipped |
| Resolution | Reported (no minimum yet: depends on PCB-29) |
| Sharpness | `less_sharp_than_other_captures`: below 50 % of the session's median (PROVISIONAL) |
| Exposure / glare on the board | `glare_or_overexposed`, `underexposed` (PROVISIONAL limits) |
| Whole board visible, not cropped | `board_not_detected`, `board_at_image_edge_possible_crop` |
| Board shape vs 116 × 83 mm | `board_shape_mismatch` (tilted, partly hidden, or detection wrong) |
| Board colour | Estimated: red / green / blue / other |
| Scale | Approximate px/mm, from the board's known 116 mm length |
| Orientation | Portrait / landscape, after the phone's rotation flag |
| Digital zoom used, flash fired, no EXIF time | Notes |
| Log integrity | Image IDs, paths, checksums (detects edited files), profile values, photos not yet ingested |

**How to read the results:**
- A flag means **look at that photo at full resolution and consider re-taking it**. It is not a verdict on the board.
- Board detection is a heuristic that finds the board by its own colour.
- PROVISIONAL limits are adjusted after more real photos have been reviewed.

## 7. Review at full resolution (the actual Experiment 0 result)

Open the ingested photos at 100 % zoom (with the operator for defective boards). Answer the following and write the answers into the Experiment 0 record (`experiments/EXP-000_camera-visibility/record.yaml`, from the template):

| # | Question |
|---|---|
| V1 | Is the whole board in the photo, with margin? |
| V2 | How many px/mm does the phone give (see the report)? |
| V3 | Are all components visible, or do tall parts (~12 mm) hide neighbours? |
| V4 | **Can each small resistor be seen, with its ends and marking?** |
| V5 | Are outer traces visible (per board colour)? |
| V6 | Which solder joints are visible from the top? |
| V7 | Sharp in the centre and the corners? |
| V8–V10 | Even lighting? Glare? Shadows from tall parts? |
| V11 | Repeatability: how much do position and brightness vary over re-placed photos of one board? |
| V12 | Which features could anchor alignment (corners, holes, large parts)? |
| V13 | **Is each recorded defect visible?** Write the answer into `visible_in_images`. |
| V14 | Does any of this change with board colour? |

When done, set `progress.experiment_0_images_collected` and `progress.experiment_0_images_inspected` to `true` in `configs/project.yaml`. `python scripts/check_readiness.py` then shows the next blocked gate.

## 8. First test on real photos (2026-10-02)

The 4 uploaded photos were processed as **copies**; the originals were not touched.
- **Ingest:** 4 photos ingested with no typing. Phone: Xiaomi Redmi 12 5G; 6144 × 8160 px; capture times read from EXIF.
- **Checks:**
  - Colour estimated as red.
  - Board fully in frame in all four.
  - Scale about 48–63 px/mm; it varies, so the camera distance changed between photos.
  - No quality flags.
- **First visual finding:** small resistors and their "222" markings are visible at 100 %, somewhat soft.

Open questions about these photos:
- Are they of one board or several? Three were taken 2 seconds apart.
- What is their inspection stage?

## 9. No data splits yet

No train / validation / test splits are made now. When they are made, they are made **by physical board ID**, with all photos of one board in the same split (`src/pcb_inspection/dataset/splitting.py`).

## 10. Outputs and decisions

| Outcome | Next step |
|---|---|
| Small resistors and known defects clearly visible | Use the measured px/mm as evidence for the production camera specification; collect more real boards |
| Partly visible | Try a closer distance, better lighting, or close-up views; specify a better camera |
| Not visible | Production imaging needs a dedicated camera / lens / lighting (or tiling); document the defect types out of reach |

**No model training starts** until real images have been collected **and inspected**.

## 11. Out of scope
Model training of any kind, accuracy metrics, annotation, artificial or synthetic defects, the final PASS / FAIL model, the production UI.

## 12. Later
Repeat Experiment 0 with the production-candidate camera, lens and lighting, under the stricter protocol in [data_collection_plan.md](data_collection_plan.md), section 4. The phone results serve as the baseline.
