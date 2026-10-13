# Experiment 2: GOOD reference + alignment + component-level comparison

| | |
|---|---|
| Status | **Prototype runs end to end. Not reliable yet:** with the current data it names a missing part correctly in 13 of 18 checkable DEFECT images, misses the TEST image's defect, and never named a wrong part. |
| Re-check 2026-10-05 (later) | With the current code and config (template-located watermark), the 10 "LED4 missing" images (S20261005-03) all PASS: LED4 similarity 0.78-0.80 > 0.735. LED4 is only 69-85 % visible because the watermark burned into the reference image lies over it. The product station therefore treats partly hidden regions as REVIEW ([docs/inspection_architecture.md](../../docs/inspection_architecture.md), section 13). |
| Date | 2026-10-05 |
| Question | Can missing components be named from GOOD images + alignment + known component positions + image comparison, without training anything? |
| Method | Deterministic image processing. Alignment, canonical frame and parameters are those of [Experiment 1](../experiment_001/README.md). |
| Command | `python scripts/inspect_pcb.py` (images in `data/testing/input/`) or `python scripts/inspect_pcb.py <image or folder>` |
| Config | [`config.yaml`](config.yaml) · component map [`components.yaml`](components.yaml) (**DRAFT, unverified**) · watermark mask [`droidcam_watermark_mask.png`](droidcam_watermark_mask.png) |
| Code | `src/pcb_inspection/comparison/components.py`, `comparison/board_inspection.py` |
| Outputs | `runs/<image>/`: `report.txt`, `result.json`, `annotated.jpg` (green present / red missing / grey not checked), `evidence.jpg` (GOOD crop next to test crop); [`runs/summary.csv`](runs/summary.csv) |

## 1. Data (inspected 2026-10-05)

| Set | Session | Images | Operator report | Physical board (inferred*) |
|---|---|---|---|---|
| GOOD | S20261005-01 | 10 (C001 = reference) | verified good | **A**: C001–C006 · **B**: C007–C010 |
| DEFECT | S20261005-02 | 10 | "R37 missing" (unverified) | **C** |
| DEFECT | S20261005-03 | 10 | "LED4 missing" (unverified) | **D** |
| TEST | S20261005-04 | 1 | none (unlabelled) | **C**, i.e. the S02 board: **not an unseen board** |

- **Capture:** all images are 2560 × 1440 (DroidCam, Redmi 12 5G), about 13.5 px/mm, all quality PASS.
- *No board IDs were recorded. The grouping comes from SIFT matching: same board gives an inlier ratio of 0.82–0.90, different boards 0.54–0.73. It is evidence, not a record.
- **Marking lots:** the resistors on boards A, B and D carry 3-digit codes ("222", "103"); board C carries 4-digit codes ("2201", "1022"), so it is probably a different component lot.
- **Stage:** every session says AFTER_SMT, but the boards carry through-hole parts. This is probably the default value left unchanged; confirm the real stage.

## 2. Component positions

- **No positions existed in the project:** profile `component_map: null`, and no BOM or centroid file (PCB-11, PCB-12 UNKNOWN).
- **The designators are printed on the board's silkscreen.**
- **On request, a draft map of 55 parts was made**, each part with a printed designator:
  - each box = a part found automatically on the GOOD reference (pixels that are not solder mask, thin silkscreen strokes removed);
  - each name = the designator printed next to it, read visually.
- **Every box is `source: assistant_draft, verified: false`**, and every report says so.
- **Verify the boxes** with `python scripts/define_components.py`: click a box, check it, press V.
- **Not mapped:**
  - parts printed with a value only ("D_4148", "100nF", "R_2K2" columns);
  - unpopulated footprints (LED1, LED9, LED17, R44, D16, R50, R32, Q11, D14);
  - hidden labels and fragmented detections;
  - red LEDs (not checkable by the first, colour-based rule; the final rule could check them).
- **A PASS means: all components *in the map* were found.**

## 3. Pipeline

```
test image → board segmentation → SIFT/RANSAC homography to the GOOD reference → canonical 116 x 83 mm frame
           → per component box: best similarity to ANY GOOD example → MISSING / present / not checked
           → PASS / FAIL / REVIEW + named components + evidence crops
```

**Similarity.**
- Normalised cross-correlation between the test and each GOOD example's box. It is insensitive to brightness and contrast.
- Searched within ±0.5 mm.
- Both images are blurred by 0.2 mm first, so the coarse layout counts (body between terminals), not the marking text.
- Score = difference = 1 − similarity. MISSING when the similarity is below 0.735 (difference above 0.265).

**GOOD examples ("gallery").**
- The reference plus all other curated GOOD images (boards A and B), aligned once and cached in `data/processed/experiment_002/`.
- A GOOD image is never compared with itself.

**Not checked.**
- A box less than 60 % comparable is not checked: either it lies outside the aligned area, or the DroidCam watermark letters cover it.
- The watermark letter mask was measured from the 30 GOOD + DEFECT frames (pixels bright in every frame); TEST was excluded.

**Status.**
- FAIL if any part is MISSING.
- REVIEW if alignment failed or a part could not be checked.
- PASS otherwise.

The status is an experiment output, not a production decision (the decision-engine thresholds stay null).

## 4. Runs

### Run 1: rule fixed before seeing DEFECT/TEST: "bare board appears"

**Rule:** MISSING if the solder-mask share of the box grows by ≥ 0.25 **and** the part no longer matches the single reference (NCC < 0.90).

**Threshold choice:**
- 0.25 was set a priori.
- 0.90 was set from the GOOD images only: lighting raised the bare-board share of present parts up to 0.27, but they kept NCC ≥ 0.925.

| Set | Result |
|---|---|
| GOOD (9) | 8 PASS, 1 REVIEW (LED4 under the watermark), 0 FAIL |
| DEFECT S02 R37 (10) | **0 detected** (10 PASS) |
| DEFECT S03 LED4 (10) | **0 detected** (8 PASS, 2 REVIEW) |
| TEST (1) | PASS: **defect missed** |

**Why it failed:**
- **Bare-board share:** an empty footprint on this board shows its **pads and solder**, not solder mask. The share rose only +0.07 to +0.15 at R37 and ~0 at LED4.
- **Similarity:**
  - The missing parts did drop clearly (R37 0.54, LED4 0.60).
  - **Present parts on other boards dropped too.** Different marking text: e.g. R21 0.62 on board C. Different placement: **Q9 0.41 on board D**, because Q9 sits offset on the *reference* board A only.

The record is kept in [`runs_v1_bare_board/`](runs_v1_bare_board/) (JSON / CSV; images removed).

### Run 2: final rule (v2)

**v2 was chosen AFTER run 1 failed, with exploratory studies on these same images.** Its results on DEFECT/TEST are therefore **not an independent validation**.

**Changes from run 1:**
- similarity instead of bare board as the decision;
- 0.2 mm blur against marking text;
- multi-image GOOD gallery against a single board's placement quirks.

**How the limit was set:** limit = lowest similarity seen between GOOD boards (images of board A scored against board B examples and vice versa: 0.785) minus a 0.05 margin = 0.735. Only GOOD images were used.

| Set | Result |
|---|---|
| GOOD (9) | 8 PASS, 1 REVIEW (C006: LED4 under the watermark), **0 FAIL** |
| DEFECT S02 "R37 missing" (10) | **9 FAIL naming exactly R37** (difference 0.27–0.28); C002 PASS: missed, 0.264 vs limit 0.265 |
| DEFECT S03 "LED4 missing" (10) | **4 FAIL naming exactly LED4** (0.28–0.30); 2 REVIEW (LED4 under the watermark); **4 missed** (PASS) |
| TEST (1) | **PASS: defect missed.** R37 was the most different part on that board (0.253) but stayed under the limit |
| Wrong part named | **none** in 30 images |

**Caveats on these numbers:**
- **Margin:** present parts on boards C/D reach differences up to 0.25 (Q9 on D, R60 on C). Missing parts sit at 0.25–0.30. The margin between them is ≈ 0.03, too small to trust.
- **GOOD figures are optimistic:** GOOD images are compared with other images of the same board in the gallery.

Example (the TEST image shows the same board as S02 C001):

```
================================================
PCB INSPECTION RESULT
================================================
Image:      DEFECT_UNASSIGNED_TOP_S20261005-02_C001.png
Reference:  GOOD_UNASSIGNED_TOP_S20261005-01_C001 + 9 more GOOD image(s)
Alignment:  OK (644 inliers, fit error 0.08 mm)
Components: 55 of 55 in the component map checked
            component map: 55 of 55 boxes are an UNVERIFIED draft
Status: FAIL
Missing/Defective Components:
1. R37
Confidence/Score (difference from the closest GOOD example; FAIL above 0.265):
R37: 0.27   (similarity 0.73; box not verified)
================================================
```

## 5. Findings

1. **Alignment is not the problem.** All 30 images register to the GOOD reference with a fit error of 0.07–0.08 mm, using Experiment 1 code unchanged.
2. **On this board, a missing part shows bare pads and solder, not solder mask.** Colour-based presence rules do not work here.
3. **Board-to-board variation is larger than expected and dominates the comparison.** It has three sources:
   - **marking text** differs between component lots;
   - **placement** varies per board (Q9 on board A);
   - **tall parts** show parallax (LEDs).

   A single golden board is not enough. Even two GOOD boards leave only a ≈ 0.03 margin, because the defect board C comes from a lot that no GOOD board represents.
4. **R37 on board C is probably not just "missing".** A light-beige body sits on the pads (see `runs/DEFECT_…-02_C001/evidence.jpg`). It looks like a resistor mounted upside down (white ceramic underside) or a different part. **The operator should check the board.** The detector reports it as missing/defective.
5. **The TEST image shows the same board as DEFECT S02.** It cannot measure performance on unseen boards. Future TEST boards need board IDs, and must not be boards used in GOOD/DEFECT sessions.
6. **The DroidCam free-edition watermark** covers part of the LED3/R49/LED4 area. LED4 is therefore not checkable in 3 of 30 images.

## 6. Missing information and the minimum next steps

1. **Verify the component map** (`python scripts/define_components.py`, about 10–15 min for 55 boxes). Add the remaining parts with their designators. The parts printed only with values need the **BOM / centroid file** (PCB-11, PCB-12), which would give all positions at once.
2. **Capture more GOOD boards, with board IDs.** At least 5–10 different boards, including **every component lot** (both the 3-digit and 4-digit marking styles). This is what the gallery needs to learn real board-to-board variation.
3. **Fix the capture:**
   - phone mounted perpendicular above the board;
   - same distance and diffuse light every time;
   - no watermark (e.g. DroidCam paid edition, or another capture app).
4. **Only then** freeze the parameters and evaluate on **new, unseen** boards: GOOD and DEFECT boards never used to set the limit, and TEST boards with IDs.
5. **Possible fail-safe for that evaluation:** a REVIEW band between the largest GOOD-to-GOOD difference and the FAIL limit. It is not applied now, because choosing it after seeing the TEST result would be tuning on the test image.

## 7. Live station (`scripts/live_inspection.py`, 2026-10-05)

A camera wrapper around this pipeline. It does not change the algorithm, the config or the thresholds.

**How it runs:**
- one capture every `--interval` seconds (default 3, measured capture start to capture start, never overlapping);
- `inspect_image` and `write_outputs` exactly as in `scripts/inspect_pcb.py`;
- the result is shown live, and `data/testing/live_output/summary.csv` is updated.

**Status mapping:**
- If the board cannot be aligned (pipeline REVIEW without an aligned image), the station shows **INSPECTION ERROR**, never PASS.
- Blank frames and missing frames are also INSPECTION ERROR.

**Verified on 2026-10-05:**
- **Real camera 1** (DroidCam, 2560 × 1440 delivered for a 1920 × 1080 request): captures 3.0 s apart. The scene was dark with no board, so every result was INSPECTION ERROR (board not aligned). The camera was released at the end.
- **Curated images replayed through the live window:** GOOD board B → PASS, DEFECT S02 → FAIL R37 (0.27), TEST → PASS (**the known miss**). 1.3–1.4 s per inspection.
- **Timing anomaly:** one earlier replay run took 7.6–8.0 s per inspection; it did not reproduce in two later runs.

**Limitation found:** in the current DroidCam stream, the watermark sits at the **top centre** of the frame (≈ x 780–1390, y 50–95). That is no longer the position the ignore mask in section 3 covers, so it is not ignored in live captures. Remove the watermark (paid edition or another app), or measure the new position before relying on parts near it.

No model was trained, no synthetic defects were made, and nothing was labelled from the images.
