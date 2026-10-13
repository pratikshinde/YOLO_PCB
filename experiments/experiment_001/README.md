# Experiment 1: PCB Alignment + Golden-Reference Comparison (baseline)

| | |
|---|---|
| Status | Run once on 2026-10-02 with the first real phone images. **Not production-ready. Not a defect detector.** |
| Profile | `lg_indicator/variant_1` (LG Indicator, 83 × 116 mm, top side) |
| Command | `python scripts/run_experiment_001.py --reference S20261002-01_C003 --test S20261002-01_C001 S20261002-01_C002 S20261002-02_C001 S20261002-03_C001` |
| Config | [`config.yaml`](config.yaml) (all thresholds are experiment parameters, not inspection limits) |
| Outputs | `reference/`, `runs/<reference>__<test>/` (images + `metrics.json`), [`results.csv`](results.csv) |
| Software | Python 3.14.5, OpenCV 5.0.0, NumPy 2.5.1; Intel64 CPU (no GPU used) |

## Objective
Can two images of the LG Indicator be aligned accurately enough to compare them? And how much difference exists **naturally** between images of good boards? No learning is involved: no neural network, no YOLO, no anomaly model, no synthetic data.

## Dataset used
Real Experiment 0 phone photos only: Xiaomi Redmi 12 5G, 6144 × 8160 px, JPEG, all taken 2026-10-02 between 15:08 and 15:11 under room light. The raw images were not modified; checksums were verified before and after the run.

| Image ID | Board | Taken | Role | Note |
|---|---|---|---|---|
| S20261002-01_C003 | GOOD_001 | 15:10:49 | **Reference** | Highest resolution of the GOOD_001 photos (≈61 px/mm), no quality flags |
| S20261002-01_C002 | GOOD_001 | 15:10:47 | Test, same board | 2 s before the reference: almost the same camera position |
| S20261002-01_C001 | GOOD_001 | 15:10:17 | Test, same board | 32 s earlier; camera moved |
| S20261002-02_C001 | GOOD_001 | 15:10:51 | Test, same board | Camera further away (≈48 px/mm) |
| S20261002-03_C001 | GOOD_002 | 15:08:36 | Test, **other good board** | Board identity from a visual check (C14 placement), still to be confirmed; inspection stage unknown |

## Reference image
`S20261002-01_C003` is an **experiment-only reference**, stored in `reference/S20261002-01_C003/` (canonical image + `reference.json`). It is **not** declared a production golden reference.

## Method
1. **Board segmentation.** The board is found by its own colour (the dominant hue in the image centre), with holes filled. The four board corners give the **canonical board frame**: the reference rectified to 83 × 116 mm at 20 px/mm (1660 × 2320 px). This crops the background and corrects perspective.
2. **Registration** (test → reference):
   - SIFT keypoints, **only inside the board masks**, on copies downscaled to 2400 px.
   - Ratio-test matching (0.75), then one RANSAC homography (3 px threshold at working resolution).
   - The test image is warped straight into the canonical frame.
   - No fiducials are assumed: the board's holes are not confirmed as fiducials (PCB-17).
3. **Independent alignment check.** Phase correlation on 6 × 6 mm tiles of the aligned pair. It measures the misalignment left over, independently of the matched features (it underestimates by about 5–10 %).
4. **Preprocessing.** Light blur (0.05 mm), a 1.5 mm strip along the board edge ignored, and global illumination normalisation: lightness gain/offset plus an a*/b* white-balance offset. Colour channels are never scaled, so a genuine colour difference cannot be normalised away.
5. **Difference maps:**
   - colour difference ΔE (CIE76 on L\*a\*b\*);
   - structural dissimilarity (1 − SSIM)/2;
   - edge mismatch;
   - **candidate regions**: ΔE > 20 or dissimilarity > 0.25, opened 0.2 mm, at least 0.5 mm².
6. **Visualisation, per pair:**
   - `reference.jpg`, `test.jpg` (with the detected board outline), `aligned.jpg`;
   - `difference.jpg` (ΔE, 0–40), `dissimilarity.jpg`, `edges.jpg`;
   - `thresholded.jpg` (candidates), `overlay_checkerboard.jpg`, `residuals.jpg`;
   - `panel.jpg` (all of the above).

On good boards, **every candidate region is a false candidate by definition.** Counting them measures the natural-variation floor. `metrics.json` lists up to the 50 largest candidate regions per pair; `results.csv` has the full counts.

## Parameters
See [`config.yaml`](config.yaml). The key ones:
- **Registration:** SIFT, 8000 features, ratio 0.75, RANSAC 3 px.
- **Residual tiles:** 6 mm.
- **Comparison:** illumination `global`; ΔE threshold 20; dissimilarity threshold 0.25; minimum region 0.5 mm²; 1.5 mm border margin.

## Results
Compared area: ≈9 040 mm² of the 9 628 mm² board.

| Test | Board | Inliers (ratio) | Fit RMSE | Residual median / p90 / max | SSIM | ΔE mean / p99 | Area ΔE>20 | False candidates | Candidate area | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| S20261002-01_C002 | GOOD_001 | 2676 (0.89) | 0.064 mm | 0.042 / 0.114 / 0.36 mm | 0.945 | 3.8 / 19.3 | 0.9 % | **17** | 25 mm² (0.3 %) | 2.6 s |
| S20261002-02_C001 | GOOD_001 | 1989 (0.83) | 0.062 mm | 0.053 / 0.128 / 0.96 mm | 0.925 | 5.0 / 24.2 | 1.8 % | **47** | 117 mm² (1.3 %) | 2.6 s |
| S20261002-01_C001 | GOOD_001 | 1991 (0.85) | 0.059 mm | 0.041 / 0.105 / 1.09 mm | 0.918 | 5.6 / 30.0 | 3.0 % | **72** | 233 mm² (2.6 %) | 2.9 s |
| S20261002-03_C001 | GOOD_002 | 506 (0.69) | 0.061 mm | 0.049 / 0.113 / 1.02 mm | 0.799 | 18.5 / 73.5 | 30.9 % | **258** | 2 455 mm² (27 %) | 3.1 s |

**Processing time:** about 2.6–3.1 s per pair on the CPU.
- Reading and segmenting each 50 MP image: 0.45 s.
- Registration: 0.64 s.
- Comparison: 0.56 s.
- Writing the images: 0.41 s.

### Registration quality
- **All 4 pairs registered.**
- **Fit error:** about 0.06 mm RMSE.
- **Residual misalignment:** median 0.04–0.05 mm (about one canonical pixel); 90 % of tiles within 0.13 mm.
- **Same accuracy across boards:** the other board (GOOD_002) aligned as well as the same board did.
- **The large maxima (≈1 mm)** are single tiles at the tall through-hole LEDs and the switch. That is parallax (these parts are not on the board plane), not a registration failure.
- **Checkerboard overlays:** traces and pads continue across the square borders on all pairs.

## Failure cases (false candidate regions on good boards)
1. **Parallax of tall parts.** Through-hole LEDs and switch SW1 (the board's tallest parts are ≈12 mm) shift sideways when the camera moves, producing crescent-shaped ΔE around the LEDs.
   - SW1 produces candidates in every pair (1–3 regions).
   - In the best case (C002), all 17 candidates sit at SW1 (2), at LED positions (11) or at board corners (4). Positions were read off the canonical image; there is no component map yet.
2. **Gloss and lighting.** The glossy solder mask and solder joints reflect room light differently from each camera position. The result is broad, smooth ΔE patches (e.g. the left side in C001) and bright spots on the solder rows. Global normalisation cannot remove local sheen.
3. **Board-to-board appearance (GOOD_002).** A deeper crimson solder-mask shade, a different exposure / ISO / white balance chosen by the phone (1/33 s ISO 350 vs 1/20 s ISO 707), and a much sharper photo (sharpness 36 vs ≈10). Together these produce differences over 27 % of the board, burying anything smaller.
4. **Board edges and corners.** A few small candidates where background or chamfers reach the 1.5 mm margin.

## Observations
- **Registration is not the bottleneck.** Board-only SIFT plus a homography aligns phone photos of this board to about 0.05 mm, without fiducials.
- **Capture conditions dominate the noise floor.** With almost identical capture conditions (C002, 2 s apart), false candidates fall to 17 small regions (0.3 % of the area), mostly on tall parts. As the camera position and lighting change, they grow to 72 regions (2.6 %).
- **Raw colour comparison across boards is not usable as is.** The colour and shade differences between boards and photos are larger than the expected size of many defects.
- **The phone chooses exposure, ISO, white balance and sharpening itself.** Each photo is therefore processed differently.

## Limitations
- **Very little data.** One reference, 3 same-board tests and 1 other-board test, all from one short session. The GOOD_002 identity and its stage are not yet confirmed.
- **No defective board** was compared (none with a verified defect location has been captured yet), so **nothing here says whether real defects stand out** above this noise floor.
- **Uncontrolled capture:** hand-positioned phone, room light, varying distance (48–61 px/mm). These results describe this capture setup, not the method's limit.
- **The thresholds** (ΔE 20, dissimilarity 0.25, 0.5 mm²) are arbitrary experiment parameters, and the candidate counts depend on them.
- **One plane, one homography:** parallax of tall parts is not modelled.
- **Phase-correlation residuals** are slightly underestimated (5–10 %).

## Conclusion
- **Alignment: promising.** About 0.05 mm residual on real phone images, without fiducials.
- **Reference comparison: not yet reliable.** Under the current uncontrolled capture, natural differences between images of the **same** good board produce 17–72 false candidate regions. Between **two** good boards, colour and lighting differences cover about 27 % of the board.
- **Most of this noise comes from capture conditions** (camera position → parallax; lighting/gloss; automatic phone exposure), which a fixed setup can reduce.
- **Whether reference comparison can detect real defects is still open.** It needs controlled captures, several good boards, and real defects with verified locations.

## Next experiment
**Experiment 1b: same pipeline, controlled capture.**
- **Setup:** phone fixed on a stand at one measured distance, plus diffuse lighting (no direct lamps or window light). If possible, lock exposure / focus / white balance (pro mode).
- **Boards:**
  - One board re-placed about 10 times → capture repeatability.
  - 5–10 distinct good boards → board-to-board variation, including other colours if available.
  - Real defective boards with a verified defect location → first check whether real defects rise above the noise floor. Verified boxes can be passed with `--defect-box`; labels are never inferred from differences.
- **Pipeline variants to try:** structure-based comparison (SSIM / edges) instead of raw colour across boards, and a reference averaged over several good boards. Tall parts may need masking or per-region tolerances.
