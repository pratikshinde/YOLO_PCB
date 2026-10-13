# Missing Part Check (Android, React Native + Expo)

Offline check of one PCB design (**LG Indicator, red board, SMT stage**, 116 x 83 mm): photograph the whole board, the app tells **PASS / FAIL / RETAKE**.
It looks for one thing only: **is every one of the 86 SMT components present?** FAIL = at least one is missing (yellow box + name on the board image).

* No network, no cloud: board detection, alignment, a small neural network (YOLO11n-cls as ONNX) and the decision all run on the phone.
* The golden reference (`assets/golden/grf.jpeg` + `regions.json`) is bundled and fixed: it defines where every part must be. It is not editable in the app (the model belongs to it).
* Boxes: **yellow** = component missing (decides FAIL), **cyan** = doubtful (the model is not sure; listed, does not change the result).
* The old colour / appearance checks (misplaced, different part, difference map, heat map, fail policy ...) were removed on 2026-10-12: they did not find really removed parts. Their code is archived in `../archive/2026-10-12_old_app_checks/`.

## How it works

```
photo -> board found and rectified (HSV + corners) -> aligned to the golden reference at 20 px/mm (SIFT + RANSAC + ECC)
      -> local alignment of every part (dense optical flow)           -> exposure normalisation of the photo
      -> for each of the 86 parts: 9 crops (128 x 128, part box + 1.5 mm, window shifted 0 / +-0.4 / +-0.8 mm) -> YOLO11n-cls (OpenCV DNN)
      -> score of the part = median over the 9 windows (p(missing), 0..1)
      -> part MISSING when score >= 0.35 (Settings), doubtful from 0.10;  board FAILS when any part is missing;  more than 20 flagged = RETAKE
```
Photo problems give **RETAKE** (never PASS or FAIL): board not found / cut off / blurred / cannot align / implausibly many parts flagged.

| Layer | What | Where |
|---|---|---|
| UI | React Native + Expo SDK 57 (Expo Router, TypeScript): Home, Inspect (camera / gallery), Result, History, Settings | `src/` |
| Native module | Kotlin Expo module **GoldenComparator**: alignment + `Presence.kt` (crops, DNN, decision) with **OpenCV 4.10 for Android**, CameraX camera view | `modules/golden-comparator/` |
| Model | `assets/model/presence_cls.onnx` (5.9 MB) + `presence_model.json` (model card: crop rule, exposure constants, thresholds, test numbers) | `assets/model/` |
| Training / export | cross-validation, final model, ONNX export, Python reference `predict_photo.py` | `../ml/presence/` (`v3/`, report `v3/REPORT_V3.md`) |

`Presence.kt` is a copy of `ml/presence/predict_photo.py` (same steps, same numbers); change one, change the other (see "Parity").

## Build the APK

```
cd android_app
npm install
npm run build:apk        # = node scripts/build-apk.js  ->  app-debug.apk (arm64, JS bundled, runs offline)
adb install -r app-debug.apk
```
`--clean` rebuilds the native project from scratch. The script finds Android Studio's JDK and the SDK (needs Android Studio installed) and works around a JDK loopback quirk on this PC.

## Use

Photograph the whole board straight from above, flat, 1-3 cm of background all round, no flash, soft light, sharp (tap to focus). Settings: sensitivity (missing threshold 0.25 / 0.35 / 0.50), thorough (9 windows, default) or fast (1 window) mode, alignment limits. The result screen shows the board with the boxes, the missing / doubtful parts with their scores, alignment details and timings; the original photo, annotated image and `result.json` are stored per inspection (History) and can be shared.

## Measured

On whole boards the model had never seen (5-fold cross-validation, 291 removed-part crops, 15,103 good-part crops, thorough mode, threshold 0.35): **0 missed, 1 good part called missing, 1 of 128 good photos failed, 51 of 51 removal photos with exactly the right list.** Details, all experiments and the limits: `../ml/presence/v3/REPORT_V3.md`.
Checked on the phone (Redmi): fast mode 8.9 s per photo; the Kotlin code equals the Python reference on 11 photos (same decisions and parts, scores within 0.025).

## Limits (read before trusting a result)

1. **Only resistors are proven** (45 real removals on 6 boards). Transistors, diodes, capacitors had a handful of real examples; **R2K2 (x8) and the ICs U1 / U2 had none**: a missing one there is not guaranteed to be found.
2. 6 physical removal boards + 21 good boards: the false-alarm rate on new boards is only roughly known.
3. Red board only. A part moved by 0.9 mm still counts as present, at 1.2 mm about half are called missing.
4. Never-fitted parts (R50, R32, Q11, D14, LEDs, switches, header) are not checked.

## Parity (Kotlin vs Python)

`tools/parity` runs the Kotlin pipeline on the desktop (JVM + desktop OpenCV) so it can be compared with `predict_photo.py` photo by photo:
```
node scripts/build-apk.js --gradle -p ../tools/parity run "--args=<golden.jpeg> <regions.json> <presence_cls.onnx> <presence_model.json> <outDir> <photo> [<photo> ...]"
ml\presence\.venv\Scripts\python.exe ml\presence\predict_photo.py <photo> ...       # Python reference
```

## Storage (app-private, `Paths.document/pcb`)

```
references/LG_INDICATOR_SMT_RED/  reference.jpg (bundled golden), canonical.png (1660 x 2320 px), structure.png, regions.json, meta.json
inspections/<timestamp>/          input.jpg (original, never modified), aligned.jpg, annotated.jpg, result.json, summary.json
settings.json
```
The golden reference is prepared at the first start (a few seconds); it is rebuilt automatically if it is missing or made for another scale.
