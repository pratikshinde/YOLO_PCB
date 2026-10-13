# Task: train a "component present / missing" model (YOLO11n-cls) for one PCB, from real phone photos

You are working on a Windows 11 PC (Intel i7-13645HX, 16 GB RAM, **NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB**, Python 3.14.5 installed,
PyTorch and Ultralytics NOT installed yet). The project is `D:\Projects\YOLO_PCB_Componet`. Read this whole prompt before you start.

## 1. Goal

For one PCB design ("LG Indicator", 116 x 83 mm, red solder mask, photographed after SMT reflow) decide **for each of 86 SMT components
whether it is present at its place or missing**. A part up to 0.9 mm away from its place counts as present. Nothing else is checked
(no wrong value, polarity, solder quality).

Approach: every phone photo is aligned to a golden reference image with EXISTING code; then a small crop is cut at the known position of
every component; a YOLO11n **classification** model (classes `present`, `missing`) judges each crop. You build the dataset from the
photos, train the model, evaluate it honestly, and export it. **This first model is a trial**: there are few "missing" examples (see 3.3).

## 2. Rules (do not break them)

1. Put ALL new files under `D:\Projects\YOLO_PCB_Componet\ml\presence\` (code, generated crops, datasets, runs, reports).
2. **Do not modify or delete anything else** in the project: not `scripts\`, not `android_app\`, not `data\`, not `src\`, not `docs\`.
   The photos in `data\presence\` are read-only: never edit, rename, move or re-save them.
3. Do not create a git repository. Do not install anything system-wide; use a virtual environment inside `ml\presence\.venv`.
4. Never use the test data (section 6) for training or for choosing thresholds. Never mix photos of the same physical board into both
   training and evaluation (see 3.4).
5. Do not invent results. Report exactly what was measured, including failures. If something is impossible (e.g. a package does not
   install for Python 3.14), say so, try the documented alternative, and report it.
6. Do not use the old "golden colour check" (`check_missing` in `golden_compare.py`) to create labels: it is known to MISS real removals
   on this board (the pads extend under the parts, so a removed part shows grey solder pads, not bare board). Labels come ONLY from the
   folders and the text files described in 3.2 / 3.3.
7. Do not create artificial "missing" images by painting parts away. The only allowed synthetic examples are the crop-window shifts of 4.3.

## 3. The data (already on disk)

### 3.1 Golden reference and component list
- Golden image: `D:\Projects\YOLO_PCB_Componet\data\presence\golden\red\grf.jpeg` (1085 x 1449, the board cropped exactly at its edges,
  portrait, header pins at the bottom; identical to `android_app\assets\golden\golden_reference.jpg`).
- Component list: `D:\Projects\YOLO_PCB_Componet\android_app\assets\golden\regions.json`. Each entry has `id`, `name`, `kind`
  (`part` or `empty`) and `bbox_mm` = [x, y, w, h] in mm in the golden frame (origin top-left, x right, y down, board 83 x 116 mm).
  **Only the 86 entries with `kind == "part"` are checked.** Entries with `kind == "empty"` (LEDs, switches, header, holes and the
  not-fitted parts `R50_DNP_*`, `R32_DNP_*`, `Q11_DNP_*`, `D14_DNP_*`, `R43_DNP_*`, `R44_DNP_*`, `D16_DNP_*`, `C17_DNP_*`) are ignored.
- The 86 part names: R1 R2 R3 R4 R9 R10 R11 R12 R13 R18 R19 R21 R22 R23 R24 R25 R26 R27 R28 R29 R30 R31 R34 R35 R36 R37 R38 R39 R40 R42
  R45 R48 R49 R51 R52 R53 R54 R55 R56 R57 R58 R59 R60, C2 C14 C16, D1 D11 D12 D13, Q1..Q10, U1 U2, R2K2_1..R2K2_8, C100N_1..C100N_8,
  D4148_1..D4148_8.

### 3.2 Good boards (all 86 parts present)
`D:\Projects\YOLO_PCB_Componet\data\presence\raw\red\good\R_01.jpg ... R_43.jpg` - 43 whole-board phone photos (about 1600 x 2250 px,
board on a grey background). Every part in these photos is **present**. The file names are photo numbers; it is NOT known which photos
show the same physical board (assume consecutive numbers may be the same board).

### 3.3 Boards with removed parts
`D:\Projects\YOLO_PCB_Componet\data\presence\raw\red\removed\m_01\ ... m_04\` - four DIFFERENT physical boards, 6 - 8 photos each.
Each folder has one text file (named `m_01.txt.txt` etc. - the double extension is a Windows artefact; read **any** `*.txt` in the folder)
listing the parts that are **missing in all photos of that folder**, comma separated:

| folder | missing parts |
|---|---|
| m_01 | R10, R36, R23, R26, R22 |
| m_02 | R12, R28, R19, R34, R54, R38 |
| m_03 | Q2, R45, R39, R27 |
| m_04 | R29, R21, R51, Q9 |

All other parts on these boards are **present**. Parse the files case-insensitively; split on commas, semicolons, spaces and new lines;
ignore lines starting with `#`; validate every name against the 86 part names and stop with a clear error for an unknown name.
These lists were checked by eye and are correct. Note the weakness: only resistors and two transistors are ever missing - there is NO
missing capacitor, diode or IC yet, so the model cannot be expected to recognise those as missing.

### 3.4 Photos that cannot be used
These photos are known to fail board detection (the board touches the image edge): R_03, R_05, R_06, R_07, R_08, R_23, R_31,
3 photos in m_02 and 2 in m_04. Your code must detect such failures itself (the alignment returns RETAKE), skip the photo and list it in the
report - do not hard-code the list.

## 4. Build the dataset (`ml\presence\make_dataset.py`)

### 4.1 Alignment: reuse the existing code, do not rewrite it
`D:\Projects\YOLO_PCB_Componet\scripts\golden_compare.py` (Python + OpenCV + NumPy) already finds the board, rectifies it, aligns it to the
golden (SIFT/ORB + RANSAC + ECC) and measures local shifts. Import it (`sys.path.insert(0, r"D:\Projects\YOLO_PCB_Componet\scripts")`,
`import golden_compare as gc`). Use it like this (this was tested on these photos at 20 px/mm: about 4 s per photo, residual ~0.12 mm):

```python
P = gc.Params()
P.px_per_mm = 20            # canonical frame 1660 x 2320 px (83 x 116 mm); small parts are ~30 px long
P.max_input_side = 4000     # do not shrink the phone photos before aligning
ref = gc.prepare_reference(gc.imread(golden_path), P, gc.load_regions(regions_json_path, P))
result, images = gc.inspect(ref, gc.imread(photo_path), P)
if result["decision"] == "RETAKE" or images is None:   # board not found / cut off / blurred / glare
    skip the photo, record result["reason"]
aligned = images["aligned"]          # BGR, 2320 x 1660, the photo warped into the golden frame
# local fine alignment (a slightly bent board / lens distortion): per-part shift in px
rg, ng, _ = gc.comparison_grays(ref.clahe, gc.clahe_gray(aligned), ref.area, P)
shifts = gc.region_shifts(gc.local_flow(rg, ng), ref, ref.regions, P)    # {region.id: (dx, dy)}; missing id = (0, 0)
```
`ref.regions` is a list of `Region` objects with `.id`, `.name`, `.kind`, `.bbox_px` = [x, y, w, h] in canonical px. The shift of a region
(estimated from the board structure AROUND the part, so it is valid even when the part is missing) says where the part's neighbourhood
really is in the aligned photo: the expected part centre is `(x + w/2 + dx, y + h/2 + dy)`.

### 4.2 Crops
For every usable photo and every part region (`kind == "part"`):
- square window centred at the expected centre, side = `max(w, h) + 2 * 1.5 mm` (1.5 mm = 30 px at 20 px/mm), clipped to the image
  (pad with the border colour if needed), resized to **128 x 128** (INTER_AREA), saved as PNG.
- label: `missing` if the part name is in the folder's missing list, else `present`.
- file name: `<board>__<photo stem>__<part name>.png`, e.g. `m_01__IMG_20261009_123711_084__R10.png`.
- write `ml\presence\crops\index.csv`: crop path, board (good photos: `good_R_01` ...; removed: `m_01` ...), photo, part, part type
  (R / C / D / Q / U / R2K2 / C100N / D4148), label, shift dx, dy (px), alignment residual (mm).
- write a contact sheet per photo (`ml\presence\crops\sheets\<photo>.jpg`): all 86 crops in a grid with the part name and label printed,
  missing ones framed red, so a human can check the labels in a minute.

### 4.3 The 0.9 mm position rule (synthetic shifts, clearly marked)
To teach "more than 0.9 mm away = not at its place", additionally cut, for `present` parts only, windows whose centre is moved by
1.3 - 2.5 mm in a random direction -> label `missing`, flag `synthetic_shift=1` in the index. Windows moved by 0 - 0.6 mm stay `present`
(use them as augmentation). Never use moves between 0.6 and 1.3 mm. Limit: at most as many synthetic-shift crops as there are real
missing crops in the same training split, and report every metric for real missing crops only (synthetic shifts excluded).

## 5. Environment (inside `ml\presence\.venv`)

1. Check which Python versions exist (`py -0`). If Python 3.12 or 3.11 is installed, prefer it for the venv (best package support);
   otherwise use 3.14.
2. The RTX 5050 is a Blackwell GPU (compute capability 12.0): it needs a PyTorch build with **CUDA 12.8 or newer**:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (or cu129/cu130 if cu128 has no wheel for the
   Python version). Then `pip install ultralytics opencv-python numpy pandas matplotlib`.
3. Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print
   `True` and the RTX 5050. If CUDA does not work, report why and train on the CPU (the dataset is small; it is slower but works).

## 6. Splits (by physical board, never by photo)

There are only 4 boards with missing parts, so use **leave-one-removed-board-out cross-validation** (4 folds):
- fold k: the removed board `m_0k` is the **validation** board; the other three removed boards are training.
- good photos: split into 4 groups of consecutive photo numbers (R_01-R_11, R_12-R_22, R_23-R_33, R_34-R_43, unusable ones skipped);
  group k is validation in fold k, the rest training (consecutive photos are probably the same board, so this limits leakage).
- dataset folder per fold in the Ultralytics classification layout:
  `ml\presence\datasets\fold_k\train\present\*.png`, `...\train\missing\*.png`, `...\val\present\*.png`, `...\val\missing\*.png`
  (use hard links or copies of the crops; synthetic-shift crops only in `train`).
- Class imbalance: there are ~50x more present than missing crops. In `train`, repeat (copy with a suffix) every real missing crop so that
  missing : present is about 1 : 4, and/or cap present crops per part per board. Never balance `val`.
- After the cross-validation, train one **final model on all data** (all 4 removed boards + all good photos) for export.

There is no separate test set yet (the user will photograph new boards later). Say clearly in the report that the cross-validation numbers
come from only 4 removed boards and 19 removed parts.

## 7. Training

Ultralytics YOLO11 classification, pretrained:
```
yolo classify train model=yolo11n-cls.pt data=ml\presence\datasets\fold_k imgsz=128 epochs=100 patience=25 batch=64 device=0 ^
     project=ml\presence\runs name=fold_k fliplr=0.5 flipud=0.0 hsv_h=0.015 hsv_s=0.5 hsv_v=0.4 degrees=5 translate=0.04 scale=0.15 erasing=0.0 seed=0
```
(`erasing=0.0`: random erasing would delete the very thing we look for. Small translate only: the position is part of the label.)
Use the Python API instead of the CLI if that is easier; keep the same settings. If GPU memory is a problem, lower `batch`.

## 8. Evaluation (`ml\presence\evaluate.py`) - report per fold and summed over the 4 folds

On every validation crop compute p(missing) with the fold's best weights. Then:
- **Real missing parts found (recall)** at thresholds 0.3 / 0.5 / 0.7, and the list of missed ones (board, photo, part).
- **False "missing" on present parts** (false-call rate): number and rate, list the worst 30 (highest p(missing)) with crop paths.
- The same per part type (R, R2K2, Q, ...) - note that C / D / U types have no real missing example.
- Board-level: for each validation photo, would the board FAIL (any part p >= threshold) - for removed boards this must be FAIL, for good
  photos PASS.
- Choose a recommended threshold (high recall first, then fewest false calls) from the cross-validation and state it.
- Contact sheets of all errors (`ml\presence\report\errors_fold_k.jpg`).

Write `ml\presence\report\REPORT.md` with: environment (Python, torch, CUDA, GPU), data counts (photos used / skipped with reasons, crops
per class and type, real vs synthetic), the cross-validation tables, the recommended threshold, the error lists, and honest
limitations (few boards, only R and Q ever missing, photos not full resolution, no independent test set).

## 9. Export (final model)

- `yolo export model=ml\presence\runs\final\weights\best.pt format=onnx imgsz=128` -> keep the `.onnx`.
- Try `format=tflite` as well (float16). On Windows this needs TensorFlow and often fails; if it fails, write down the exact error and
  leave the ONNX file - the phone app integration (LiteRT / TFLite in a Kotlin module) is a later, separate step and is NOT part of this task.
- Save `ml\presence\models\presence_v0.onnx` (+ `.tflite` if it worked) and `ml\presence\models\presence_v0.json` with: class order,
  input size 128, crop rule (square, part box + 1.5 mm, 20 px/mm frame), recommended threshold, the list of the 86 part names, training
  data summary, date.

## 10. Also deliver

- `ml\presence\predict_photo.py <photo>`: aligns one whole-board photo (same code as above), cuts the 86 crops, runs the final model, prints
  PRESENT / MISSING per part and PASS / FAIL for the board, and writes an annotated JPG (red boxes on missing parts) next to the report.
  Run it on one good photo and one photo of every removed board and include the outputs in the report.
- `ml\presence\README.md`: how to rebuild the dataset, train, evaluate, export and predict, in 10 - 20 lines, with the exact commands.

## 11. Done when

1. `make_dataset.py`, `evaluate.py`, `predict_photo.py` run without errors from the venv;
2. the 4 folds and the final model are trained;
3. `REPORT.md` has real measured numbers (no estimates) and the limitations;
4. the exported model files exist;
5. nothing outside `ml\presence\` was changed.

At the end, give a short summary: how many real missing parts were found in cross-validation, how many present parts were wrongly called
missing, the recommended threshold, and what data is needed next (more boards with removed capacitors, diodes, transistors, ICs).
