# Planner: "Is the component there?" - YOLO + golden reference, on the phone

| | |
|---|---|
| Board | LG Indicator, 116 x 83 mm, top side, **after SMT reflow** (LEDs, switches, header and other THT parts are **not** fitted yet) |
| Board colours | **red now** (only red boards are available). Blue and green later: same PCB, same parts; added as their own golden + photos when available |
| The only question | For every SMT component: **present at its place, or not?** |
| Tolerance | A part up to **0.9 mm** away from its place counts as **present**. Further away counts as **missing** (it is not where it belongs). |
| Not checked now | wrong part / value, polarity, solder quality, damage, extra parts, foreign objects |
| Camera | **the phone** (no other camera). Photos are taken with the in-app camera of our app |
| Result | **Android APK**: take a photo -> PASS / FAIL (list of missing parts with red boxes) / REVIEW / RETAKE, everything offline on the phone |
| Use | personal project (so the AGPL licence of Ultralytics YOLO is fine) |
| Date | 2026-10-09 |

> **STATUS 2026-10-12 (newest, read first): the model is in the app.** Data grew to 21 good boards + 6 physical removal boards; label errors in the text files were found and corrected (`ml/presence/v3/label_overrides.json`);
> 12 cross-validation experiments were run (`ml/presence/v3/REPORT_V3.md`). Chosen: YOLO11n-cls + exposure normalisation, median over 9 windows per part, threshold 0.35 -> on unseen boards 0 of 291 removed parts missed, 1 of 15,103 good parts called missing.
> The Android app (`android_app/`) now does ONLY the missing-part check with this model (old colour / misplaced / difference features removed and archived in `archive/2026-10-12_old_app_checks/`); phone: ~9 s fast mode. Boxes are yellow (red is invisible on the red board).
> Still open: real removals of part types without examples (U1, U2, R2K2 x8, more C / C100N / D / D4148 / Q), labels for the 11 photos in `data/presence/test/red`, a bigger test set.
>
> **STATUS 2026-10-09 (history; the plan below is the original v2 and is corrected where marked)**
> * **A (YOLO crop classifier) is trained and checked** in `ml/presence/` (YOLO11n-cls, 128 x 128 crops at 20 px/mm). Cross-validation on 4 real removal boards: 102/102 removed-part crops found, 0/4198 present crops
>   called missing at 0.5; independent check on all 71 photos: 134/134 found, 1/5972 false call at 0.5, 0 at 0.6+. ONNX export done and tested with OpenCV 4.10 DNN (already inside the app). `ml/presence/predict_photo.py` checks photos on the PC.
> * **C (golden colour check) does NOT work for real removals** and is **dropped as a second vote**: the first real removals (19 parts) were all missed (pads stay under the old body zone, so no bare board is exposed). The "100 % of simulated missing
>   parts found" claims were invalid. The golden reference is still essential - for alignment and for cutting the crops - but not as a vote. Decision rule for now: **A alone** (FAIL when any part has p(missing) >= 0.5).
> * Not yet true: only R resistors and Q2, Q9 were ever removed (no capacitors, diodes, ICs), 4 removal boards, no independent test set, not in the APK. Next: more removal boards (all part types), a test set, then the app integration (OpenCV DNN, no new library).
> * **Board count (checked 2026-10-09, `ml/presence/report/LEAKAGE_CHECK.md`):** the 43 good photos R_01-R_43 are only **6 physical boards** (6-9 photos each); with the 4 removal boards the data holds **10 different boards**. Photos of one board are not independent, so below "boards" means physical boards:
>   the "at least 20 good boards" of set 2 is still missing (6 so far), and splits must be made by board (`ml/presence/board_groups.json`), never by photo. The text files never reach the prediction (blind test passed).

---

## 1. The answer first: what photos do you need?

**Take FULL-BOARD photos with your phone. Do not photograph single components, and do not crop anything by hand.**

Why:

1. The YOLO model will be trained on **component crops**, but those crops are **cut out automatically** by the computer from your full-board
   photos: every photo is first aligned to the golden reference (this already works in the app), so the position of every component is known
   in every photo, and the program cuts a small image around each component and names it (`R12`, `Q4`, ...).
2. One full-board photo of a good board = **86 "present" training images** at once. One photo of a board where you removed 5 parts =
   5 "missing" + 81 "present" images. Hand-made close-ups of single parts would look different from what the app sees and would need hand labels.
3. A phone photo of the whole board is exactly what the app gets later. The model learns from the same kind of picture it will judge.

Why the model works on **crops** and not on the whole board at once: in a phone photo of the whole board a small resistor (0805 / 0603) is only
a few pixels when YOLO shrinks the board to 640 px. Cut out at full resolution, every part is big enough. And because we cut each crop
**exactly where the golden says the part must be**, the crop itself answers the position question: if the part is not inside the crop
(more than 0.9 mm off, or gone), it is "missing".

---

## 2. How the app will decide (version 1)

```
phone photo (in-app camera, 4000 x 3000, whole board)
   |
   v
1. find the board, rectify, align to the golden of the same colour          (exists: corners, SIFT/ORB, RANSAC, ECC)
   |                                                                          -> RETAKE: not found / cut off / blurred / glare
   v
2. local alignment around every part (optical flow, exists)                  -> remaining error ~0.2 - 0.3 mm
   |
   +--> A. YOLO classifier (yolo11n-cls) on 86 crops: "present" / "missing"   -> p(present)        (NEW, trained by us)
   +--> C. golden check: share of the part's body zone that shows bare board -> exposure           (exists, no training)
   |
   v
3. decision per part:   A present + C present  -> PRESENT
                        A missing + C missing  -> MISSING
                        A and C disagree, or A unsure (0.2 < p < 0.8), or C skipped (glare) -> REVIEW (operator looks)
   board:  any MISSING -> FAIL   |  else any REVIEW -> REVIEW  |  else -> PASS
```

- **A** is the learned model (YOLO). **C** is the golden-reference colour comparison that was already built (measured on 260 real photos: 4 false
  "missing" in 22,174 checks). ~~100 % of simulated missing parts found~~ - **corrected 2026-10-09: on real removals C found 0 of 19, so it cannot be the second vote; see the status box at the top.**
- ~~Two independent methods that must agree make a wrong PASS very unlikely~~ (only true if both methods work; C does not, so A stands alone for now - a different second method, e.g. a feature-distance check against the golden crop, is an optional later experiment).
- **B (later, optional):** a YOLO **detector** on 640 x 640 tiles of the board, to measure the offset of each part in mm. Not needed for
  present / missing; only if we later want "misplaced" as its own category.

### The 0.9 mm rule inside the model

- Each crop is cut at the golden position of the part with a margin, for example the part box + 1.5 mm on every side, resized to 96 x 96 px.
- Training label **present** = the part is in the crop and its centre is within 0.9 mm of the crop centre.
- Training label **missing** = the part is not there, **or** it is more than 0.9 mm away. Examples of "more than 0.9 mm away" are made
  from real photos by **moving the crop window** by 1.3 - 2.5 mm (the part then sits off-centre); window moves of 0 - 0.6 mm stay "present".
  Moves between 0.6 and 1.3 mm are not used for training (too close to the limit to label them).
- The alignment error (about 0.2 - 0.3 mm after local alignment) is already inside this budget.

---

## 3. Photos to take (your part)

### 3.1 How every photo must look

| Item | Rule | Why |
|---|---|---|
| Whole board | all four board edges inside the photo, some background around it | the app finds the board by its outline |
| Size | the board fills **60 - 80 %** of the frame (phone 20 - 30 cm above the board) | small parts need >= 20 pixels per mm |
| Straight from above | tilt below 10 degrees | |
| Background | plain, **not** red / blue / green (white paper, grey table, black cloth) | the board colour must stand out |
| Light | room light or daylight, **no flash, no lamp shining straight on the board** (reflections hide parts) | |
| Sharp | tap on the board to focus, hold still | |
| Camera | **in-app camera** of our app (full 12 MP, saves the original) - or the phone camera app at full resolution, no zoom, no filter | |
| Never | edit, crop, rotate or compress the photos afterwards | the training must see real photos |

Vary between photos of the same board: turn the board (0 / 90 / 180 / 270 degrees), a slightly different height, a different spot in the
frame, a different light (day / evening, another room). This variation is what makes the model robust.

### 3.2 What to photograph

| # | Set | What you do | Photos | Gives (automatically) |
|---|---|---|---|---|
**Now: red boards only.** The numbers below are for red; when blue / green boards come, the same sets are added for that colour.

| # | Set | What you do | Photos (red) | Gives (automatically) |
|---|---|---|---|---|
| 1 | **Golden** | one good red board, 3 very careful photos (best light, sharp) | 3 | the golden reference photographed with your phone |
| 2 | **Good boards** | fully assembled SMT-stage boards that you know are good. **As many different boards as you can**, at least **20** | 20 boards x 6 photos = **120** | ~10,000 "present" crops |
| 3 | **Missing parts (most important)** | take **6 boards** you may destroy. Step by step: photograph (6 photos), **desolder 5 parts**, write their names down, photograph again, desolder 5 more, ... until 30 - 40 parts are gone. Use different parts on each board so that every part type is missing at least 3 times | 6 boards x ~7 steps x 6 photos = **~250** | ~1,300 real "missing" crops (with real solder / flux left on the pads) + many "present" crops |
| 4 | **Bare boards** (if you have unpopulated red PCBs) | an empty PCB | 2 x 6 = 12 | 86 "missing" crops per photo |
| 5 | **Test boards** | at the very end: **other boards** than in sets 2 - 4 (8 good + 4 with removed parts), photographed on another day | ~100 | the honest result numbers |

If you can only get a few boards: the minimum that makes sense is **8 good boards + 3 boards for set 3**.

Later, per extra colour (blue / green): golden (3 photos) + at least 10 good boards + 3 removal boards + test boards. Until then the
app only accepts red boards (other colours: message "red boards only" / RETAKE - the old golden colour check cannot find removed parts, see the status box at the top).

For every folder of photos that shows missing parts you put **one text file in the same folder**, named `<board>_missing.txt`
(step-by-step boards: `<board>_<step>_missing.txt`), with **all** parts missing in those photos, separated by commas (case does not matter):

```
raw\red\removed\R01\R01_missing.txt                  Q9, R12
raw\red\removed\R21\step2\R21_step2_missing.txt      R12, C14, Q4, D11, R31
raw\red\removed\R21\step3\R21_step3_missing.txt      R12, C14, Q4, D11, R31, U1, R2K2_5, C100N_3, D4148_6, R45
```

Never list the not-fitted parts (R50, R32, Q11, D14, R43, R44, D16, C17) or LEDs / switches. Good and bare boards need no file.

### 3.3 Folder layout on the PC

Created in `D:\Projects\YOLO_PCB_Componet\data\presence\`; the full instructions (folders, `<board>_missing.txt` format, the 86 part names,
photo rules) are in `data\presence\HOW_TO_ADD_PHOTOS.txt`.

```
data/presence/
   golden/red/                                  set 1   (golden/blue/, golden/green/ later)
   raw/red/good/<board_id>_<nn>.jpg             set 2 (all good boards in one folder, board ID in the file name)
   raw/red/removed/<board_id>/ + <board_id>_missing.txt      set 3 (step-by-step: <board_id>/stepN/ + <board_id>_stepN_missing.txt)
   raw/red/bare/<board_id>/IMG_*.jpg            set 4
   test/red/good|removed/<board_id>/...         set 5 (do not look at it while training)
```

Photos are **not cropped**: the whole board with 1 - 3 cm of background around it, exactly as the phone took it.

Board IDs: write a number on the back of each board (`R01`, `B01`, `G01` ...). **All photos of one board stay together**: a board is in
training, or in validation, or in test - never in two of them (otherwise the numbers lie).

---

## 4. The component list (do once, together)

The golden has **86 component positions** (`android_app/assets/golden/regions.json`). Before training, each one is checked once on
your photo of the board:

- **name** (R12, C14, Q4, U2, ...) and **box** around body + pads (some boxes today also cover a neighbour: these are fixed),
- **expected state at the SMT stage**: *fitted* or *not fitted* (DNP). **Confirmed by you (2026-10-09): R50, R32, Q11 and D14 are never
  fitted in this variant** -> *not fitted*, never checked, never reported (they are `R50_DNP_*`, `R32_DNP_*`, `Q11_DNP_*`, `D14_DNP_*` in the
  component list). The list also marks **R43, R44, D16 and C17** as not fitted (only pads on the golden) - to be confirmed. LED*, SW*, the
  header and the mounting holes are empty at this stage anyway,
- **part type** (used for statistics): chip resistor (R*, R2K2_*), ceramic capacitor (C*, C100N_*), small diode SOD-123 (D4148_*),
  diode SMA "M7" (D1, D11, D12), transistor SOT-23 (Q*), bridge rectifier MB10S (D13), regulator 78M05 DPAK (U1), microcontroller QFP (U2).

One component list is used for all three colours (same PCB).

---

## 5. Building the training data (my part)

A new script `scripts/make_presence_dataset.py`, built on the existing alignment code:

1. Read every photo, find the board, align it to the golden of the same colour (+ local alignment). Failed photos go to a `retake.txt` list.
2. Render the aligned board at **20 px/mm** (1660 x 2320 px).
3. Cut the **86 crops** (part box + 1.5 mm, 96 x 96 px) and label them from the folder + `<board>_missing.txt`:
   good -> present, bare -> missing, removed -> missing for the listed parts, present for the others.
4. Add the "shifted crop" examples for the 0.9 mm rule (section 2).
5. Write the YOLO classification layout:

```
datasets/presence/train/present/*.png   datasets/presence/train/missing/*.png
datasets/presence/val/present/*.png     datasets/presence/val/missing/*.png
datasets/presence/test/present/*.png    datasets/presence/test/missing/*.png
```

6. Write a **contact sheet** per photo (all 86 crops with their labels) so that you can check the labels by eye in a minute. A wrong label
   is fixed in the `_missing.txt` file, never in the images.

---

## 6. Training (my part, on your PC or free Google Colab)

| Item | Choice |
|---|---|
| Model | **YOLO11n-cls** (Ultralytics), pretrained on ImageNet, 96 x 96 input, 2 classes `present` / `missing` |
| Command | `yolo classify train data=datasets/presence model=yolo11n-cls.pt imgsz=96 epochs=60 batch=256` |
| Colours | version 1: red only. Augmentation: brightness, contrast, white balance, blur, JPEG noise, +-3 degree rotation, no mirror flips. **Hue rotation** (a red crop turned blue / green) is added already, so that the model is ready to be fine-tuned when blue / green photos exist - blue / green results are only trusted after real photos of those colours |
| Balance | far more "present" than "missing" crops: oversample "missing" (or class weights) |
| Hardware | CPU is enough (a few thousand small crops, about 1 h); Colab GPU is faster |
| Export for the phone | `yolo export model=best.pt format=tflite imgsz=96` (float16; int8 later if speed needs it) |

Then the thresholds of the decision (section 2) are set on the **validation** boards, and the final numbers are measured **once** on the
**test** boards.

### Targets (on the test boards)

| What | Target |
|---|---|
| Missing part reported as present (the dangerous error) | **0** on the test boards (>= 99.5 % of missing parts found) |
| Present part reported as missing | <= 0.1 % of parts, <= 1 board in 20 |
| REVIEW | <= 2 % of parts |
| Time on the phone | <= 3 s per board |
| Each colour separately | the same targets for every colour (red now; blue / green when their photos exist) |

---

## 7. The APK (my part)

1. **Model in the app**: `presence_v1.tflite` (+ the component list it was trained with) in the APK assets; run with **LiteRT
   (TensorFlow Lite)** inside the existing Kotlin module, after the alignment. 86 crops of 96 x 96 px take well below a second on the phone.
2. **Golden per colour**: the app detects the board colour and uses the golden of that colour (Golden screen: one reference per colour,
   photographed in the app with your phone).
3. **Result screen, presence only**: big **PASS / FAIL / REVIEW / RETAKE**; FAIL lists the missing parts (red boxes on the board, name,
   "missing"); REVIEW shows the crop of the new board next to the golden crop so that you decide; all other categories are removed.
4. **Learning from use**: every REVIEW and every correction you make is saved (crop + your answer) and can be exported to the PC for the
   next training round.
5. Build: `npm run build:apk` -> `android_app/app-debug.apk`, install with `adb install -r app-debug.apk` (or copy to the phone).

---

## 8. Steps and order

| Step | Who | What | Done when |
|---|---|---|---|
| 0 | me | switch the app to presence only (no other categories); check the 86 boxes + fitted / not fitted with you | the current APK shows only present / missing |
| 1 | you | golden photos of a red board (set 1) + a first 10 good-board photos | photos in `data/presence/` |
| 2 | me | `make_presence_dataset.py` + contact sheets from the first photos | you have checked the contact sheets |
| 3 | you | all good boards (set 2) and the step-by-step removal boards (set 3) (+ bare boards, set 4) | ~700 photos |
| 4 | me | train YOLO11n-cls, set thresholds on validation, numbers per colour | report |
| 5 | you | test boards (set 5), photographed last | ~150 photos |
| 6 | me | final numbers on the test boards; TFLite in the app; APK | APK + numbers |
| 7 | you + me | use it on real boards; every REVIEW / mistake becomes training data | version 2 of the model |

---

## 9. Transformers - can they improve it?

| Option | Use here | Verdict |
|---|---|---|
| **DINOv2** (pretrained vision transformer) as a feature extractor + nearest neighbour or a small head | judges a crop by comparing it with many good crops of the same part; needs few "missing" examples, robust to light / colour changes | **worth one experiment after version 1** (step 7), kept only if the test numbers get better; ~2 - 4 s extra on the phone |
| **RT-DETR** (transformer detector) | instead of a YOLO detector | not needed: we classify crops, we do not detect objects in the whole board |
| Small mobile transformers (MobileViT, EfficientViT) | instead of the YOLO classifier | only if YOLO11n-cls is not good enough |

A transformer does **not** replace the photos of missing parts: the step-by-step removal boards (set 3) matter more than any model choice.

---

## 10. Risks

| Risk | What we do |
|---|---|
| Not enough photos of missing parts | the step-by-step removal (set 3) gives many from a few boards; bare boards help; hue / light augmentation |
| Blue and green boards behave differently from red | golden per colour, every colour in train / val / test, numbers per colour |
| Phone photos not sharp enough | RETAKE rule (blur, glare, cut off) and the photo rules in 3.1; the board must fill >= 60 % of the frame |
| Glare / reflections | parts under glare -> REVIEW; too many -> RETAKE |
| Parts without a body that differs from the board | (old golden colour check: dropped) the YOLO crop model judges every part |
| The test boards were seen during training | strict board-wise split; test photos only at the end |
