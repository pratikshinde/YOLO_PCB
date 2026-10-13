# Experiment 0: Capture Checklist (mobile phone)

Camera-first: take the photo, copy the original, run one command. **No typing per image.** The full protocol is in [experiment_0_camera_visibility.md](experiment_0_camera_visibility.md).

## Before you start
- [ ] Plain **white or light-grey** matte paper or card as the background (not wood, not blue / red / green)
- [ ] Phone stand or another steady support (if available)
- [ ] Phone: main 1× camera, highest resolution, normal photo mode, JPEG; **no** zoom, portrait mode, filters or flash
- [ ] A verified-good LG Indicator board (and any defective boards with known defect information)

## Take the photo
- [ ] Board top side up, centred, **whole board plus a margin** in the frame
- [ ] Phone as top-down as practical, held still
- [ ] Tap to focus on the **small resistors**; avoid strong glare spots
- [ ] Zoom into the photo on the phone: small resistors sharp? Otherwise take it again.

## Bring it into the project
- [ ] Copy the **original** file by USB cable (never via WhatsApp / Telegram / e-mail resize)
- [ ] Put it, unrenamed, into `data/experiment_0/inbox/`
      (several photos of one board? put them in `inbox/GOOD_001/` instead)
- [ ] Run:

  ```bash
  python scripts/e0_ingest.py
  ```

  Optional, once for the whole batch: `--stage after_tht` / `--stage after_smt_reflow`, `--distance-mm 300`, `--lighting "ceiling LED"`.

## Check the result
- [ ] Look at the table the command prints: every photo `ok`? Colour and px/mm plausible?
- [ ] Open flagged photos at full resolution, and re-take them if needed
- [ ] Open at least one photo at 100 %: are the small resistors, their ends and markings visible?
- [ ] Defective boards only: record each defect in `data/experiment_0/metadata/defects.csv`

## Never
- ✘ Renaming, editing or re-saving photos
- ✘ Artificial defects, synthetic images, annotations
- ✘ Train / test splits or model training
