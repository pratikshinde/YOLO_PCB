"""Is every SMT part on the board?  Phone photo(s) of the red LG Indicator board in -> PASS / FAIL / RETAKE out.

    ml\\presence\\.venv\\Scripts\\python.exe ml\\presence\\predict_photo.py PHOTO [PHOTO ...]
    (a folder or a pattern such as  data\\presence\\test\\red\\*.jpg  also works)

Options
    --threshold 0.35  a part counts as MISSING when p(missing) >= this value (default: from the model card)
    --doubt 0.10      parts between --doubt and --threshold are listed as "doubtful" but do not fail the board
    --out DIR         result images / csv (default ml\\presence\\predictions)
    --model FILE      weights (default v3\\final\\presence_cls.pt = the model in the app)
    --card FILE       model card (default v3\\final\\presence_model.json); --no-norm for the older v1 / v2 models
    --golden FILE     golden reference photo (default data\\presence\\golden\\red\\grf.jpeg)

What it does (same steps as the training crops, see make_dataset.py): find the board, align it to the golden reference at
20 px/mm, correct the small local shift of every part with optical flow, cut a square 128 x 128 crop around each of the 86
SMT part boxes (box + 1.5 mm on every side) and let the YOLO11n-cls model say missing / present.

Exit code: 0 = every photo PASS, 1 = at least one FAIL, 2 = at least one RETAKE (and no FAIL).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import golden_compare as gc  # noqa: E402

ML = ROOT / "ml" / "presence"
DEFAULT_MODEL = ML / "v3" / "final" / "presence_cls.pt"        # the model that is also in the app (v3, experiment E10)
DEFAULT_CARD = ML / "v3" / "final" / "presence_model.json"    # its model card: crop rule, exposure normalisation, thresholds
DEFAULT_GOLDEN = ROOT / "data" / "presence" / "golden" / "red" / "grf.jpeg"
REGIONS = ROOT / "android_app" / "assets" / "golden" / "regions.json"
MAX_FLAGGED = 20      # more flagged parts than this = wrong board / colour / bad photo, not twenty missing parts
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


def make_params() -> gc.Params:
    P = gc.Params()
    P.px_per_mm = 20.0          # canonical frame 1660 x 2320 px (83 x 116 mm), as in training
    P.max_input_side = 4000     # do not shrink phone photos before aligning
    P.min_inlier_ratio = 0.10   # 15 usable photos were lost at 0.25 although they had 150+ inliers
    return P


def align(img: np.ndarray, ref, P: gc.Params):
    """Board -> golden frame. If the board touches the photo edge, retry once with a grey margin added in memory."""
    def attempt(im):
        board = gc.detect_board(im, P)
        al = gc.align(board, ref.canonical, ref.valid, P)
        aligned, valid = gc.warp_to_reference(board, al, P)
        if float((valid[ref.area > 0] > 0).mean()) < P.min_coverage:
            raise gc.Retake("board not completely visible")
        gc.board_hue_of(aligned, P)
        return aligned, valid, al
    try:
        return attempt(img) + (False,)
    except gc.Retake as e:
        if "edge" in str(e):
            padded = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=(128, 130, 128))
            return attempt(padded) + (True,)
        raise


def square_crop(image: np.ndarray, cx: float, cy: float, side: int, out_size: int = 128) -> np.ndarray:
    """Identical to extract_square_crop() in make_dataset.py (the function that cut the training crops)."""
    H, W = image.shape[:2]
    half = side / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2, y2 = x1 + side, y1 + side
    pad_top, pad_bottom = max(0, -y1), max(0, y2 - H)
    pad_left, pad_right = max(0, -x1), max(0, x2 - W)
    patch = image[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    patch = patch[:side, :side]
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)


def exposure_gain(aligned: np.ndarray, card: dict) -> float:
    """Exposure normalisation of the v3 model: median colour of the red solder-mask pixels -> luma -> target_luma / luma (clipped).
    The Kotlin code of the app (Presence.exposureGain) does the same."""
    hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 10) | (h >= 170)) & (s >= 120) & (v >= 50)
    if int(mask.sum()) < 20000:
        return 1.0
    med = np.median(aligned[mask], axis=0)
    luma = 0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2]
    return float(np.clip(card["exposure_target_luma"] / luma, card["exposure_gain_min"], card["exposure_gain_max"]))


WINDOWS_9 = [(0.0, 0.0)] + [(dx * s, dy * s) for s in (0.4, 0.8) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]


def part_scores(model, aligned, shifts, parts, P: gc.Params, offsets_mm=((0.0, 0.0),)):
    """-> (score, centre score): per part the MEDIAN of p(missing) over the windows (offsets in mm from the part's local position, the first is
    the centre); with one window it is just the centre score. The Kotlin code of the app (Presence.inspect) does the same."""
    crops = []
    for r in parts:
        x, y, w, h = r.bbox_px
        dx, dy = shifts.get(r.id, (0.0, 0.0))
        side = int(round(max(w, h) + 2 * 1.5 * P.px_per_mm))
        for ox, oy in offsets_mm:
            crops.append(square_crop(aligned, x + w / 2.0 + dx + ox * P.px_per_mm, y + h / 2.0 + dy + oy * P.px_per_mm, side))
    missing_idx = [i for i, n in model.names.items() if n == "missing"][0]
    pr = np.array([float(res.probs.data[missing_idx]) for res in model.predict(crops, imgsz=128, device="cpu", verbose=False)])
    pr = pr.reshape(len(parts), len(offsets_mm))
    return np.median(pr, axis=1), pr[:, 0]


def draw(aligned, parts, shifts, p, thr, doubt, P, headline, colour):
    vis = aligned.copy()
    pad = int(round(1.0 * P.px_per_mm))
    for r, pm in zip(parts, p):
        if pm < doubt:
            continue
        x, y, w, h = r.bbox_px
        dx, dy = shifts.get(r.id, (0.0, 0.0))
        c = (0, 230, 255) if pm >= thr else (255, 229, 0)        # yellow = missing (visible on the red board), cyan = doubtful
        x0, y0 = int(x + dx) - pad, int(y + dy) - pad
        corner = (int(x + dx + w) + pad, int(y + dy + h) + pad)
        cv2.rectangle(vis, (x0, y0), corner, (0, 0, 0), 11)          # black edge first, the coloured box on top
        cv2.rectangle(vis, (x0, y0), corner, c, 8)
        label = f"{r.name} {pm:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 4)
        ty = max(th + 100, y0 - 12)                                  # keep the label below the headline bar
        cv2.rectangle(vis, (x0 - 2, ty - th - 8), (x0 + tw + 8, ty + base + 4), c, -1)
        cv2.putText(vis, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 90), colour, -1)
    cv2.putText(vis, headline, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
    return cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def expand(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_TYPES)
        elif any(ch in it for ch in "*?["):
            out += sorted(Path(f) for f in glob.glob(it))
        else:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="PASS / FAIL / RETAKE for photos of the red LG Indicator board (missing-part check).")
    ap.add_argument("photos", nargs="+", help="photo files, folders or patterns")
    ap.add_argument("--threshold", type=float, default=None, help="default: the threshold of the model card (0.35)")
    ap.add_argument("--doubt", type=float, default=None, help="default: the doubt threshold of the model card (0.10)")
    ap.add_argument("--out", type=Path, default=ML / "predictions")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--card", type=Path, default=DEFAULT_CARD, help="model card json (exposure normalisation, thresholds)")
    ap.add_argument("--no-norm", action="store_true", help="no exposure normalisation (for the older v1 / v2 models)")
    ap.add_argument("--windows", type=int, default=None, choices=(1, 9), help="9 = thorough (median over 9 windows per part), 1 = fast; default: the model card (9)")
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    args = ap.parse_args()
    card = json.loads(args.card.read_text(encoding="utf-8")) if args.card.exists() else {}
    if args.threshold is None:
        args.threshold = float(card.get("threshold", 0.5))
    if args.doubt is None:
        args.doubt = float(card.get("doubt", 0.2))
    use_norm = bool(card.get("exposure_norm")) and not args.no_norm
    if args.windows is None:
        args.windows = int(card.get("windows_default", 1))

    from ultralytics import YOLO
    P = make_params()
    ref = gc.prepare_reference(gc.imread(args.golden), P, gc.load_regions(REGIONS, P))
    parts = [r for r in ref.regions if r.kind == "part"]
    model = YOLO(str(args.model))
    args.out.mkdir(parents=True, exist_ok=True)

    photos = expand(args.photos)
    if not photos:
        print("no photos found")
        return 2
    n_fail = n_retake = 0
    for path in photos:
        try:
            img = gc.imread(path)
            aligned, valid, al, margin_added = align(img, ref, P)
        except gc.Retake as e:
            print(f"{path.name}: RETAKE - {e}")
            n_retake += 1
            continue
        except Exception as e:                      # unreadable file etc.
            print(f"{path.name}: RETAKE - cannot read / align this photo ({e})")
            n_retake += 1
            continue

        rg, ng, _ = gc.comparison_grays(ref.clahe, gc.clahe_gray(aligned), np.bitwise_and(ref.area, valid), P)
        shifts = gc.region_shifts(gc.local_flow(rg, ng), ref, ref.regions, P)
        gain = exposure_gain(aligned, card) if use_norm else 1.0
        src = np.clip(aligned.astype(np.float32) * gain, 0, 255).astype(np.uint8) if use_norm else aligned
        offsets = WINDOWS_9 if args.windows >= 2 else [(0.0, 0.0)]
        p, p_centre = part_scores(model, src, shifts, parts, P, offsets)

        order = np.argsort(-p)
        flagged = [(parts[i].name, float(p[i])) for i in order if p[i] >= args.threshold]
        doubtful = [(parts[i].name, float(p[i])) for i in order if args.doubt <= p[i] < args.threshold]
        if len(flagged) > MAX_FLAGGED:
            verdict, colour = "RETAKE", (0, 140, 255)
            note = f"{len(flagged)} parts flagged - that many missing is not plausible: wrong board colour, bad photo or bad alignment"
            n_retake += 1
        elif flagged:
            verdict, colour = "FAIL", (0, 0, 200)
            note = f"{len(flagged)} part(s) missing: " + ", ".join(f"{n} ({v:.2f})" for n, v in flagged)
            n_fail += 1
        else:
            verdict, colour = "PASS", (0, 140, 0)
            top = parts[int(order[0])].name
            note = f"all {len(parts)} parts present (highest p(missing) {p[order[0]]:.3f} at {top})"
        print(f"{path.name}: {verdict} - {note}")
        if doubtful:
            print("    doubtful, not counted: " + ", ".join(f"{n} ({v:.2f})" for n, v in doubtful))
        if margin_added:
            print("    (the board touched the photo edge: a grey margin was added; keep 1-3 cm of background next time)")

        stem = path.stem
        headline = verdict if verdict == "PASS" else f"{verdict}: " + (", ".join(n for n, _ in flagged[:8]) + (" ..." if len(flagged) > 8 else ""))
        cv2.imwrite(str(args.out / f"{stem}_result.jpg"), draw(aligned, parts, shifts, p, args.threshold, args.doubt, P, headline, colour), [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(args.out / f"{stem}_scores.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["part", "p_missing", "status", "p_centre", "shift_dx_px", "shift_dy_px", "exposure_gain", "windows"])
            for i in order:
                status = "missing" if p[i] >= args.threshold else ("doubtful" if p[i] >= args.doubt else "present")
                sdx, sdy = shifts.get(parts[i].id, (0, 0))
                w.writerow([parts[i].name, f"{p[i]:.4f}", status, f"{p_centre[i]:.4f}", int(sdx), int(sdy), f"{gain:.4f}", len(offsets)])
    print(f"\nresult images + scores: {args.out}")
    return 1 if n_fail else (2 if n_retake else 0)


if __name__ == "__main__":
    sys.exit(main())
