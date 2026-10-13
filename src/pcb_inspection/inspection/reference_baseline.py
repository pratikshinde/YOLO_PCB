"""REFERENCE_BASELINE: the experimental classical inspection backend (Experiment 2 code, product-agnostic wrapper).

    capture -> frame valid? -> watermark located + inpainted -> PCB present? -> SIFT/RANSAC registration to the
    product's geometric golden reference -> alignment gate -> canonical frame -> every region of the product's
    region map compared with the product's GOOD reference BANK (all GOOD boards; leave-one-board-out) -> decision

It is a REFERENCE BASELINE, not trained AI and not a validated inspection: one measured similarity per region,
compared with a limit taken from GOOD boards. Results inside the uncertainty band around the limit are REVIEW.
Only `missing_component` can be detected. Algorithm parameters (registration, watermark, component check) come
from the engine config named in the profile; ALL product data (board size, golden reference, regions, board
registry, data store, alignment limits, decision rules) comes from the product profile, so the same code serves
every product and never uses another product's references.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..alignment.registration import register
from ..alignment.validation import validate_alignment
from ..comparison import board_inspection as bi
from ..comparison.components import MISSING, NOT_CHECKED, ComponentMap
from ..dataset.registry import load_registry
from ..preprocessing.presence import NO_PCB as PRESENCE_NO_PCB
from ..preprocessing.presence import VALID, check_frame, detect_pcb
from ..preprocessing.watermark import locate_watermark, remove_watermark
from ..products.profile import ProductProfile
from ..visualization import panels
from .bank import BROKEN, INSUFFICIENT, NOT_CONFIGURED, READY, ReferenceBank
from .types import (
    ALIGN_FAILED,
    ALIGN_NOT_RUN,
    ALIGN_OK,
    ALIGNMENT,
    DEFECT,
    GOOD,
    IMAGE_QUALITY,
    IMPLAUSIBLE,
    NO_PCB,
    NO_REGIONS,
    NOT_CHECKABLE,
    PROFILE_INCOMPLETE,
    REVIEW,
    UNCERTAIN,
    Defect,
    ProductMatch,
    Reason,
)

NAME = "REFERENCE_BASELINE"
MEASURE = "best similarity of the region to any GOOD board of the bank (1 = identical); limit from GOOD boards"


@dataclass
class BaselineBank:
    cfg: bi.InspectionConfig
    reference: bi.InspectionReference
    cmap: ComponentMap
    gallery: bi.GoodGallery
    limits: dict[str, float] | None
    limits_note: str


@dataclass
class BaselineRun:
    """What write_evidence needs from one inspection."""
    image: np.ndarray
    result: bi.InspectionResult
    bank: BaselineBank


class ReferenceBaselineBackend:
    name = NAME

    # --- golden reference bank --------------------------------------------------------------
    def load_bank(self, profile: ProductProfile) -> ReferenceBank:
        bank = ReferenceBank(profile.product_id, profile.variant_id, NAME, NOT_CONFIGURED, "")
        golden, root = profile.golden, profile.project_root
        missing = [what for what, value in (
            ("golden_bank.yaml geometric_reference", golden.get("geometric_reference")),
            ("regions.yaml component_regions.file", (profile.regions.get("component_regions") or {}).get("file")),
            ("product.yaml board length_mm / width_mm", profile.board_mm),
            ("variant engine.config", profile.engine.get("config")))
            if value is None]
        if missing:
            bank.reason = "not configured: " + ", ".join(missing) + " UNKNOWN"
            return bank
        bank.min_boards = golden.get("min_boards")
        bank.geometric_reference = golden["geometric_reference"]
        try:
            cfg = self._config(profile)
            bank.data_roots = [str(cfg.data_root)]
            ref_dir = cfg.cache_dir / "reference" / cfg.reference_file.stem
            reference, cmap = bi.load_reference_and_map(cfg, ref_dir)
            bank_rotation = golden.get("max_rotation_deg")
            bank_cfg = cfg if bank_rotation is None else replace(
                cfg, alignment=replace(cfg.alignment, max_rotation_deg=float(bank_rotation)))
            gallery = bi.load_gallery(bank_cfg, reference)
            limits, note = bi.load_calibration(cfg, gallery)
        except (ValueError, OSError, KeyError, TypeError) as e:
            bank.status, bank.reason = BROKEN, f"golden reference / regions could not be loaded: {e}"
            return bank
        bank.data = BaselineBank(cfg, reference, cmap, gallery, limits, note)
        bank.boards = gallery.board_ids
        bank.images = len(gallery.names)
        bank.skipped = list(gallery.skipped)
        bank.regions = len(cmap.components)
        bank.unverified_regions = len(cmap.unverified)
        if cfg.registry_file and cfg.registry_file.is_file():
            registry = load_registry(cfg.registry_file, cfg.image_boards_file)
            bank.unverified_boards = sorted(b for b in bank.boards if b in registry.boards and registry.boards[b].provisional)
        if bank.min_boards is None:
            bank.status, bank.reason = INSUFFICIENT, "golden_bank.yaml min_boards is UNKNOWN: sufficiency cannot be judged"
        elif len(bank.boards) < int(bank.min_boards):
            bank.status = INSUFFICIENT
            bank.reason = (f"{len(bank.boards)} physical GOOD board(s) in the bank ({', '.join(bank.boards) or 'none'}); "
                           f"at least {bank.min_boards} needed")
        else:
            bank.status = READY
            bank.reason = f"{len(bank.boards)} physical GOOD boards, {bank.images} images; {note}"
        return bank

    def _config(self, profile: ProductProfile) -> bi.InspectionConfig:
        """Engine (algorithm) config with every product-specific value replaced from the profile."""
        root, golden = profile.project_root, profile.golden
        cfg = bi.load_inspection_config(root, root / profile.engine["config"])
        long_mm, short_mm = profile.board_mm
        reference = root / golden["geometric_reference"]
        good_dirs = (profile.collection_dir("curated", "good"), profile.collection_dir("acquisition", "good"))
        if not any(reference.resolve().is_relative_to(d.resolve()) for d in good_dirs):
            raise ValueError(f"geometric_reference must be a GOOD image of this product's collection "
                             f"({', '.join(str(d) for d in good_dirs)}), got {golden['geometric_reference']}")
        regions_file = root / profile.regions["component_regions"]["file"]
        alignment = {k: v for k, v in profile.alignment.items() if v is not None
                     and k in bi.AlignmentValidationParams.__dataclass_fields__}
        cache = golden.get("cache_dir")
        cache_dir = root / cache if cache else profile.storage_root / "cache" / NAME.lower()
        calibration = golden.get("calibration")
        max_defects = profile.decision.get("max_plausible_defects")
        return replace(
            cfg,
            pipeline=replace(cfg.pipeline, board_mm=(short_mm, long_mm)),
            alignment=replace(cfg.alignment, **alignment),
            max_plausible_defects=int(max_defects) if max_defects is not None else cfg.max_plausible_defects,
            reference_file=reference, component_map_file=regions_file,
            registry_file=profile.registry_dir / "boards.csv", image_boards_file=profile.registry_dir / "image_boards.csv",
            reference_split=golden.get("reference_split") or "reference",
            calibration_file=root / calibration if calibration else None,
            calibration_min_boards=int(golden.get("calibration_min_boards") or cfg.calibration_min_boards),
            cache_dir=cache_dir, input_dir=profile.storage_root, output_dir=profile.storage_root,
            experiment=f"{profile.product_id}/{profile.variant_id} {NAME}",
            data_root=profile.acquisition_root, extra_data_roots=(),
            acquisition_dir=profile.collection_layout()["acquisition"])

    # --- product identification evidence ----------------------------------------------------
    def match(self, image: np.ndarray, profile: ProductProfile, bank: ReferenceBank) -> ProductMatch:
        """How well this image registers to the product's geometric golden reference (no inspection)."""
        d: BaselineBank = bank.data
        base = dict(product_id=profile.product_id, variant_id=profile.variant_id)
        frame = check_frame(image, d.cfg.presence)
        if frame.status != VALID:
            return ProductMatch(**base, status="INVALID_FRAME", reason=frame.reason)
        wm = locate_watermark(image, d.reference.template, d.cfg.watermark)
        clean = remove_watermark(image, wm)
        presence = detect_pcb(clean, d.cfg.presence, d.cfg.pipeline.segmentation)
        if presence.status == PRESENCE_NO_PCB:
            return ProductMatch(**base, status="NO_PCB", reason=presence.reason)
        registration = register(d.reference.image, d.reference.mask, clean, presence.mask, d.cfg.pipeline.registration)
        check = validate_alignment(registration, d.reference.quad, d.reference.frame.source_px_per_mm, image.shape,
                                   d.cfg.alignment)
        return ProductMatch(**base, status=check.status, inliers=registration.inliers,
                            inlier_ratio=round(float(registration.inlier_ratio), 3),
                            fit_error_mm=None if check.fit_error_mm is None else round(check.fit_error_mm, 3),
                            reason=check.reason or registration.message, registration=registration)

    # --- inspection ------------------------------------------------------------------------
    def inspect(self, image: np.ndarray, profile: ProductProfile, bank: ReferenceBank, board_id: str | None = None,
                match: ProductMatch | None = None) -> tuple[dict[str, Any], BaselineRun]:
        d: BaselineBank = bank.data
        registration = match.registration if match is not None and match.status in ("OK", "POSITION_INVALID") else None
        exclude = board_id if board_id and board_id in d.gallery.board_ids else None
        r = bi.inspect_image(image, d.reference, d.cmap, d.cfg, d.gallery, exclude, d.limits, d.limits_note,
                             registration=registration)
        run = BaselineRun(image, r, d)
        check = r.alignment_check
        align_error = None if check is None else {
            "status": check.status, "reason": check.reason, "inlier_ratio": round(check.inlier_ratio, 3),
            "inliers": r.registration.inliers if r.registration else None,
            "fit_error_mm": None if check.fit_error_mm is None else round(check.fit_error_mm, 3),
            "rotation_deg": None if check.rotation_deg is None else round(check.rotation_deg, 1),
            "corner_angle_deviation_deg": None if check.corner_angle_deviation_deg is None
            else round(check.corner_angle_deviation_deg, 1),
            "board_width_fraction": None if check.board_width_fraction is None else round(check.board_width_fraction, 3)}
        out: dict[str, Any] = dict(alignment_status=ALIGN_NOT_RUN, alignment_error=align_error,
                                   details={"engine_stages": r.stages, "engine_status": r.status,
                                            "good_examples": r.examples, "limits": r.limits_note,
                                            "left_out_board": exclude})
        if r.status == bi.INVALID_FRAME:
            return {**out, "decision": REVIEW, "reasons": [Reason(IMAGE_QUALITY, f"frame not usable: {r.reason}")]}, run
        if r.status == bi.NO_PCB:
            return {**out, "decision": REVIEW, "reasons": [Reason(NO_PCB, f"no PCB detected: {r.reason}")]}, run
        if r.status in (bi.ALIGNMENT_FAILED, bi.POSITION_INVALID):
            return {**out, "alignment_status": ALIGN_FAILED, "decision": REVIEW,
                    "reasons": [Reason(ALIGNMENT, f"alignment outside tolerance, not inspected: {r.reason}")]}, run
        out["alignment_status"] = ALIGN_OK
        out["regions_checked"] = sum(1 for c in r.components if c.status != NOT_CHECKED)
        if r.status == bi.INSPECTION_ERROR:
            return {**out, "decision": REVIEW, "reasons": [Reason(IMPLAUSIBLE, r.reason)]}, run
        return {**out, **self._decide(r, d, profile)}, run

    def _decide(self, r: bi.InspectionResult, d: BaselineBank, profile: ProductProfile) -> dict[str, Any]:
        rules = profile.decision
        band = rules.get("uncertainty_band")
        notes = [] if band is not None else ["no uncertainty band configured (inspection_rules.yaml decision.uncertainty_band "
                                             "UNKNOWN): limits applied without a margin"]
        band = float(band or 0.0)
        min_visible = rules.get("min_region_visible_fraction")
        ppm = d.cfg.pipeline.px_per_mm
        definite, uncertain, unchecked = [], [], []
        for c in r.components:
            if c.status == NOT_CHECKED or c.similarity is None or c.limit is None:
                unchecked.append({"region": c.refdes, "reason": c.reason or "not checkable"})
                continue
            visible = _visible_fraction(r.usable, c.box_px)
            if min_visible is not None and visible is not None and visible < float(min_visible):
                # partly hidden (watermark / ignored pixels / outside the aligned view): neither GOOD nor DEFECT evidence
                unchecked.append({"region": c.refdes, "reason": f"partly hidden: {visible:.0%} visible "
                                  f"(needs {float(min_visible):.0%})", "engine_status": c.status,
                                  "similarity": round(c.similarity, 3), "limit": round(c.limit, 3)})
                continue
            margin = c.similarity - c.limit
            if c.status == MISSING and margin <= -band:
                definite.append(c)
            elif abs(margin) < band:
                uncertain.append({"region": c.refdes, "similarity": round(c.similarity, 3), "limit": round(c.limit, 3),
                                  "engine_status": c.status, "box_mm": _box_mm(c.box_px, ppm)})
        result: dict[str, Any] = {"uncertain_regions": uncertain, "limitations": notes}
        if not r.components:
            return {**result, "decision": REVIEW, "reasons": [Reason(NO_REGIONS, "the region map has no regions")]}
        if definite:
            if "missing_component" not in profile.enabled_defect_types:
                return {**result, "decision": REVIEW, "reasons": [Reason(
                    PROFILE_INCOMPLETE, "regions differ from every GOOD board, but missing_component is not an enabled "
                    "defect type of this product: " + ", ".join(c.refdes for c in definite))]}
            defects = [self._defect(n, c, ppm, band, profile) for n, c in enumerate(definite, 1)]
            reasons = [Reason("DEFECT_FOUND", ", ".join(f"{c.refdes} missing" for c in definite))]
            if uncertain or unchecked:
                reasons.append(Reason(UNCERTAIN, f"also {len(uncertain)} uncertain and {len(unchecked)} not checkable "
                                                 "region(s): see uncertain_regions / details"))
            return {**result, "decision": DEFECT, "defects": defects, "reasons": reasons,
                    "details_extra": {"not_checkable": unchecked}}
        if uncertain:
            return {**result, "decision": REVIEW, "reasons": [Reason(UNCERTAIN, (
                f"{len(uncertain)} region(s) within +-{band:g} of the limit, too close to decide: "
                + ", ".join(u["region"] for u in uncertain)))], "details_extra": {"not_checkable": unchecked}}
        if unchecked:
            return {**result, "decision": REVIEW, "reasons": [Reason(NOT_CHECKABLE, (
                f"{len(unchecked)} region(s) could not be checked: "
                + ", ".join(f"{u['region']} ({u['reason']})" for u in unchecked)))],
                "details_extra": {"not_checkable": unchecked}}
        if d.cmap.unverified and not rules.get("allow_good_with_unverified_regions"):
            return {**result, "decision": REVIEW, "reasons": [Reason(PROFILE_INCOMPLETE, (
                f"all regions matched, but {len(d.cmap.unverified)} regions are unverified drafts and this product "
                "does not allow GOOD on unverified regions"))]}
        return {**result, "decision": GOOD, "reasons": [Reason("ALL_REGIONS_MATCH", (
            f"all {len(r.components)} regions match a GOOD board of the bank by more than {band:g}"))]}

    def _defect(self, n: int, c, ppm: float, band: float, profile: ProductProfile) -> Defect:
        box = _box_mm(c.box_px, ppm)
        return Defect(
            defect_id=f"D{n}", defect_type="missing_component", confidence=None,
            location={"frame": "canonical_mm (origin top-left of the board, x right, y down)", "box_mm": box,
                      "center_mm": [round((box[0] + box[2]) / 2, 2), round((box[1] + box[3]) / 2, 2)]},
            region=c.refdes,
            evidence={"measure": MEASURE, "similarity": round(c.similarity, 3), "limit": round(c.limit, 3),
                      "difference": round(c.score, 3) if c.score is not None else None,
                      "below_limit_by": round(c.limit - c.similarity, 3), "uncertainty_band": band,
                      "region_verified": c.verified, "region_source": c.source, "image": "evidence.jpg"},
            severity=profile.severity("missing_component"))

    # --- evidence files ----------------------------------------------------------------------
    def write_evidence(self, run: BaselineRun, out_dir: Path) -> dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        r, d, files = run.result, run.bank, {}
        (out_dir / "report.txt").write_text(r.report + "\n", encoding="utf-8")
        files["report"] = "report.txt"
        if run.image is not None and run.image.size and r.status != bi.INVALID_FRAME:
            panels.write_jpeg(out_dir / "overview.jpg", bi.overview(run.image, r))
            files["overview"] = "overview.jpg"
        if r.aligned is not None:
            panels.write_jpeg(out_dir / "aligned.jpg", r.aligned)
            files["aligned"] = "aligned.jpg"
            defect_img = bi.defect_location_overlay(r, d.cfg.pipeline.px_per_mm)
            if defect_img is not None:
                panels.write_jpeg(out_dir / "defect_location.jpg", defect_img)
                files["defect_location"] = "defect_location.jpg"
            panels.write_jpeg(out_dir / "annotated.jpg", bi.annotate(r, d.cfg.pipeline.px_per_mm))
            files["annotated"] = "annotated.jpg"
            proof = bi.evidence(r, d.reference, d.cfg.pipeline.px_per_mm)
            if proof is not None:
                panels.write_jpeg(out_dir / "evidence.jpg", proof)
                files["evidence"] = "evidence.jpg"
        return files

    def limitations(self, bank: ReferenceBank) -> list[str]:
        d: BaselineBank = bank.data
        notes = [f"{NAME} is an experimental classical baseline (Experiment 2), not a validated inspection; "
                 "only missing_component can be detected"]
        if d is not None and d.cmap.unverified:
            notes.append(f"region map: {len(d.cmap.unverified)} of {len(d.cmap.components)} regions are unverified "
                         "drafts; parts outside the map are not inspected")
        if bank.unverified_boards:
            notes.append("golden bank boards with unverified identity: " + ", ".join(bank.unverified_boards))
        return notes


def _visible_fraction(usable: np.ndarray | None, box_px: tuple[int, int, int, int]) -> float | None:
    if usable is None:
        return None
    x0, y0, x1, y1 = box_px
    part = usable[max(0, y0):y1, max(0, x0):x1]
    return float(part.mean()) if part.size else 0.0


def _box_mm(box_px: tuple[int, int, int, int], ppm: float) -> list[float]:
    return [round(v / ppm, 2) for v in box_px]
