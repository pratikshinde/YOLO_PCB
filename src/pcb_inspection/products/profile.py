"""Product profiles: everything product-specific, loaded from configs/products/<product_id>/.

    product.yaml              identity, physical facts, storage locations, identification
    variants/<variant>.yaml   engine, alignment, quality, and which golden / region / rule files apply
    golden/golden_bank.yaml   the GOOD reference bank (where the GOOD boards come from; not one image)
    regions/regions.yaml      inspection regions (never invented: null when unknown)
    inspection_rules.yaml     enabled defect types, severities, decision thresholds

The code never branches on a product name: `load_product_profile(product_id)` returns the same structure for
every product, and adding a product is adding a folder. null = UNKNOWN; nothing here fills in defaults for
unknown facts (paths only: a product without storage.root stores under data/products/<product_id>).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import find_unresolved, load_config

PRODUCT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")
PRODUCT_FILE = "product.yaml"
VARIANTS_DIR = "variants"
ENGINE_TYPES_NOTE = "engine.type must name a registered inspection backend (pcb_inspection.inspection.engine.BACKENDS)"


class ProfileError(ValueError):
    """A product profile is missing, malformed or inconsistent."""


@dataclass
class ProductProfile:
    product_id: str
    name: str | None
    revision: str | None
    variant_id: str
    variant_label: str | None
    folder: Path                                  # configs/products/<product_id>
    project_root: Path
    product: dict[str, Any]                       # product.yaml
    variant: dict[str, Any]                       # variants/<variant>.yaml
    golden: dict[str, Any]                        # golden bank file
    regions: dict[str, Any]                       # regions file
    rules: dict[str, Any]                         # inspection rules file
    files: list[Path] = field(default_factory=list)

    # --- identity -------------------------------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.product_id}/{self.variant_id}"

    @property
    def display_name(self) -> str:
        label = self.variant_label or self.variant_id
        return f"{self.name or self.product_id} (variant {label})"

    # --- physical board -------------------------------------------------------------------
    @property
    def board(self) -> dict[str, Any]:
        return self.product.get("board") or {}

    @property
    def board_mm(self) -> tuple[float, float] | None:
        """(long side, short side) in mm, or None if unknown."""
        length, width = self.board.get("length_mm"), self.board.get("width_mm")
        if length is None or width is None:
            return None
        return (float(max(length, width)), float(min(length, width)))

    @property
    def stages(self) -> list[str]:
        return list(self.product.get("stages") or [])

    @property
    def colour_variants(self) -> list[str]:
        return list((self.product.get("appearance") or {}).get("colour_variants") or [])

    @property
    def side(self) -> str | None:
        return self.board.get("inspection_side")

    # --- engine / settings ----------------------------------------------------------------
    @property
    def engine(self) -> dict[str, Any]:
        return self.variant.get("engine") or {}

    @property
    def engine_type(self) -> str | None:
        return self.engine.get("type")

    @property
    def alignment(self) -> dict[str, Any]:
        values = dict(self.variant.get("alignment") or {})
        values["max_rotation_deg"] = self.board.get("allowed_rotation_deg")
        return values

    @property
    def quality(self) -> dict[str, Any]:
        return self.variant.get("quality") or {}

    @property
    def decision(self) -> dict[str, Any]:
        return self.rules.get("decision") or {}

    @property
    def enabled_defect_types(self) -> list[str]:
        return sorted(t for t, v in (self.rules.get("defect_types") or {}).items() if (v or {}).get("enabled"))

    def severity(self, defect_type: str) -> str | None:
        return ((self.rules.get("defect_types") or {}).get(defect_type) or {}).get("severity")

    @property
    def auto_identification(self) -> bool:
        return bool((self.product.get("identification") or {}).get("auto"))

    # --- storage --------------------------------------------------------------------------
    @property
    def storage(self) -> dict[str, Any]:
        return self.product.get("storage") or {}

    def path(self, value: str | None) -> Path | None:
        return None if value is None else (self.project_root / value)

    @property
    def storage_root(self) -> Path:
        root = self.storage.get("root")
        return self.project_root / (root or f"data/products/{self.product_id}")

    @property
    def acquisition_root(self) -> Path:
        """Root of the data-collection store (by default acquisition/, curated/, quarantine/; see collection_layout)."""
        return self.path(self.storage.get("acquisition_root")) or self.storage_root

    def collection_layout(self, default: dict[str, str] | None = None) -> dict[str, str]:
        """Folder names inside acquisition_root: acquisition (originals + session.json), curated (KEEP copies; ""
        = directly under <type>/), quarantine (REJECT copies). storage.layout overrides `default`."""
        layout = {"acquisition": "acquisition", "curated": "curated", "quarantine": "quarantine", **(default or {})}
        layout.update({k: v for k, v in (self.storage.get("layout") or {}).items() if k in layout and v is not None})
        return layout

    def collection_dir(self, kind: str, dataset_type: str | None = None) -> Path:
        """acquisition_root / layout folder of `kind` (acquisition | curated | quarantine) [/ dataset_type]."""
        path = self.acquisition_root / self.collection_layout()[kind]
        return path / dataset_type if dataset_type else path

    @property
    def registry_dir(self) -> Path:
        return self.path(self.storage.get("registry")) or self.storage_root / "dataset"

    @property
    def legacy_sources(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self.storage.get("legacy_sources") or []]

    @property
    def inspections_dir(self) -> Path:
        return self.storage_root / "inspections"

    @property
    def reviews_dir(self) -> Path:
        return self.storage_root / "reviews"

    # --- completeness ---------------------------------------------------------------------
    def unknowns(self) -> list[str]:
        """Dotted paths of every UNKNOWN (null) value in the profile files."""
        found = []
        for name, data in (("product", self.product), ("variant", self.variant), ("golden", self.golden),
                           ("regions", self.regions), ("rules", self.rules)):
            found += [f"{name}.{p}" for p in find_unresolved(data)]
        return found

    def summary(self) -> dict[str, Any]:
        return {"product_id": self.product_id, "name": self.name, "variant_id": self.variant_id,
                "revision": self.revision, "board_mm": self.board_mm, "side": self.side, "stages": self.stages,
                "colour_variants": self.colour_variants, "engine": self.engine_type,
                "enabled_defect_types": self.enabled_defect_types, "unknown_values": len(self.unknowns())}


def products_dir_of(project_root: Path) -> Path:
    settings = project_root / "configs" / "inspection.yaml"
    rel = load_config(settings).get("products_dir", "configs/products") if settings.is_file() else "configs/products"
    return project_root / rel


def list_products(products_dir: Path) -> list[str]:
    """Product IDs: every folder of products_dir with a product.yaml (folders starting with '_' are ignored)."""
    if not products_dir.is_dir():
        return []
    return sorted(p.name for p in products_dir.iterdir()
                  if p.is_dir() and not p.name.startswith("_") and (p / PRODUCT_FILE).is_file())


def _load_part(folder: Path, rel: str | None, what: str, files: list[Path]) -> dict[str, Any]:
    if rel is None:
        return {}
    path = folder / rel
    if not path.is_file():
        raise ProfileError(f"{folder.name}: {what} file {rel} does not exist")
    files.append(path)
    return load_config(path)


def load_product_profile(product_id: str, variant_id: str | None = None, *, project_root: Path,
                         products_dir: Path | None = None) -> ProductProfile:
    """The profile of one product variant (default: the product's default_variant)."""
    if not PRODUCT_ID_PATTERN.fullmatch(product_id or ""):
        raise ProfileError(f"invalid product id {product_id!r}: lowercase letters, digits and '_'")
    products_dir = products_dir or products_dir_of(project_root)
    folder = products_dir / product_id
    product_file = folder / PRODUCT_FILE
    if not product_file.is_file():
        known = ", ".join(list_products(products_dir)) or "none"
        raise ProfileError(f"unknown product {product_id!r} (no {product_file}); known products: {known}")
    files = [product_file]
    product = load_config(product_file)
    if product.get("product_id") != product_id:
        raise ProfileError(f"{product_file}: product_id {product.get('product_id')!r} does not match its folder")
    variants = list(product.get("variants") or [])
    variant_id = variant_id or product.get("default_variant")
    if not variant_id or variant_id not in variants:
        raise ProfileError(f"{product_id}: variant {variant_id!r} is not one of {variants}")
    variant = _load_part(folder, f"{VARIANTS_DIR}/{variant_id}.yaml", "variant", files)
    if variant.get("variant_id") != variant_id:
        raise ProfileError(f"{product_id}/{variant_id}: variant_id {variant.get('variant_id')!r} does not match its file")
    profile = ProductProfile(
        product_id=product_id, name=product.get("name"), revision=variant.get("revision") or product.get("revision"),
        variant_id=variant_id, variant_label=variant.get("variant_label"), folder=folder, project_root=project_root,
        product=product, variant=variant,
        golden=_load_part(folder, variant.get("golden"), "golden bank", files),
        regions=_load_part(folder, variant.get("regions"), "regions", files),
        rules=_load_part(folder, variant.get("rules"), "inspection rules", files),
        files=files)
    return profile


def load_all_profiles(project_root: Path, products_dir: Path | None = None) -> tuple[list[ProductProfile], list[str]]:
    """The default variant of every product, plus problems (a broken profile is reported, never skipped silently)."""
    products_dir = products_dir or products_dir_of(project_root)
    profiles, problems = [], []
    for pid in list_products(products_dir):
        try:
            profiles.append(load_product_profile(pid, project_root=project_root, products_dir=products_dir))
        except (ProfileError, OSError, ValueError) as e:
            problems.append(f"{pid}: {e}")
    roots: dict[Path, str] = {}
    for p in profiles:
        for root in {p.storage_root.resolve(), p.acquisition_root.resolve()}:
            if root in roots and roots[root] != p.product_id:
                problems.append(f"{p.product_id} and {roots[root]} share the storage folder {root}: product data would mix")
            roots.setdefault(root, p.product_id)
    return profiles, problems
