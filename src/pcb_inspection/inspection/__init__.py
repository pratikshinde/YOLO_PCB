"""Product-agnostic inspection: InspectionEngine.inspect(image, product_profile, reference_bank).

    types           GOOD / DEFECT / REVIEW / ERROR, Defect, InspectionOutcome
    bank            ReferenceBank: the GOOD boards of one product variant (a bank, not one golden image)
    engine          InspectionEngine, backend registry (BACKENDS), load_reference_bank
    identification  ProductIdentifier: AUTO (registration to each product's own references) or operator selection
    reference_baseline  REFERENCE_BASELINE backend (experimental, classical; Experiment 2 code)
"""

from .bank import BROKEN, INSUFFICIENT, NOT_CONFIGURED, READY, ReferenceBank
from .engine import BACKENDS, InspectionEngine, backend_for, load_reference_bank
from .identification import AUTO, Identification, IdentificationSettings, ProductIdentifier
from .types import DECISIONS, DEFECT, ERROR, GOOD, REVIEW, Defect, InspectionOutcome, ProductMatch, Reason

__all__ = ["AUTO", "BACKENDS", "BROKEN", "DECISIONS", "DEFECT", "ERROR", "GOOD", "INSUFFICIENT", "NOT_CONFIGURED",
           "READY", "REVIEW", "Defect", "Identification", "IdentificationSettings", "InspectionEngine",
           "InspectionOutcome", "ProductIdentifier", "ProductMatch", "Reason", "ReferenceBank", "backend_for",
           "load_reference_bank"]
