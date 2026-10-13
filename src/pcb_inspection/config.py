"""Load the project configuration and inspection profiles; report unresolved values.

configs/project.yaml holds project-wide settings (production line, station,
data protocol, defect taxonomy) and names the active inspection profile.
Everything specific to one PCB model / variant lives in its profile,
configs/profiles/<model>/<variant>.yaml (docs/pcb_inspection_spec.md, section 2).

A null value means UNKNOWN / TO BE CONFIRMED. Ambiguous answers also stay null
and are recorded verbatim under `needs_clarification`. Nothing here
substitutes defaults for those values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

PROFILE_ID_PATTERN = re.compile(r"[a-z0-9_]+/[a-z0-9_]+")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return config


def load_profile(profiles_dir: str | Path, profile_id: str) -> dict[str, Any]:
    """Load the inspection profile '<model>/<variant>' from profiles_dir."""
    if not PROFILE_ID_PATTERN.fullmatch(profile_id):
        raise ValueError(f"invalid profile id {profile_id!r}, expected lowercase '<model>/<variant>'")
    path = Path(profiles_dir) / f"{profile_id}.yaml"
    profile = load_config(path)
    if profile.get("profile_id") != profile_id:
        raise ValueError(f"{path}: profile_id {profile.get('profile_id')!r} does not match its location")
    return profile


def find_unresolved(node: Any, prefix: str = "") -> list[str]:
    """Return the dotted path of every null leaf, e.g. 'camera.settings.exposure_us'."""
    if node is None:
        return [prefix]
    if isinstance(node, dict):
        found = []
        for key, value in node.items():
            found += find_unresolved(value, f"{prefix}.{key}" if prefix else str(key))
        return found
    if isinstance(node, list):
        found = []
        for index, value in enumerate(node):
            found += find_unresolved(value, f"{prefix}[{index}]")
        return found
    return []


def get_value(config: Mapping[str, Any], dotted_path: str) -> Any:
    """Return the value at 'a.b.c'. Raises KeyError if the path does not exist."""
    node: Any = config
    for key in dotted_path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise KeyError(dotted_path)
        node = node[key]
    return node


def gate_blockers(requires: Iterable[str], project: Mapping[str, Any], profile: Mapping[str, Any]) -> list[str]:
    """Return the required paths of a readiness gate that are still unknown (null, false or empty).

    Paths starting with 'profile:' are looked up in the profile, all others in the
    project config. A path that does not exist raises KeyError (a typo in the gate).
    """
    blockers = []
    for ref in requires:
        config, path = (profile, ref.removeprefix("profile:")) if ref.startswith("profile:") else (project, ref)
        value = get_value(config, path)
        if value is None or value is False or value == [] or value == {}:
            blockers.append(ref)
    return blockers
