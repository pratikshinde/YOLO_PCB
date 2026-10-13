from pathlib import Path

import pytest

from pcb_inspection.config import find_unresolved, gate_blockers, get_value, load_config, load_profile

ROOT = Path(__file__).resolve().parents[1]


def test_find_unresolved_reports_dotted_paths():
    config = {"pcb": {"model": None, "dimensions_mm": {"length": None, "width": 100}}, "rules": [{"t": None}, {"t": 1}]}
    assert find_unresolved(config) == ["pcb.model", "pcb.dimensions_mm.length", "rules[0].t"]


def test_empty_containers_are_not_unresolved():
    assert find_unresolved({"a": [], "b": {}, "c": False, "d": 0}) == []


def test_get_value():
    config = {"a": {"b": {"c": None}}}
    assert get_value(config, "a.b.c") is None
    with pytest.raises(KeyError):
        get_value(config, "a.x.c")


def test_load_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_load_profile_checks_id_and_location(tmp_path):
    (tmp_path / "board_x").mkdir()
    (tmp_path / "board_x" / "variant_a.yaml").write_text("profile_id: board_x/variant_b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_profile(tmp_path, "board_x/variant_a")
    with pytest.raises(ValueError, match="invalid profile id"):
        load_profile(tmp_path, "Board X")


def test_every_listed_profile_loads(project_config):
    profiles = project_config["profiles"]
    assert profiles["active"] in profiles["available"]
    for profile_id in profiles["available"]:
        assert load_profile(ROOT / profiles["directory"], profile_id)["profile_id"] == profile_id


def test_protocol_decisions(project_config):
    assert project_config["data"]["split"]["group_by"] == "board_id"
    acquisition = project_config["acquisition"]
    assert not {"jpg", "jpeg"} & set(acquisition["allowed_image_formats"])
    assert acquisition["auto_exposure"] is False
    assert acquisition["auto_gain"] is False
    assert acquisition["auto_white_balance"] is False


def test_profile_rules_match_taxonomy(project_config, active_profile):
    taxonomy = set(project_config["defects"]["taxonomy"])
    ruled = {rule["defect_type"] for rule in active_profile["decision_engine"]["rules"]}
    assert ruled == taxonomy


MUST_STAY_UNKNOWN = {
    "project": [
        "station.production.camera.model",        # production camera
        "station.production.lens",                # lens
        "station.production.lighting",            # production lighting
        "station.production.working_distance_mm", # working distance
        "station.production.mounting",            # camera mount
        "station.production.enclosure",           # enclosure
        "production.throughput_boards_per_hour",  # production throughput
        "production.boards_per_day",
        "production.inspection_time_budget_s",    # inspection time budget
        "production.escape_rate_target",          # acceptance thresholds
        "production.false_call_rate_target",
        "production.esd_rules.company_policy",    # not "no policy"
        "station.prototype.compute.gpu_memory_gb",
    ],
    "profile": [
        "pcb.smallest_feature_to_detect_mm",      # exact smallest physical feature
        "pcb.smallest_package_size_mm",
        "cad_data.bom",                           # BOM
        "cad_data.centroid_file",                 # pick-and-place
        "cad_data.gerber_or_odb",                 # Gerber / ODB++
        "defect_scope.frequency",                 # exact defect frequencies
        "pcb.acceptance_standard",                # acceptance criteria
        "imaging.working_distance_mm",
        "decision_engine.required_stages",
        "inspection_stages.third_stage",
        "pcb.holes.feature_type",
        "pcb.component_count",
        "data_availability.controlled_defects_permitted",
    ],
}


@pytest.mark.parametrize("which", ["project", "profile"])
def test_unverified_facts_remain_unknown(which, project_config, active_profile):
    config = project_config if which == "project" else active_profile
    for path in MUST_STAY_UNKNOWN[which]:
        assert get_value(config, path) is None, f"{path} must stay UNKNOWN (null)"


def test_no_decision_threshold_is_set(active_profile):
    for rule in active_profile["decision_engine"]["rules"]:
        assert rule["fail_threshold"] is None and rule["review_threshold"] is None, rule


def test_gate_blockers():
    project = {"a": {"x": 1, "empty": [], "off": False, "unknown": None}}
    profile = {"pcb": {"model": "M"}}
    assert gate_blockers(["a.x", "profile:pcb.model"], project, profile) == []
    assert gate_blockers(["a.empty", "a.off", "a.unknown"], project, profile) == ["a.empty", "a.off", "a.unknown"]
    with pytest.raises(KeyError):
        gate_blockers(["a.typo"], project, profile)


def test_shipped_readiness_gates(project_config, active_profile):
    gates = {name: gate_blockers(g["requires"], project_config, active_profile)
             for name, g in project_config["readiness_gates"].items()}
    assert gates["experiment_0_phone_capture"] == []
    assert "profile:pcb.smallest_feature_to_detect_mm" in gates["production_camera_selection"]
    assert "station.production.camera.model" in gates["production_dataset_v1"]
    assert "progress.production_dataset_v1_collected" in gates["model_development"]
    assert "production.escape_rate_target" in gates["model_development"]


def test_inspection_mode_needs_no_operator_metadata(project_config):
    mode = project_config["inspection_mode"]
    assert mode["operator_inputs"] == []
    assert mode["colour"] == "not_an_input"
    assert mode["stage_source"] is None  # not decided yet; never invented


def test_phone_is_prototype_only(project_config):
    station = project_config["station"]
    assert station["prototype"]["camera"] == "mobile_phone"
    assert station["production"]["camera"]["model"] is None


def test_colour_is_a_visual_variation_not_a_variant(active_profile):
    assert active_profile["pcb"]["variant_count"] == 1
    colour = active_profile["appearance_variations"]["board_colour"]
    assert colour["varies"] is True
    assert colour["list_complete"] is None


def test_split_not_created_and_grouped_by_board(project_config):
    split = project_config["data"]["split"]
    assert split["created"] is False
    assert split["group_by"] == "board_id"
    assert all(v is None for v in split["ratios"].values())


def test_phone_capture_config(project_config):
    capture = load_config(ROOT / project_config["station"]["prototype"]["capture_config"])
    assert capture["purpose"] == "experiment_0_camera_visibility"
    assert capture["profile"] in project_config["profiles"]["available"]
    assert not {"heic", "dng"} & set(capture["allowed_image_formats"])
    settings = capture["phone_settings"]
    assert settings["digital_zoom"] is False and settings["flash"] is False
    assert settings["portrait_background_blur"] is False and settings["filters"] is False
    assert capture["device"]["model"] is None


@pytest.mark.parametrize("which", ["project", "profile"])
def test_clarification_items_are_recorded_and_their_values_stay_null(which, project_config, active_profile):
    config = project_config if which == "project" else active_profile
    for item in config.get("needs_clarification", []):
        assert item["reported"] and item["clarify"] and item["source"]
        for path in item["affects"]:
            assert get_value(config, path) is None, f"{item['id']}: {path} must stay null until clarified"
