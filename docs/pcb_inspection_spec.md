# PCB Inspection Specification: Current Scope

| | |
|---|---|
| Status | Draft 0.2. Current development scope, from two rounds of verified answers. No dataset, production imaging setup or model exists yet. Experiment 0 (mobile phone) is ready to run. |
| Date | 2026-10-02 |
| Source of facts | Production/PCB mentor, 2026-10-02 (see [production_profile.md](production_profile.md)) |
| Config | [`configs/profiles/lg_indicator/variant_1.yaml`](../configs/profiles/lg_indicator/variant_1.yaml) · [`configs/capture/mobile_phone_e0.yaml`](../configs/capture/mobile_phone_e0.yaml) |
| Related | [requirements.md](requirements.md) · [data_collection_plan.md](data_collection_plan.md) · [experiment_0_camera_visibility.md](experiment_0_camera_visibility.md) |

Every fact in this specification comes from the verified answers. Calculations are labelled as such, and their inputs are stated. Anything not answered is marked **UNKNOWN**; anything ambiguous is marked **NEEDS CLARIFICATION**.

---

## 1. Current scope in one table

| Item | Current scope | Source |
|---|---|---|
| PCB | LG Indicator | PCB-01 |
| Variant / revision | Variant 1 (the only one); single revision reported | PCB-02, PCB-03 |
| Board colour | May vary between blue / red / green, depending on availability; same PCB and same function. **A visual variation, not a variant.** | PCB-18 |
| Board size / thickness | 83 × 116 mm / 1.6 mm | PCB-04, PCB-06 |
| Maximum component height | Approximately 12 mm | PCB-06 |
| Inspected side | Top side | PCB-07 |
| Inspection stages | After SMT/reflow; after THT (a third stage is not confirmed) | PRD-06 |
| Assembly | Mixed assembly, as reported (NEEDS CLARIFICATION: which technologies) | PCB-08 |
| Smallest visual objects | Small resistors (exact package / dimensions UNKNOWN). 1206 was reported earlier but is **not** the confirmed smallest feature. | PCB-29, PCB-10 |
| Holes | Lower-left position: none; remaining 3 positions: holes (type and geometry NEEDS CLARIFICATION) | PCB-17 |
| Conformal coating | None | PCB-19 |
| Board handling | Individual board, not panelized | PCB-05 |
| Loading / positioning | Manual / manual | PRD-13, PRD-09 |
| Existing inspection | Manual inspection | PRD-07 |
| Board identification | None in current scope (no barcode / DataMatrix / serial) | PCB-21, PRD-19 |
| Boards available | ~500 known-good boards; rejected boards available; at least one good and one defective board ready for the first experiment | DATA-01, DATA-02 |
| Operator | Available for ground truth and REVIEW | DATA-06, PRD-17 |
| Prototype camera | Mobile phone, for Experiment 0 only | IMG-01 |
| Production camera / lens / lighting / mount / enclosure | Not decided (UNKNOWN) | IMG-01 to IMG-04 |
| Prototype compute | NVIDIA RTX 5050 GPU system | PRD-24 |
| Image / result storage | Required; local storage allowed; no confidentiality restriction reported | PRD-20, PRD-25, DATA-08 |

## 2. Configuration model: profiles, variations and stages

In **inspection mode** none of the following is typed by the operator. The profile comes from the station configuration (later barcode / MES), the stage from the recipe / station / PLC (source not decided), the board identity from an automatic inspection ID, and colour is never an input (`inspection_mode` in `configs/project.yaml`; [requirements.md](requirements.md), section 2.1). In **development / dataset mode** these values are recorded, mostly automatically.

The current scope is one PCB model, one variant and one revision. The software is not built around this one board. It separates four things:

| Concept | What it is | Where it lives | LG Indicator today |
|---|---|---|---|
| **Inspection profile** | One PCB model + variant: board facts, reference data, component map, defect rules, model version | `configs/profiles/<model>/<variant>.yaml` | `lg_indicator/variant_1` |
| **Visual variation** | Appearance changes of the *same* board that do not change its function | `appearance_variations` in the profile; `colour` recorded per board and per capture | Board colour: blue / red / green (list not confirmed complete) |
| **Inspection stage** | The point in assembly at which the board is photographed | `inspection_stages` in the profile; `inspection_stage` recorded per capture | `after_smt_reflow`, `after_tht` |
| **Capture configuration** | How images are taken (device, settings, quality-check thresholds) | `configs/capture/*.yaml` | `mobile_phone_e0` (Experiment 0 only) |

```
LG Indicator  (profile lg_indicator/variant_1)
  └── Variant 1
        ├── visual variations: colour = blue | red | green      (recorded, not separate models)
        ├── inspection stages: after_smt_reflow | after_tht
        ├── golden / reference data        (none yet; possibly per stage, per colour: decided by experiments)
        ├── component map                  (none yet; needs BOM + pick-and-place file)
        ├── defect rules                   (structure only; thresholds null, so every board gets REVIEW)
        ├── inspection configuration       (null until the production imaging is decided)
        └── model / configuration version  (profile_version 2; no model selected)
```

**Why colour is not a variant.** The mentor stated it is the same PCB with the same function, and only the colour changes with availability.
- Making colour a variant would multiply profiles, references and models without evidence that this is needed.
- Instead colour is recorded as data. The experiments then decide between one model / reference for all colours and colour-specific references or settings ([model_strategy.md](model_strategy.md), section 9).

**Why stage is recorded per capture.** The same physical board may be photographed at both stages. *If* through-hole parts are fitted after SMT/reflow, the board looks different at each stage. Keeping the stage on every capture allows per-stage references later, without forcing that design now.

**Implemented:** profile loading (`pcb_inspection.config.load_profile`); validation of `colour` and `inspection_stage` in the Experiment 0 capture log; `inspection_stage` as a required column in the production-representative capture table.

**Not implemented (deliberately):** runtime switching between profiles, per-colour or per-stage reference stores, multi-profile datasets.

## 3. Inspection scope

### In scope
- Visual inspection of the **top side** of individual LG Indicator boards, at the stages **after SMT/reflow** and **after THT**.
- Every board colour that occurs (blue / red / green, as reported).
- The result per board is PASS / FAIL / REVIEW with reasons (section 6). The operator handles REVIEW boards and provides ground truth.

### Out of the current scope
| Item | Why |
|---|---|
| Bottom side | Inspection side is top only (PCB-07, PRD-14) |
| Board identification / serial traceability | Not in current scope (PCB-21, PRD-19) |
| Conformal-coating inspection | No coating (PCB-19) |
| Line integration (trigger, PLC / MES signalling) | Unknown (PRD-12, PRD-15, PRD-18); not in this phase |
| Hidden solder joints, inner layers, electrical values of unmarked parts | Physically not visible to a 2D camera ([requirements.md](requirements.md), section 1.2) |

### Depends on clarification
- **A third inspection stage** (PRD-06, "all 3"): not confirmed and not invented.
- **Through-hole joints** (PCB-08): *if* through-hole parts are inserted from the top, their solder joints are on the bottom side and outside this scope.

## 4. Defect scope

All 12 project defect categories are relevant at this stage (PCB-24, "all"):

> missing component · damaged component · misaligned component · wrong component · reversed polarity · missing / damaged lead · broken trace · missing trace · damaged PCB · solder defect · foreign object · other anomaly

**Still unknown:**
- Which categories actually occur on this board, and how often.
- Which components are polarized, critical or look-alikes.

**Visibility for this board is not established yet.** Experiment 0 makes the first measurement, with a phone. Categories that are not visible are documented as such.

## 5. Imaging

### 5.1 Prototype vs production
| | Prototype (now) | Production (later) |
|---|---|---|
| Camera | Mobile phone (IMG-01) | UNKNOWN; selected after Experiment 0 |
| Lens, lighting, mount, working distance, enclosure | Phone lens at 1×; room lighting as found and **recorded**; fixed support; distance measured and recorded | All UNKNOWN ("not decided yet") |
| Purpose | Experiment 0: can a simple camera see what must be detected? | Production-representative dataset and inspection |
| Image format | Original phone JPEG at highest quality | Lossless PNG / TIFF |
| Data location | `data/experiment_0/` | `data/raw/`, `data/metadata/` |

**Phone results are never treated as production-ready imaging.** They show what a simple camera can and cannot see, and they set the requirements for the production camera.

### 5.2 Field of view
The board is ≥ 116 × 83 mm, plus a margin on every side.

### 5.3 Resolution needed for different feature sizes
*Calculation, not a requirement.* It uses the planning guideline of 10 px across the smallest feature ([requirements.md](requirements.md), section 8.1) over the board area only (116 × 83 mm). The exact size of the small resistors is UNKNOWN (PCB-29), so several feature sizes are shown:

| Smallest feature that must be seen | mm / px | Pixels (116 × 83 mm) | ≈ Megapixels |
|---|---|---|---|
| 1.6 mm | 0.16 | 725 × 519 | 0.4 |
| 0.5 mm | 0.05 | 2 320 × 1 660 | 3.9 |
| 0.25 mm | 0.025 | 4 640 × 3 320 | 15.4 |
| 0.1 mm | 0.01 | 11 600 × 8 300 | 96 |

**Reading:**
- The real requirement is set by the smallest *feature* that must be seen on the small resistors (body, terminations, markings, polarity, solder), not by a package code.
- Experiment 0 measures how many pixels per mm the phone actually delivers. `scripts/check_e0_images.py` estimates px/mm from the board's known 116 mm length. Once PCB-29 gives the resistor dimensions, the measurement shows whether whole-board photos are enough, or whether close-ups / tiling are needed.

### 5.4 Component height (~12 mm): depth of field and parallax
- **Depth of field.** Phone cameras generally focus on one plane. Experiment 0 checks whether board-level small parts stay sharp when the phone is positioned to see the whole board, and how the tall parts look.
- **Parallax.** The top of a component of height *h* at distance *r* from the image centre appears shifted outward by about `d ≈ h · r / (WD − h)`, where *WD* is the camera-to-board distance.
  - *Illustrative values*, worst case at a board corner (r ≈ 71 mm): WD 300 mm → ≈ 3.0 mm; WD 500 mm → ≈ 1.7 mm.
  - Tall parts can therefore hide neighbouring small parts. Experiment 0 checks this on the real board.

### 5.5 Board colour
The colour varies with availability (PCB-18). Lighting, exposure and the background must work for every colour that occurs, so Experiment 0 captures each available colour under the same setup. The background must be neutral (white or light grey, matte), never blue, red or green.

### 5.6 Positioning and holes
Loading and positioning are manual.
- Experiment 0 uses a simple repeatable placement (e.g. corner guides taped to the background) and one documented reference orientation.
- *If* PCB-17 is confirmed (holes at 3 positions, none at the lower-left), the hole pattern may help define orientation. Its use for fixture design and alignment waits for the clarification.

## 6. Output in the current scope
- **Verdict:** PASS / FAIL / REVIEW per board, with reasons, findings, an annotated image and processing time ([requirements.md](requirements.md), section 7).
- **Identification:** results are keyed by an **inspection ID and timestamp**, because there is no board identification in the current scope.
- **Storage:** local; location and retention period UNKNOWN (PRD-20).
- **REVIEW:** the operator's decision on each REVIEW board is recorded, with the operator's identity.

## 7. What is blocked, and what is not
| Activity | Status |
|---|---|
| Experiment 0 phone capture | **Ready.** Camera-first ingest (`scripts/e0_ingest.py`) tested on copies of the first 4 real photos; a good and a defective board are available. |
| Production camera selection | Waits for Experiment 0 results and PCB-29 (resistor dimensions) |
| Production-representative dataset (v1) | Waits for production imaging, PCB-17 (fixture / alignment), the basis of "known-good" (DATA-01) |
| Annotation | Waits until the production image-acquisition setup is defined |
| Model development | Waits until real images have been collected **and inspected** |

The same gates are defined in `configs/project.yaml` (`readiness_gates`) and evaluated by `python scripts/check_readiness.py`, which lists the exact blocking values of each gate.
