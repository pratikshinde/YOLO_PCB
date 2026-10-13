# Production Profile: LG Indicator

| | |
|---|---|
| Source | Production/PCB mentor, 2026-10-02 (round 1 and round 2) |
| Date | 2026-10-02 |
| Config | Profile [`configs/profiles/lg_indicator/variant_1.yaml`](../configs/profiles/lg_indicator/variant_1.yaml) · line and station [`configs/project.yaml`](../configs/project.yaml) · phone prototype [`configs/capture/mobile_phone_e0.yaml`](../configs/capture/mobile_phone_e0.yaml) |
| Related | [production_questions.md](production_questions.md) (full questions and answers) · [pcb_inspection_spec.md](pcb_inspection_spec.md) (current inspection scope) |

This document separates three kinds of information:
- **KNOWN**: verified by the mentor.
- **NEEDS CLARIFICATION**: an answer exists, but its meaning is not fully certain.
- **UNKNOWN**: no answer yet, or "not decided yet".

Nothing here is inferred. Where an answer leaves part of a question open, the open part is listed under UNKNOWN.

---

## 1. KNOWN (verified)

### The board

| Item | Verified value (as reported) | ID |
|---|---|---|
| PCB model | LG Indicator | PCB-01 |
| Number of variants | 1 | PCB-02 |
| Board revisions | Single revision reported | PCB-03 |
| Board dimensions | 83 × 116 mm | PCB-04 |
| Board / panel at inspection | Individually (not panelized) | PCB-05 |
| Board thickness | 1.6 mm | PCB-06 |
| Maximum component height | Approximately 12 mm (side not stated) | PCB-06 |
| Inspection side | Top side | PCB-07 |
| Assembly technology | Mixed assembly (meaning to be confirmed, see section 2) | PCB-08 |
| Holes | Lower-left position: no hole. The remaining 3 positions: holes. (What the holes are is NOT known, see section 2.) | PCB-17 |
| **Board colour** | **The same PCB; depending on availability, the colour may change; the working / function of the PCB stays the same.** Colours may include blue / red / green (from the first answer). | PCB-18 |
| Conformal coating | No | PCB-19 |
| Board identification | No barcode / DataMatrix / serial-number identification in the current scope; may be considered in future | PCB-21, PRD-19 |
| Defect categories | All previously listed target categories are considered relevant at this stage. This does **not** mean they are known to occur at known frequencies. | PCB-24 |
| Small resistors | Small resistors are among the smallest visual objects that must be considered ("resistors like too small") | PCB-29 |

### Inspection stages and the production line

| Item | Verified value (as reported) | ID |
|---|---|---|
| **Inspection stages** | **After SMT/reflow**, and **after THT** (a third stage is not confirmed) | PRD-06 |
| Current scope | 1 PCB variant. The software stays configurable for future models and variants. | PRD-02 |
| Existing inspection | Manual inspection | PRD-07 |
| PCB positioning | Manual | PRD-09 |
| Loading | Manual | PRD-13 |
| Both-side inspection | Not applicable (top side only, unless later changed) | PRD-14 |
| Operator | Available | PRD-17 |
| Result / image storage | Required | PRD-20 |
| IT / security | No restrictions reported. Images and results can be stored locally. Cloud requirements are not currently defined. | PRD-25 |
| ESD / handling | Mentor: "No rules." Recorded as: no special restrictions were reported by the mentor. (Not a statement that the company has no ESD policy.) | PRD-27, DATA-09 |

### Imaging and compute (prototype)

| Item | Verified value (as reported) | ID |
|---|---|---|
| **Prototype camera** | **Mobile phone**, for Experiment 0 (camera visibility / data acquisition) only. **Not** the production camera. | IMG-01 |
| Prototype compute | NVIDIA RTX 5050 GPU system (GPU memory and other specifications not provided) | PRD-24 |

### Data access

| Item | Verified value (as reported) | ID |
|---|---|---|
| Known-good boards | Approximately 500 known-good boards available. A real good board is available for the first capture experiment. | DATA-01 |
| Rejected / defective boards | Available. A defective board is available for the first capture experiment. | DATA-02 |
| Ground truth | Operator will provide / verify ground truth | DATA-06 |
| Pilot camera station near the line | Possible | DATA-07 |
| Confidentiality | No confidentiality restrictions reported; images can be stored locally | DATA-08 |

## 2. NEEDS CLARIFICATION

Each answer is recorded **verbatim**. The corresponding config values stay `null` until it is clarified; a unit test enforces this. **None of these items blocks the Experiment 0 phone capture.**

| ID | Reported answer (verbatim) | What must be clarified | Holds up |
|---|---|---|---|
| PCB-17 | Round 1: "yes and expect down left every position has the holes".<br>Round 2: "There are no holes in the lower-left side. The remaining 3 positions have holes." | • Whether the holes are fiducials, tooling holes, mounting holes or another feature (**not assumed**).<br>• Exact coordinates and geometry.<br>• What the four "positions" are, and from which viewing orientation "lower-left" is meant. | Fixture design; choice of alignment method |
| PRD-06 | Round 1: "all 3". Round 2: after SMT/reflow; after THT. | Whether a third stage exists, and which. **Status: PARTIALLY CLARIFIED.** | Nothing yet |
| PCB-29 | "resistors like too small" | Which resistors are meant, and their exact package / physical dimensions (from the BOM or PCB documentation; **not assumed**). | The production camera resolution requirement |
| PCB-10 | "1206" | Whether 1206 really is the smallest package (PCB-29 names small resistors as the smallest objects; their package is unknown); the size-code system (imperial 1206 is 3.2 × 1.6 mm nominal; metric codes differ). 1206 is **not** treated as the confirmed smallest feature. | Nothing yet |
| PCB-08 | "Mixed assembly" | Which technologies are mixed (e.g. SMT + THT), which soldering processes are used, and where THT solder joints are, if any. | Which solder defects are visible from the top |

## 3. UNKNOWN

### The board
| Topic | IDs |
|---|---|
| How the revision is marked; outline drawing | PCB-03, PCB-04 |
| Which sides carry components | PCB-07 |
| Number of components | PCB-09 |
| Other component / package types; hidden-joint packages | PCB-10 |
| **Exact package / dimensions of the smallest resistors** | PCB-29 |
| BOM, pick-and-place file, Gerber / ODB++, assembly drawing (not assumed available) | PCB-11, PCB-12, PCB-22 |
| Polarized, critical, look-alike and alternate components | PCB-13 to PCB-16 |
| Whether blue / red / green is the complete list of colours; share of each colour | PCB-18 |
| Surface finish, silkscreen colour | PCB-18 |
| Relevant outer traces, trace width and spacing | PCB-20 |
| Acceptance standard, placement tolerances | PCB-23 |
| **Actual defect frequency / Pareto**, overall defect rate, defects passing electrical test | PCB-24, PCB-25, PCB-26 |
| Warpage, acceptable visual variation | PCB-27, PCB-28 |

### The production line
| Topic | IDs |
|---|---|
| A possible third inspection stage | PRD-06 |
| Boards per day, electrical test time, inspection time budget, throughput | PRD-01, PRD-03, PRD-04, PRD-05 |
| Variant changeover and identification (for future variants) | PRD-02 |
| What manual inspection covers; why a new system is wanted | PRD-07 |
| Station space, conveyor, ambient lighting, trigger | PRD-08, PRD-10, PRD-11, PRD-12 |
| Whether board orientation is guaranteed when placed manually | PRD-09 |
| Result signalling, handling of FAIL / REVIEW boards, integration | PRD-15, PRD-16, PRD-18 |
| Maximum acceptable review rate | PRD-17 |
| Storage location and retention period | PRD-20 |
| Traceability requirements, escape / false-call targets, error costs | PRD-21, PRD-22, PRD-23 |
| Production compute constraints; hardware budget; sign-off; fallback | PRD-24, PRD-26, PRD-28, PRD-29 |
| **Formal company ESD policy** (UNKNOWN unless documented) | PRD-27 |

### Imaging hardware
| Topic | IDs |
|---|---|
| **Production camera** (to be selected later) | IMG-01 |
| Phone make / model (recorded per capture during Experiment 0) | IMG-01 |
| Lens: not decided yet | IMG-02 |
| Lighting; enclosure / controlled lighting: not decided yet | IMG-03 |
| Camera working distance / mount: not decided yet | IMG-04 |
| Station location, capture PC, calibration targets | IMG-05, IMG-06, IMG-07 |
| Prototype compute details: GPU memory, CPU, RAM, OS | PRD-24 |

### Data access
| Topic | IDs |
|---|---|
| How many of the ~500 good boards may be photographed; production lots; how "known-good" was established | DATA-01 |
| Number of rejected boards, their defect categories and records | DATA-02, DATA-04 |
| Electrical test results per board | DATA-03 |
| Permission for controlled defects (**not assumed**) | DATA-05 |
| Identity and qualification of the operator(s) verifying ground truth | DATA-06 |

## 4. Direct consequences of the verified answers

These follow from the answers without further assumptions. Consequences that depend on an open answer are phrased conditionally.

1. **Colour is a visual variation, not a variant.** The data model records `colour` per board and per capture, under the single variant `1`. No separate model per colour is created in advance. Experiments decide whether one model handles all colours, or whether colour-specific references or settings are needed. Data splits will be stratified by colour so that each colour is represented.
2. **Two inspection stages.** Every capture records its `inspection_stage` (`after_smt_reflow` or `after_tht`). *If* through-hole parts are fitted after SMT/reflow, the same board looks different at the two stages, so references and expected-component maps may need to exist per stage. The data are kept separable by stage for that reason.
3. **The phone is a prototype.** Experiment 0 results show what a simple camera can and cannot see. They do not define production imaging, and phone images are kept apart from the production-representative dataset.
4. **Smallest feature still open.** "Small resistors" sets the visual target for Experiment 0: check whether the smallest resistors are visible and countable in whole-board phone photos. A resolution requirement still needs their real dimensions (PCB-29).
5. **Holes: orientation help, but not yet an alignment reference.** *If* the hole pattern is confirmed (one position without a hole), it could help define the board's orientation. Its use for alignment depends on PCB-17.
6. **Top side only; no board identification.** These are unchanged from round 1. Board identity during data collection is maintained by us (labelled bags / trays with IDs such as `GOOD_001`, `DEFECT_001`), never by marking the board.
7. **ESD.** No special restrictions were reported. The protocol still uses careful, ESD-conscious handling: it protects the boards, and fingerprints or dust would appear in the images.
