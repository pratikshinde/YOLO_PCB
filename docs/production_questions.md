# Questions for the Production / PCB Team

| | |
|---|---|
| Status | Two rounds answered, 2026-10-02. Many items still open (see summary). |
| Answers source | Production/PCB mentor, 2026-10-02 (round 1 and round 2) |
| How to use | Record each answer with its source (person, document, date). Then copy **verified** values into the config:<br>• PCB-specific values (PCB-xx, DATA-xx, PRD-06 inspection stages) go into the inspection profile [`configs/profiles/lg_indicator/variant_1.yaml`](../configs/profiles/lg_indicator/variant_1.yaml).<br>• Line-level values (PRD-xx) and station values (IMG-xx) go into [`configs/project.yaml`](../configs/project.yaml).<br>Ambiguous answers are recorded verbatim and stay `null` until clarified. |
| Related | [production_profile.md](production_profile.md) (KNOWN / UNKNOWN / NEEDS CLARIFICATION) · [pcb_inspection_spec.md](pcb_inspection_spec.md) |

**Priority:**
- **P1**: blocks the first data collection.
- **P2**: needed before experiments and model choice.
- **P3**: needed before integrating with the production line.

**Status values:**
- **ANSWERED**: the question is answered.
- **PARTIAL**: part of the question is answered; the rest is UNKNOWN.
- **PARTIALLY CLARIFIED**: an ambiguous answer has been partly clarified; the rest is still open.
- **NEEDS CLARIFICATION**: the answer is recorded verbatim; its meaning must be confirmed.
- **UNKNOWN**: no answer yet. "Not decided yet" answers are also UNKNOWN.
- **N/A**: not applicable to the current scope.

**Summary after round 2** (74 questions, including PCB-29 and IMG-01 to IMG-07): 12 ANSWERED · 16 PARTIAL · 1 PARTIALLY CLARIFIED · 4 NEEDS CLARIFICATION · 1 N/A · 40 UNKNOWN.

---

## NEXT QUESTIONS: status after round 2

**The Experiment 0 mobile-phone capture is no longer blocked.** A good board and a defective board are available, the prototype camera is decided (mobile phone), and the protocol is ready ([experiment_0_camera_visibility.md](experiment_0_camera_visibility.md)). The open items below block the **production-representative** dataset and the production imaging design, not the phone experiment.

| # | Topic | Status after round 2 | Still to ask | IDs |
|---|---|---|---|---|
| 1 | Holes / fiducials | Partly answered: lower-left position has no hole; the remaining 3 positions have holes | What kind of feature the holes are (fiducial, tooling, mounting, other)? Exact coordinates and diameter? What are the four "positions", and from which viewing orientation is "lower-left" meant? A photo of the top side with the holes marked answers all three. | PCB-17 |
| 2 | Inspection stage | Partially clarified: after SMT/reflow and after THT | Is there a third stage ("all 3")? If yes, which? | PRD-06 |
| 3 | PCB colour | Clarified: same PCB, colour may vary with availability, function the same | Is blue / red / green the complete list? Roughly how many boards of each colour are available? (Not blocking.) | PCB-18 |
| 4 | Camera | Prototype: mobile phone (Experiment 0 only). Production camera and lens: not decided | Decided after Experiment 0 | IMG-01, IMG-02 |
| 5 | Imaging location | Not answered | Where will the pilot station stand? Space and height above the board? | IMG-05, PRD-08 |
| 6 | Lighting | Lighting and enclosure: not decided. Ambient conditions: not answered | Describe the ambient lighting where the photos will be taken | PRD-11, IMG-03 |
| 7 | Smallest feature | Mentor: "resistors like too small". Exact package / dimensions UNKNOWN; 1206 is not confirmed as the smallest | Which resistors? Their package / dimensions, from the BOM or PCB documentation | PCB-29, PCB-10 |
| 8 | Real defects | Not answered | Which defect categories actually occur, and how often? Does manual inspection keep a log? | PCB-24, PCB-25, PRD-07 |
| 9 | Rejected-board labels | A defective board is available; whether rejects carry defect records: not answered | Does each rejected board come with a defect record (type, location)? How many rejected boards are there? | DATA-02, DATA-04 |
| 10 | CAD files | Not answered | Are the BOM, the pick-and-place file and Gerber data available? | PCB-11, PCB-12, PCB-22 |
| 11 | ESD | Mentor: "No rules." Recorded as: no special restrictions reported by the mentor | Is a formal company ESD policy documented anywhere? (Not blocking the phone experiment; standard careful handling is used anyway.) | PRD-27 |
| 12 | "Known-good" basis | Not answered | How was "known-good" established for the ~500 boards? Lots? | DATA-01, DATA-06 |

**To ask at the bench during Experiment 0** (the operator can answer these directly):
- Which inspection stage each available board is in: after SMT/reflow, or after THT.
- What exactly is wrong with each defective board, and where.
- Which colour each board is.

---

## A. The PCB itself (Phase 1)

| ID | P | Question | Why we need it | Answer | Status | Source |
|---|---|---|---|---|---|---|
| PCB-01 | P1 | What is the PCB model / part number to be inspected? | Identifies the product; all data are organised per model. | LG Indicator | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PCB-02 | P1 | How many variants exist, and how do they differ? | Each visually different variant needs its own reference data. | 1 | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PCB-03 | P1 | Which board revisions are in production, and how is the revision marked on the board? | Revisions can look different. | "single". How the revision is marked on the board: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PCB-04 | P1 | Board dimensions (length × width, mm)? Is an outline drawing available? | Determines field of view and camera resolution. | 83 × 116 mm. Outline drawing: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PCB-05 | P1 | At the planned inspection point, is the board single or still in a panel? | Changes field of view, fixture and board-identity handling. | Individually | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PCB-06 | P1 | Board thickness (mm)? Maximum component height on each side (mm)? | Fixture design, depth of field, parallax from tall parts. | 1.6 mm / approximately 12 mm. The side of the 12 mm part was not stated. | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PCB-07 | P1 | Which sides carry components? Which sides must be inspected? | One or two capture views. | Top side (inspection). Which sides carry components: not stated. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PCB-08 | P1 | Assembly technology and process sequence: SMT, THT or mixed; reflow, wave, selective or hand soldering? Parts added after the inspection point? | Defines what the board looks like at the moment of inspection. | Mixed assembly. Which technologies are mixed, the soldering processes, and parts added after the inspection point: UNKNOWN. | NEEDS CLARIFICATION | Production/PCB mentor, 2026-10-02 |
| PCB-09 | P2 | How many components are on each side? | Annotation effort, inspection time. | UNKNOWN | UNKNOWN | — |
| PCB-10 | P1 | Which component types and packages are used? Smallest package? Fine-pitch ICs? BGA / QFN / LGA? Connectors? THT parts? | The smallest package sets the required resolution; hidden-joint packages cannot be inspected with 2D. | Round 1: smallest package reported as 1206. Round 2: 1206 is **not** to be treated as the confirmed smallest physical feature (see PCB-29). Other component / package types: UNKNOWN. Size-code system (imperial / metric) not stated. | NEEDS CLARIFICATION | Production/PCB mentor, 2026-10-02 |
| PCB-11 | P2 | Can we get the BOM with reference designators, part numbers and packages? | Component identity. | UNKNOWN (not assumed available) | UNKNOWN | — |
| PCB-12 | P2 | Can we get the centroid / pick-and-place file (refdes, X, Y, rotation, side)? | Expected position and orientation of every component. | UNKNOWN (not assumed available) | UNKNOWN | — |
| PCB-13 | P2 | Which components are polarized, and what polarity marking do they carry? | Polarity is only checkable where a visible mark exists. | UNKNOWN | UNKNOWN | — |
| PCB-14 | P2 | Which components are critical, and why? | Decision rules treat critical components differently. | UNKNOWN | UNKNOWN | — |
| PCB-15 | P2 | Are there look-alike parts (same package, different value, no marking)? Where? | Swaps of unmarked look-alikes are not visually detectable. | UNKNOWN | UNKNOWN | — |
| PCB-16 | P2 | Are there approved alternate parts? How do they differ visually? | A correct alternate looks different from the reference. | UNKNOWN | UNKNOWN | — |
| PCB-17 | P1 | Does the board have fiducial marks? Where, what type? Tooling holes? | Basis for image alignment and fixture design. | Round 1 (verbatim): "yes and expect down left every position has the holes".<br>Round 2 (verbatim): "There are no holes in the lower-left side. The remaining 3 positions have holes."<br>**Known:** lower-left position has no hole; the three remaining positions have holes.<br>**Unknown:** whether these are fiducials, tooling holes, mounting holes or another feature; exact coordinates and geometry; what the four "positions" are and from which orientation "lower-left" is meant. | NEEDS CLARIFICATION | Production/PCB mentor, 2026-10-02 |
| PCB-18 | P1 | Solder mask colour, silkscreen colour, surface finish? Variation between lots / suppliers? | Lighting choice and normal appearance variation. | Round 1: "PCB colours reported: blue, red, green".<br>Round 2 (verbatim): "It is the same PCB. Depending on availability, the colour may change, but the working/function of the PCB remains the same."<br>**Recorded:** same PCB model and function; PCB colour can vary; current colours may include blue / red / green; the functional design is the same. Colour is treated as a visual variation, not a separate variant.<br>Whether the colour list is complete: UNKNOWN. Surface finish: UNKNOWN. Silkscreen colour: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PCB-19 | P1 | Is conformal coating applied before the inspection point? | Coating changes appearance and glare. | No | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PCB-20 | P2 | Which outer-layer traces matter? Minimum trace width and spacing? Covered by solder mask? | Trace visibility. | UNKNOWN | UNKNOWN | — |
| PCB-21 | P1 | Are there labels, barcodes, DataMatrix codes or serial-number markings on the board? | Board identity. | "No barcode/DataMatrix/serial-number identification in the current scope; may be considered in future." | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PCB-22 | P2 | Is Gerber / ODB++ data available? Assembly drawing? | CAD-based references. | UNKNOWN | UNKNOWN | — |
| PCB-23 | P2 | What acceptance standard applies? Allowed placement offset and rotation? | Defect vs acceptable variation. | UNKNOWN | UNKNOWN | — |
| PCB-24 | P1 | Which defect types occur today, and how often? | Which defects to cover first. | Mentor answered "all". Recorded as: "All previously listed target defect categories are considered relevant to the project at this stage." Actual defect frequency / Pareto: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PCB-25 | P2 | Overall defect rate (% of boards or ppm)? | How many boards give enough defect samples. | UNKNOWN | UNKNOWN | — |
| PCB-26 | P2 | Which defects currently pass electrical/functional test? | Where visual inspection adds value. | UNKNOWN | UNKNOWN | — |
| PCB-27 | P2 | Is board warpage typical? How much? | Alignment accuracy and focus. | UNKNOWN | UNKNOWN | — |
| PCB-28 | P2 | What visual variation is known to be acceptable? | Prevents normal variation being called a defect. | UNKNOWN | UNKNOWN | — |
| PCB-29 | P1 | Beyond the smallest package, what is the smallest physical feature that must be detected (polarity mark, lead / pin pitch, solder joint, marking text, trace width / gap)? Dimensions in mm. | Package size alone does not set the camera resolution; the smallest feature to be seen does. | Verbatim: "resistors like too small".<br>**Recorded:** small resistors are among the smallest visual objects to be considered. Their exact package and physical dimensions: UNKNOWN (to be confirmed from the BOM or PCB documentation, not assumed). | NEEDS CLARIFICATION | Production/PCB mentor, 2026-10-02 |

## B. Production line and process (Phase 2)

| ID | P | Question | Why we need it | Answer | Status | Source |
|---|---|---|---|---|---|---|
| PRD-01 | P2 | How many boards are inspected per day / per shift? | Throughput and data volume. | UNKNOWN | UNKNOWN | — |
| PRD-02 | P2 | How many PCB variants run on this line? Changeover frequency? How is the active variant identified? | Variant handling. | "Current scope: 1 PCB variant". Changeover and variant identification: UNKNOWN. The architecture stays configurable for future models / variants. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PRD-03 | P2 | How long does electrical/functional testing take per board? What does it cover? | Time budget; what test already catches. | UNKNOWN | UNKNOWN | — |
| PRD-04 | P2 | How much time is available for visual inspection per board? | Latency requirement. | UNKNOWN | UNKNOWN | — |
| PRD-05 | P2 | Required throughput / takt time? | Latency and hardware sizing. | UNKNOWN | UNKNOWN | — |
| PRD-06 | P1 | Where exactly would the station sit: after reflow, after THT/wave, directly before electrical test? | Defines the board state the data must represent. | Round 1: "Mentor answered 'all 3'; exact inspection point requires clarification."<br>Round 2: **after SMT/reflow** and **after THT**.<br>A third stage is not confirmed and stays UNKNOWN. | PARTIALLY CLARIFIED | Production/PCB mentor, 2026-10-02 |
| PRD-07 | P1 | What inspection exists today: SPI, AOI, AXI, manual visual? What does each cover? Why a new system? | Defines this system's role. | Manual inspection. What it covers, and why a new system is wanted: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PRD-08 | P2 | What space is available for the camera station? Can it be enclosed? Vibration? | Camera distance, lens, enclosure. | UNKNOWN | UNKNOWN | — |
| PRD-09 | P1 | How is the PCB positioned: fixture, conveyor, carrier? Is the orientation guaranteed? | Position repeatability drives alignment requirements. | Manual. Whether orientation is guaranteed: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PRD-10 | P2 | Conveyor type, width and speed? Does the board stop for capture? | Moving boards need global shutter / strobe. | UNKNOWN / not specified | UNKNOWN | — |
| PRD-11 | P2 | Ambient lighting at the station? Can it be shielded? | Uncontrolled light causes false calls. | UNKNOWN | UNKNOWN | — |
| PRD-12 | P3 | How should capture be triggered? | Integration design. | UNKNOWN | UNKNOWN | — |
| PRD-13 | P2 | Is loading manual or automatic? | Position variation and fixture design. | Manual | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PRD-14 | P2 | If both sides need inspection: operator flip, flip station, or two cameras? | Station layout. | Not applicable to the current requirement, because the current inspection side is top only, unless later changed. | N/A | Production/PCB mentor, 2026-10-02 |
| PRD-15 | P3 | How should PASS / FAIL / REVIEW be signalled? | Integration design. | UNKNOWN | UNKNOWN | — |
| PRD-16 | P3 | What happens to FAIL and REVIEW boards? | REVIEW workflow. | UNKNOWN | UNKNOWN | — |
| PRD-17 | P2 | Is an operator available to review boards? Maximum acceptable review rate? | How strictly uncertain results may go to REVIEW. | Operator is available. Maximum acceptable review rate: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PRD-18 | P3 | Integration requirements: MES / ERP / PLC? Protocols? | Interface design. | UNKNOWN | UNKNOWN | — |
| PRD-19 | P2 | How is each board identified at the station? | Links results to a board. | Currently no barcode / DataMatrix / serial-number identification planned. Future consideration. | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PRD-20 | P3 | Which results must be stored? Where, and for how long? | Storage and retention. | Yes, storage is required. Exact location and retention period: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PRD-21 | P2 | Traceability or regulatory requirements? | Record keeping. | UNKNOWN | UNKNOWN | — |
| PRD-22 | P2 | Maximum acceptable escape rate and false-call rate? | Acceptance criteria. | UNKNOWN | UNKNOWN | — |
| PRD-23 | P2 | Relative cost of an escape, a false call, a review? | Tuning the decision rules. | UNKNOWN | UNKNOWN | — |
| PRD-24 | P3 | Compute constraints: industrial PC, GPU, network, OS? | Deployment hardware. | Prototype compute: NVIDIA RTX 5050 GPU system. GPU memory and other machine specifications: not provided. Production compute constraints: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PRD-25 | P1 | IT / security policy: may images leave the site? Cloud allowed? | Where data may be stored. | No restrictions reported. Images / results can be stored locally. Cloud requirements are not currently defined. | ANSWERED | Production/PCB mentor, 2026-10-02 |
| PRD-26 | P2 | Budget for camera, lens, lighting, compute? Preferred suppliers? | Hardware selection. | UNKNOWN | UNKNOWN | — |
| PRD-27 | P1 | What ESD and safety rules apply? | Must be followed from the first capture. | Round 1: no special restrictions reported by mentor.<br>Round 2 (verbatim): "No rules."<br>**Recorded:** no special restrictions were reported by the mentor. This does **not** state that the company has no ESD policy. A formal company ESD policy remains UNKNOWN unless it is documented. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| PRD-28 | P3 | Who signs off acceptance? What validation is required? | Final evaluation protocol. | UNKNOWN | UNKNOWN | — |
| PRD-29 | P3 | Fallback when the station is down? | Availability. | UNKNOWN | UNKNOWN | — |

## C. Data access and permissions (Phases 3 to 5)

| ID | P | Question | Why we need it | Answer | Status | Source |
|---|---|---|---|---|---|---|
| DATA-01 | P1 | How many known-good boards can be lent for imaging? For how long? From how many lots? | The first dataset consists of known-good boards. | "Approximately 500 known-good boards available." Round 2: a real good board is available for the first capture experiment. Not confirmed: whether all can be photographed, whether all belong to the same lot, loan period, how "known-good" was established. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| DATA-02 | P1 | Can we access rejected or failed boards? How many, which defects? Already reworked? | Preferred source of defect data. | Yes, available. Round 2: a defective board is available for the first capture experiment. Exact number, defect categories and traceability: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| DATA-03 | P2 | Are electrical test results available per serial number? | Ground-truth support. | UNKNOWN | UNKNOWN | — |
| DATA-04 | P2 | Are repair / rework records available per board? | Ground truth for defect labels. | UNKNOWN | UNKNOWN | — |
| DATA-05 | P2 | Is creating controlled defect samples permitted? On which boards? Who approves? | Requires explicit authorization. | UNKNOWN. Permission is **not** assumed. | UNKNOWN | — |
| DATA-06 | P1 | Who is qualified to verify ground truth? | Every label traces to a named person. | "Operator will provide/verify ground truth." If several operators exist, their identity and qualification process: UNKNOWN. | PARTIAL | Production/PCB mentor, 2026-10-02 |
| DATA-07 | P2 | Can a pilot camera station be placed on or next to the line? | Production-representative data. | Yes | ANSWERED | Production/PCB mentor, 2026-10-02 |
| DATA-08 | P1 | Confidentiality: NDA, storage location, access? | Applies to the very first image. | No confidentiality restrictions reported. Images can be stored locally. | ANSWERED | Production/PCB mentor, 2026-10-02 |
| DATA-09 | P1 | Handling rules for borrowed boards? | Protects boards; marks change appearance. | No restrictions reported by mentor. This does not override any formal company ESD / safety procedure (see PRD-27). | ANSWERED | Production/PCB mentor, 2026-10-02 |

## F. Imaging hardware and pilot station (added 2026-10-02)

These may be answered by the project team rather than production. Values go into `station` in `configs/project.yaml`.

| ID | P | Question | Why we need it | Answer | Status | Source |
|---|---|---|---|---|---|---|
| IMG-01 | P1 | Is a camera available for the pilot? Make/model, sensor resolution, pixel size, colour/mono, shutter type, interface, SDK? | Every imaging calculation depends on it. | **Prototype camera** (Experiment 0 only): mobile phone. Phone make / model: not stated (recorded per capture). **Production camera:** UNKNOWN (to be selected later). | PARTIAL | Production/PCB mentor, 2026-10-02 |
| IMG-02 | P1 | Which lens(es) are available: focal length, mount, aperture range, minimum working distance? | Field of view, working distance, depth of field. | Not decided yet. | UNKNOWN | Production/PCB mentor, 2026-10-02 |
| IMG-03 | P1 | Which lighting hardware is available (ring light, dome, bar lights, LED panel)? Can an enclosure be built? | Lighting candidates. | Lighting: not decided yet. Enclosure / controlled lighting: not decided yet. | UNKNOWN | Production/PCB mentor, 2026-10-02 |
| IMG-04 | P1 | Camera mounting: stand / arm, and the height available above the board? | Working distance, rigidity, repeatability. | Camera working distance / mount: not decided yet. | UNKNOWN | Production/PCB mentor, 2026-10-02 |
| IMG-05 | P1 | Exact location of the pilot station (bench / area near the line); power available? | Ambient light, space, workflow. | UNKNOWN | UNKNOWN | — |
| IMG-06 | P2 | Capture computer: OS, USB3 / GigE ports, storage space? | Camera SDK, image storage. | UNKNOWN (the RTX 5050 system is recorded as prototype compute, PRD-24) | UNKNOWN | — |
| IMG-07 | P2 | Calibration targets: available, or to be procured (geometric grid sized to ≥ 83 × 116 mm, grey card)? | Required in every capture session. | UNKNOWN. For the phone experiment, a ruler is used as the scale reference. | UNKNOWN | — |

## D. Documents and files to request

| Item | Related questions | Status |
|---|---|---|
| Bill of materials (BOM) with reference designators | PCB-11 | Availability UNKNOWN |
| Centroid / pick-and-place file | PCB-12 | Availability UNKNOWN |
| Gerber or ODB++ (outer copper, solder mask, silkscreen, outline) | PCB-22 | Availability UNKNOWN |
| Assembly drawing (top) | PCB-22 | Availability UNKNOWN |
| Photo or drawing marking fiducials and tooling holes | PCB-17 | Requested (blocking item 1) |
| Acceptance criteria document (IPC class or internal spec) | PCB-23 | Availability UNKNOWN |
| Defect records / manual inspection log | PCB-24, PRD-07, DATA-04 | Availability UNKNOWN |
| Electrical / functional test specification and coverage | PRD-03 | Availability UNKNOWN |
| Company ESD / handling procedure, if one exists | PRD-27 | Requested (blocking item 11) |

## E. Physical samples

| Item | Related questions | Status |
|---|---|---|
| Known-good assembled LG Indicator boards | DATA-01, DATA-06 | ~500 reported available. A real good board is available for Experiment 0. How many may be photographed: UNKNOWN. |
| Rejected / failed boards with their defect records | DATA-02, DATA-04 | Reported available. A defective board is available for Experiment 0. Count and defect records: UNKNOWN. |
| Scrap or engineering boards for controlled defects | DATA-05 | **Not approved** (permission UNKNOWN) |
| One bare (unassembled) LG Indicator PCB, if available | PCB-20, PCB-22 | Not yet asked |
