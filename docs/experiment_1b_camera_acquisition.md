# Experiment 1B: Camera Acquisition and Review System

> **Superseded on 2026-10-06** by the clean GOOD / FAULT collection: `python scripts/collect.py`, [clean_data_collection.md](clean_data_collection.md).
> - It is the same application (camera, countdown, session store, review), plus a PCB crop and a capture-time check, in `data/collection/`.
> - The Experiment 1B images are archived. `scripts/acquire.py` runs only with `--legacy`.

| | |
|---|---|
| Status | **Implemented and unit-tested without a camera (mock camera).** Not yet run with the real Redmi 12 5G + DroidCam stream: no images exist. |
| Date | 2026-10-05 |
| Start | `python scripts/acquire.py` |
| Config | [`configs/capture/droidcam_e1b.yaml`](../configs/capture/droidcam_e1b.yaml) |
| Code | `src/pcb_inspection/acquisition/` (`camera.py`, `session.py`, `controller.py`, `ui.py`) |
| Related | [Experiment 1B protocol](../experiments/experiment_001b/README.md) · [data collection plan](data_collection_plan.md) · [Experiment 0](experiment_0_camera_visibility.md) |

## 1. Purpose

Collect many PCB images **directly from a live camera** into the project. The phone (Xiaomi Redmi 12 5G) is used as a webcam through DroidCam. Nothing is photographed on the phone and transferred by hand.

```
PCB → live DroidCam camera → guided capture (3-2-1 countdown) → original saved automatically
    → metadata + SHA-256 automatically → quality check automatically
    → human review: KEEP → data/curated/<type>/   REJECT (+ reason) → data/quarantine/<type>/
```

The operator types **nothing per image**. Per session they choose the dataset type, an optional board ID, the stage, the number of images and the interval.

**Scope.** This is camera → acquisition → quality check → human review → curated data, nothing more:
- no model training of any kind;
- no synthetic defects;
- no PASS / FAIL logic;
- no production hardware, PLC or deployment.

The phone is a **prototype camera**. These images are kept apart from the production-representative track (`data/raw/`) and from the Experiment 0 phone photos (`data/experiment_0/`).

## 2. Architecture

| Module | Responsibility | Tested with |
|---|---|---|
| `acquisition/camera.py` | Lists video devices **by name** (DirectShow), opens the chosen one (OpenCV, DirectShow backend), reads frames on a background thread, reports `live` / `connecting` / `disconnected` / `frozen`, re-opens after a disconnect | mock camera, fake clock |
| `acquisition/controller.py` | `CaptureController`: countdown → capture → (wait for review) → next; STOP; pause on camera loss. `capture_once()`: takes one real capture from a fresh frame and records failures | fake clock, mock camera |
| `acquisition/session.py` | IDs, folders, `session.json`, saving + checksum verification, quality check, KEEP / REJECT copies, failure records, interrupted sessions, `audit()`, board split groups | temporary folders, generated test images |
| `acquisition/ui.py` | Tkinter window: shows state, forwards operator actions; no logic of its own | smoke run with a mock camera |
| `scripts/acquire.py` | Starts the application with the project config, profile and capture config | |
| `scripts/acquisition_summary.py` | Per-session / per-board summary and consistency audit (read-only) | |

**Reused from the existing project (no second implementation):**
- `image_quality.checks.check_image` / `QualityParams`: the Experiment 0 checks, with thresholds in the capture config.
- `image_quality.report.quality_row`: the quality summary fields.
- `flag_relative_sharpness`: applied within the session.
- `dataset.experiment0.sha256_of`: checksums.
- `dataset.naming.SESSION_ID_PATTERN` / `SIDES`: the project session-ID format and sides.
- `config.load_config` / `load_profile`: board size, stages, PCB model, variant, side, defect taxonomy.
- `dataset.splitting`: the board-level split functions work directly on the acquisition records (section 12).

## 3. Camera setup (DroidCam)

1. Install the **DroidCam** app on the phone and the **DroidCam Windows client** on the PC. Both are already present on this PC: Windows lists a device named `DroidCam Video`.
2. Connect the phone (USB recommended for a stable stream; Wi-Fi also works).
3. **Start the DroidCam client on the PC and press Start** so that it streams. The application can only open `DroidCam Video` while the client is streaming. When it is not streaming, opening fails; this was observed on this PC on 2026-10-05.
4. **Resolution:** set the highest available resolution in the DroidCam client (target 1920 × 1080, 30 FPS).
   - The DroidCam edition installed may limit the resolution; this is not verified here.
   - The application always records the resolution the camera **actually delivers**.
   - It shows a warning if this differs from the requested resolution.
5. Physical setup: as in the [Experiment 1B protocol](../experiments/experiment_001b/README.md), section 1.
   - Phone fixed on a stand above the board, as perpendicular as practical.
   - Plain matte light background with a corner guide.
   - Diffuse light, no flash.
   - Lock focus / exposure in the DroidCam app if it offers this.

List the cameras and test one without starting the window:

```bash
python scripts/check_camera.py --list
```

```bash
python scripts/check_camera.py
```

The second command opens the configured camera with `cv2.VideoCapture(index, cv2.CAP_DSHOW)` and checks:
- `cap.isOpened()`;
- 30 `cap.read()` calls, printing the frame size;
- that the frames show a real picture;
- that the camera is released (it must report closed and open again).

Use `--index 1` for another camera.

**On this PC (2026-10-05):**

| Index | Device | Result |
|---|---|---|
| `0` | `Integrated Camera` (Lenovo laptop webcam, SunplusIT) | Opens and reads 1920 × 1080 at ~21–24 FPS. **The picture was blank (uniform)**: the camera was covered or switched off (privacy shutter / camera key) |
| `1` | `DroidCam Video` (Redmi 12 5G over USB) | **Configured.** With the DroidCam client streaming: 2560 × 1440 at ~33 FPS, real picture. A real 2-image GOOD session passed end-to-end. Fails to open while the client is not streaming |

## 4. Starting and selecting the camera

The camera is set **once, in the capture config**:

```yaml
camera:
  index: 1                 # OpenCV index (DirectShow order); 1 = DroidCam Video on this PC
  backend: dshow           # cv2.VideoCapture(index, cv2.CAP_DSHOW)
  connect_on_start: true
  requested_width: 1920
  requested_height: 1080
```

```bash
python scripts/acquire.py
```

To use another camera for one run (e.g. the laptop camera):

```bash
python scripts/acquire.py --camera-index 0
```

1. **Opening.** The configured camera is pre-selected and opened at start. Another camera is never substituted. If the configured index does not exist, the application says so and lists the devices.
   - The device list (top of the centre column) can still be used to switch camera; press **Connect**.
   - The camera is opened once. Connecting again first stops the reader thread, which releases the camera.
2. **Status line:**
   - `LIVE`, the camera name;
   - the delivered resolution;
   - the FPS reported by the driver and the FPS measured from the stream.
3. **Problems get a clear status and message, and no capture is possible meanwhile:**
   - `DISCONNECTED`: cannot open the camera / `cap.read()` fails. The message names the index, the backend and what to check.
   - `FROZEN`: the picture stopped changing.
   - `BLANK PICTURE`: the camera sends uniform frames because it is covered or switched off.
4. **Refresh** re-lists the devices. **Reconnect** re-opens the device.

**Session metadata.** The phone model (`device_model`) is recorded only when the camera is the DroidCam stream. With the integrated camera it is `null`, and `camera.name` says `Integrated Camera`.

## 5. The three collection modes

The dataset type is chosen once per session with the large **GOOD PCB / DEFECT PCB / TEST PCB** buttons. It never changes within a session, and the three types are stored in separate folders.

| | GOOD PCB | DEFECT PCB | TEST PCB |
|---|---|---|---|
| Board | Verified good | Verified defective | Unseen, **unlabelled** |
| Board ID | `GOOD_###` (optional, recommended) | `DEFECT_###` (optional, recommended) | `TEST_###` (optional, recommended) |
| `declared_condition` | `good` (operator, at capture) | `defective` (operator, at capture) | `null`, never GOOD / DEFECT |
| Extra input | none | defect information, optional (section 6) | none |
| KEEP goes to | `curated/good/` | `curated/defect/` | `curated/test/` |

**Board ID input.** Typing `1` in a GOOD session gives `GOOD_001`; `good_7` gives `GOOD_007`. A board ID of another pool is refused:
- `GOOD_001` cannot be used in a DEFECT or TEST session;
- `DEFECT_001` cannot be used in a GOOD or TEST session.

So a board is never labelled ambiguously. Without a board ID, the application asks for confirmation. The images can then only be grouped by session, not by physical board.

**Stage** is chosen once per session:
- `AFTER_SMT` (`after_smt_reflow`) or `AFTER_THT` (`after_tht`): the confirmed stages from the profile;
- `OTHER`: needs a note describing the stage, because a third stage is not confirmed (PRD-06).

**Guidance text** for each mode is shown on screen (`guidance` in the capture config):
- GOOD: STEP 1 to 4.
- DEFECT: place a verified defective PCB; defect information is annotation, not verified ground truth.
- TEST: do not label the board.

**Colour** is never entered. Each image gets `colour_estimate` from the existing board-colour heuristic, with `colour_source: automatic_estimate`. It is an estimate for later review, not a label.

## 6. DEFECT PCB: defect information

**Defect known? YES / UNKNOWN.** UNKNOWN is the default, and no classification is forced.

With YES, a defect type is required:

| Option shown | Stored `defect_type` | Project taxonomy (`taxonomy_defect_type`) |
|---|---|---|
| Missing component | `missing_component` | `missing_component` |
| Damaged component | `damaged_component` | `damaged_component` |
| Broken component | `broken_component` | **`null`**: not in the taxonomy. Not mapped by guess; NEEDS CLARIFICATION |
| Wrong component | `wrong_component` | `wrong_component` |
| Broken trace | `broken_trace` | `broken_trace` |
| Missing trace | `missing_trace` | `missing_trace` |
| Solder defect | `solder_defect` | `solder_defect` |
| Component misalignment | `component_misalignment` | `misaligned_component` |
| Other | `other` | `other_anomaly` |

Optional fields:
- component / reference (e.g. `C14`);
- location;
- notes.

All of it is stored once per session in `session.json → defect_report`, with `source: operator_at_capture` and `verified: false`.

It is **annotation information, not ground truth**. It does not skip the quality check or the review. The Experiment 1B analysis still takes verified defects only from `data/experiment_0/metadata/defects.csv` (section 13).

## 7. Batch capture and the countdown

Session settings:
- **Images**: 1 to 500; the default 10 is what the first experiment uses.
- **Interval**: default 3 s, minimum 1 s.

Press **START SESSION**. For every capture:

```
Preparing capture... / NEXT CAPTURE IN   3   2   1   CAPTURE   →   CAPTURED
```

- **The capture is real and timed.** When the countdown ends, the application waits for a frame that **arrived after** that moment. A frame already on screen is never saved.
- **A frozen stream is refused.** A picture identical to the previous capture is refused, and no pixel change for 3 s counts as a frozen stream.
- **Saving.** The frame is saved **losslessly as PNG**, exactly as delivered: no resizing, cropping or overlay. The dashed inspection-area frame exists only in the preview.
- **What the operator sees:** `Captured: n / N`, `Failed: n`, the last image name, and the quality result with its flags.

**Two review modes** (checkbox "Review each image before the next capture"):
- **after each** (default): the countdown waits until the image is KEEP-ed or REJECT-ed, then starts again automatically.
- **after session**: capture continues without waiting. The images queue up on the right and can be reviewed at any time, also during capture or after the session.

A session ends after **requested** capture attempts (successful + failed), or when **STOP** is pressed.

## 8. Quality check

After every capture, automatically:
1. The image is saved with exclusive create, so an existing file is never overwritten.
2. The file is verified: it exists, its size matches, and its SHA-256 matches the captured bytes.
3. The existing Experiment 0 checks run, and the image is read back. The checks cover:
   - readable, resolution, sharpness;
   - exposure (glare / under-exposure on the board);
   - board detected, board at the image edge (possible crop), board shape;
   - colour estimate, approximate px/mm;
   - relative sharpness within the session.
4. The result is recorded and shown: `quality.status` is `pass` or `flagged`, plus the flag list and measurements.

**A flag never deletes or rejects an image.** It is evidence for the human review. Thresholds are the PROVISIONAL Experiment 0 values, in the capture config.

## 9. Review: KEEP / REJECT and quarantine

The **CAPTURE REVIEW** panel shows:
- the image preview;
- image ID, type (with the reported defect, if any), board, stage;
- quality, flags, colour estimate;
- the queue length.

| Action | Result |
|---|---|
| **KEEP** | Verified copy to `data/curated/<type>/<board>/<session>/`; message "Image accepted." |
| **REJECT** | A reason is required. Verified copy to `data/quarantine/<type>/<board>/<session>/`; message "Image quarantined." |

**Reject reasons** (`review.reject_reasons` in the config):
- blurry, PCB not fully visible, bad alignment;
- obstruction, reflection;
- wrong board, wrong stage;
- duplicate, operator mistake, other.

An optional note can be added.

**Rules:**
- **The original in `data/acquisition/` is never moved, modified or deleted.** KEEP and REJECT copy it. Before copying, the application checks that the original still matches its checksum.
- A decision is final. A second review of the same image is refused. Nothing is ever deleted.
- The queue shows the running session first, then images left pending from earlier sessions (e.g. after STOP or a crash).

## 10. Directory structure

```
data/
  acquisition/                                   originals, never modified
    good/GOOD_001/S20261005-01/
      session.json                               session + every image record
      images/GOOD_001_TOP_S20261005-01_C001.png
    defect/DEFECT_001/S20261005-02/...
    test/TEST_001/S20261005-03/...
    good/UNASSIGNED/S20261005-04/...             session without board ID
  curated/                                       KEEP (verified copies)
    good/GOOD_001/S20261005-01/GOOD_001_TOP_S20261005-01_C001.png
    defect/...   test/...
  quarantine/                                    REJECT (verified copies)
    good/...   defect/...   test/...
```

`acquisition → board → session` keeps the physical board visible in the folder tree.

**TEST isolation:**
- The destination folder is always derived from the session's own type. A TEST image can only reach `curated/test/` or `quarantine/test/`.
- `store.records()` (the dataset view) excludes TEST unless TEST is explicitly requested.
- `python scripts/acquisition_summary.py` audits all of this. It reports any copy outside its type's folder and any file in `curated/` or `quarantine/` that no record accounts for.

## 11. Identifiers and metadata

**IDs** follow the project conventions in `dataset/naming.py`:

| ID | Format | Example |
|---|---|---|
| `session_id` | `S<YYYYMMDD>-<NN>` | `S20261005-01` |
| `image_id` | `<board_id>_<side>_<session_id>_C<NNN>` | `GOOD_001_TOP_S20261005-01_C001` |

- **Session numbers** count up per day across all three types. They also skip numbers already used by the Experiment 0 ingest, so the two tracks never share a session ID.
- **Image IDs.** The ID shows the type (board prefix), board, side, session and capture sequence. Without a board ID the image ID is `GOOD_UNASSIGNED_TOP_...`.
- **Never overwritten.** If a file name already exists, the next free capture number is used.

**`session.json`** is the one metadata record. It is rewritten atomically after every event. There is no second CSV log.

| Level | Fields |
|---|---|
| Session (at start) | `session_id`, `dataset_type`, `declared_condition`, `condition_source`, `board_id`, `split_group`, `stage`, `side`, `pcb_model`, `variant`, `profile_id`, `capture_config_id`, `purpose`, `device_model`, `device_connection`, `start_time`, `camera` {name, index, backend, width, height, fps_reported, fps_measured}, `resolution`, `fps`, `capture_interval_s`, `requested_image_count`, `review_mode`, `image_format`, `defect_report`, `notes` |
| Session (at the end) | `status` (`running` → `completed` / `stopped`; `interrupted` if the application ended without closing the session), `end_time`, `counts` {`successful_captures`, `failed_captures`, `kept_count`, `rejected_count`, `pending_review_count`}, `failures` [{attempt, time, reason}] |
| Image | `image_id`, `session_id`, `dataset_type`, `board_id`, `split_group`, `stage`, `side`, `sequence`, `capture_id`, `captured_at` (frame arrival, ms), `camera`, `width`, `height`, `fps`, `frame_seq`, `image_format`, `original_file`, `saved_file` (curated / quarantine copy after review), `file_size_bytes`, `sha256`, `colour_estimate`, `colour_source`, `quality` {status, flags, sharpness, exposure, board box, …}, `review` {status, decided_at, reason, note}, `annotation` {status, defects} |

**Failed captures** never create an image record. They are listed in `failures` and counted in `failed_captures`. Examples:
- camera not live;
- no fresh frame within 2 s;
- frozen or identical frame;
- write or verification error;
- unreadable file.

**Stop / crash.**
- STOP saves the session as `stopped`. A capture in progress is finished first. Every captured image is kept, e.g. `requested 50, successful 27, status stopped`.
- If the application ends without closing a session, the next start marks it `interrupted`. Its images stay valid and reviewable.

**Camera disconnect.**
- The application does not crash. The camera status shows `DISCONNECTED`.
- The countdown pauses ("PAUSED", without using up an attempt), and the preview is darkened and marked "NO LIVE PICTURE".
- The camera is re-opened automatically every 2 s, or with **Reconnect**. The countdown restarts when the stream is live again.
- If the disconnect happens exactly at a capture moment, that attempt is recorded as failed, never as a captured image.

## 12. Physical board tracking and leakage prevention

Ten images of one board are **one** physical sample, not ten.

**Split groups.** Every session and image carries `split_group`:
- the board ID;
- or `UNASSIGNED:<session_id>` when no board ID was given.

**Splitting.** `split_groups(store.records())` maps `image_id → split_group` for the existing functions:
- `dataset.splitting.split_boards`;
- `assign_captures`;
- `find_leakage`.

All images of a board, across all its sessions, therefore land in the same split. This is tested. No split is created now.

**Unassigned sessions.** A session without a board ID cannot be linked to other sessions of the same board. Hence the "recommended" board ID.

## 13. Future annotation support (not built)

Each image has `annotation: {status: "not_started", defects: []}`. The session has the operator's `defect_report`, unverified.

Bounding boxes, masks or defect coordinates can later be added to an image's `annotation.defects`, linked by `image_id`. The `board_id`, `session_id` and `defect_report` relationships are already in place.

**Still open:**
- **No annotation tool exists.**
- **The Experiment 1B analysis does not read acquisition data yet.** `scripts/run_experiment_001b.py` still reads the Experiment 0 capture log and ground truth (`data/experiment_0/metadata/`). Connecting curated acquisition sessions to it is a separate, later step.

## 14. Experiment 1B: first collection plan

| # | What | How |
|---|---|---|
| 1 | Same GOOD board, ~10 images, controlled (board not moved) | GOOD PCB, board `GOOD_001`, stage, 10 images, 3 s, review after each |
| 2 | Same GOOD board, ~10 images, **lift and re-place** the board between captures | New GOOD session, same board ID. Use interval 5 to 10 s (or after-each review) to have time to re-place; write "re-placed between captures" in Notes |
| 3 | Additional verified GOOD boards | One session per board: `GOOD_002`, `GOOD_003`, … |
| 4 | Other board colours, if available | As 3. Colour is estimated automatically, not entered |
| 5 | At least one verified DEFECT board | DEFECT PCB, `DEFECT_001`, defect information if known |
| 6 | TEST boards (later) | TEST PCB, `TEST_001`, … No label |

## 15. First 10-image GOOD PCB session (step by step)

1. Fix the camera above the board; plain light background; diffuse light.
   - **Phone:** connect it, start the DroidCam client on the PC and press Start; the picture appears in the client.
   - **Laptop camera (index 0):** open the privacy shutter / switch the camera on.
2. Test the camera (close the application first). It must end with `OK`:

   ```bash
   python scripts/check_camera.py
   ```

   The configured camera is the phone (index 1); add `--index 0` for the laptop camera.
3. Start the application. Use `python scripts/acquire.py --camera-index 0` for the laptop camera.

   ```bash
   python scripts/acquire.py
   ```

4. The configured camera opens automatically.
   - Wait for `LIVE`.
   - Check the delivered resolution in the status line. If a warning appears, raise the resolution (DroidCam client) and press Reconnect.
5. Press **GOOD PCB**. Read the guidance.
6. Board ID: `GOOD_001` (or `1`). Stage: `AFTER_THT`. Images: `10`. Interval: `3`. Leave "Review each image before the next capture" ticked.
7. Place the verified good board so that it lies completely inside the dashed frame.
8. Press **START SESSION**. Watch 3 → 2 → 1 → CAPTURE → CAPTURED.
9. Review the image on the right:
   - **KEEP**, giving "Image accepted.";
   - or choose a reason and press **REJECT**, giving "Image quarantined.".

   The next countdown starts automatically.
10. Repeat until `Captured: 10 / 10`. The message line shows the session summary.
11. Check the result:

    ```bash
    python scripts/acquisition_summary.py
    ```

    Images: `data/curated/good/GOOD_001/<session_id>/`. Originals: `data/acquisition/good/GOOD_001/<session_id>/images/`.

## 16. Assumptions and items to confirm

- **DroidCam resolution / frame rate** is not verified here: the phone was not connected during development. 1920 × 1080 at 30 FPS is the target; the actual values are recorded per session.
- **The DroidCam stream is assumed to show every frame from the phone.** If the client applies its own processing (e.g. compression), that is part of the "original".
- **The phone model** "Xiaomi Redmi 12 5G" comes from the project team. It is stored as `device_model`; the camera name comes from Windows.
- **"Broken component"** has no equivalent in the project defect taxonomy. It is stored as given, with `taxonomy_defect_type: null`.
- **Quality thresholds** are the PROVISIONAL Experiment 0 values. Adjust them after the first real sessions.
- **Frozen-stream detection** assumes a live stream always changes slightly (sensor noise). It triggers after 3 s without any pixel change.
  - If a perfectly static, heavily compressed scene ever shows `FROZEN` while the picture is clearly live, raise `camera.frozen_after_s` or set it to `null` (off).
  - The check that refuses a capture identical to the previous one stays active either way.
