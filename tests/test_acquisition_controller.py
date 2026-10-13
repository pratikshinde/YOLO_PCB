"""Timed batch capture (countdown, review gating, stop, camera loss) and capture_once, without a camera."""

from pathlib import Path

import numpy as np
import pytest

from pcb_inspection.acquisition.camera import FrameGrabber
from pcb_inspection.acquisition.controller import (
    PHASE_CAMERA_LOST,
    PHASE_CAPTURING,
    PHASE_COUNTDOWN,
    PHASE_FINISHED,
    PHASE_WAITING_PCB,
    PHASE_WAITING_REVIEW,
    SEQ_BOARD,
    SEQ_PLACE,
    SEQ_REMOVE,
    BoardSequence,
    CaptureController,
    CaptureOutcome,
    capture_once,
)
from pcb_inspection.acquisition.session import AcquisitionStore, SessionConfig, settings_from_configs
from test_acquisition_camera import FakeCamera, FakeClock
from test_acquisition_session import CAMERA, CAPTURE_CFG, START

OK, FAIL = CaptureOutcome(True, record={"image_id": "x"}), CaptureOutcome(False, error="no frame")


class Flags:
    def __init__(self):
        self.camera_ok, self.pending = True, 0


def controller(requested=3, interval=3.0, review_each=False):
    flags = Flags()
    c = CaptureController(requested, interval, review_each, camera_ok=lambda: flags.camera_ok,
                          pending_reviews=lambda: flags.pending)
    return c, flags


def test_countdown_three_two_one_capture():
    c, _ = controller()
    c.start(10.0)
    shown = []
    for t in (10.0, 10.5, 11.0, 11.5, 12.0, 12.99):
        assert c.tick(t) is False
        shown.append(c.countdown_value(t))
    assert shown == [3, 3, 2, 2, 1, 1]
    assert c.tick(13.0) is True and c.phase == PHASE_CAPTURING and c.attempts == 1
    assert c.countdown_value(13.0) is None
    assert c.tick(13.1) is False                   # no second capture while one is in progress


def test_review_each_waits_for_the_operator():
    c, flags = controller(review_each=True)
    c.start(0.0)
    assert c.tick(3.0)
    flags.pending = 1
    c.capture_done(OK, 3.2)
    assert c.phase == PHASE_WAITING_REVIEW
    for t in (5.0, 30.0, 300.0):
        assert c.tick(t) is False and c.phase == PHASE_WAITING_REVIEW
    flags.pending = 0                               # operator pressed KEEP or REJECT
    c.tick(301.0)
    assert c.phase == PHASE_COUNTDOWN and c.countdown_value(301.0) == 3
    assert c.tick(304.0) and c.attempts == 2


def test_countdown_waits_for_a_correctly_placed_pcb():
    flags = Flags()
    ready = {"ok": False}
    c = CaptureController(2, 3.0, True, camera_ok=lambda: flags.camera_ok, pending_reviews=lambda: flags.pending,
                          scene_ready=lambda: ready["ok"])
    c.start(0.0)
    for t in (0.1, 3.0, 10.0, 60.0):                # PCB misplaced (live check RECAPTURE): never captured
        assert c.tick(t) is False and c.phase == PHASE_WAITING_PCB and c.attempts == 0
    ready["ok"] = True                              # PCB OK: a full 3 s countdown, then the capture
    c.tick(61.0)
    assert c.phase == PHASE_COUNTDOWN and c.countdown_value(61.0) == 3
    assert c.tick(63.9) is False and c.tick(64.0) is True and c.attempts == 1
    c.capture_done(OK, 64.1)
    ready["ok"] = False                             # moved during the next countdown: the countdown stops
    c.tick(65.0)
    assert c.phase == PHASE_WAITING_PCB


def test_recapture_is_retaken_automatically_and_not_counted():
    c, flags = controller(requested=2, review_each=True)
    c.start(0.0)
    assert c.tick(3.0)
    c.capture_done(CaptureOutcome(True, record={"image_id": "a"}, retake=True), 3.1)   # quarantined automatically
    assert c.phase == PHASE_COUNTDOWN and c.attempts == 0 and c.retakes == 1          # no review wait, not counted
    assert c.tick(6.1)
    flags.pending = 1
    c.capture_done(OK, 6.2)
    assert c.phase == PHASE_WAITING_REVIEW and c.successes == 1 and c.consecutive_retakes == 0


def test_too_many_retakes_in_a_row_stop_the_session():
    c, _ = controller(requested=5)
    c.start(0.0)
    t = 0.0
    for _ in range(3):
        t += 3.0
        assert c.tick(t)
        c.capture_done(CaptureOutcome(True, record={"image_id": "x"}, retake=True), t)
    assert c.finished and c.finish_status == "stopped" and "3 captures in a row" in c.finish_reason


def test_kept_goal_counts_kept_images_of_the_board():
    flags, kept = Flags(), {"n": 7}                 # e.g. 7 images of this board kept in an earlier session
    c = CaptureController(10, 3.0, True, camera_ok=lambda: True, pending_reviews=lambda: flags.pending,
                          kept=lambda: kept["n"])
    c.start(0.0)
    t = 0.0
    for _ in range(2):                              # automatically kept (no pending review)
        t += 3.0
        assert c.tick(t)
        kept["n"] += 1
        c.capture_done(OK, t)
        assert c.phase == PHASE_COUNTDOWN
    t += 3.0
    assert c.tick(t)
    flags.pending = 1                               # a REVIEW image: waits for the operator
    c.capture_done(OK, t)
    assert c.phase == PHASE_WAITING_REVIEW and not c.finished
    flags.pending, kept["n"] = 0, 10                # operator KEEP: 10 of 10
    c.tick(t + 5)
    assert c.finished and c.finish_status == "completed" and c.goal_reached()


def test_board_sequence_goes_board_after_board():
    seq = BoardSequence(1, boards=2, removal_s=1.0)
    assert seq.state == SEQ_PLACE and seq.update(None, 0.0) is False and seq.update(False, 1.0) is False
    assert seq.update(True, 2.0) is True and seq.state == SEQ_BOARD and seq.number == 1   # board 1 placed: start
    seq.board_finished(completed=True)
    assert seq.state == SEQ_REMOVE and seq.completed == [1]
    for t in (10.0, 11.0, 12.0):                    # board still in place: nothing happens
        assert seq.update(True, t) is False and seq.state == SEQ_REMOVE and seq.number == 1
    seq.update(False, 13.0)
    assert seq.state == SEQ_REMOVE                  # just lifted: must stay away for removal_s
    seq.update(True, 13.5)                          # put back: the removal timer restarts
    seq.update(False, 14.0)
    seq.update(False, 15.0)
    assert seq.state == SEQ_PLACE and seq.number == 2
    assert seq.update(True, 16.0) is True and seq.number == 2
    seq.board_finished(completed=True)
    seq.update(False, 17.0)
    seq.update(False, 18.1)
    assert seq.finished and seq.completed == [1, 2] and seq.stop_reason is None


def test_board_sequence_reopen_and_stop():
    seq = BoardSequence(5)
    seq.update(True, 0.0)
    seq.board_finished(True)
    assert seq.reopen() and seq.state == SEQ_PLACE and seq.number == 5 and seq.completed == []   # REJECT LAST
    assert seq.update(True, 1.0) is True and seq.number == 5                                     # same board again
    seq.board_finished(False, "stopped by the operator")
    assert seq.finished and seq.stop_reason == "stopped by the operator" and not seq.reopen()
    with pytest.raises(ValueError):
        BoardSequence(0)


def test_failed_capture_does_not_wait_for_review():
    c, flags = controller(review_each=True)
    c.start(0.0)
    c.tick(3.0)
    c.capture_done(FAIL, 3.1)
    assert c.phase == PHASE_COUNTDOWN and c.failures == 1


def test_continuous_capture_and_completion_counts_failures():
    c, flags = controller(requested=3, review_each=False)
    c.start(0.0)
    t = 0.0
    for outcome in (OK, FAIL, OK):
        t += 3.0
        assert c.tick(t)
        flags.pending += outcome.ok                 # reviews pile up; capture continues
        c.capture_done(outcome, t)
    assert c.phase == PHASE_FINISHED and c.finish_status == "completed"
    assert (c.attempts, c.successes, c.failures) == (3, 2, 1)


def test_stop_during_countdown_and_during_capture():
    c, _ = controller(requested=50)
    c.start(0.0)
    c.tick(3.0)
    c.capture_done(OK, 3.0)
    c.stop()
    assert c.finished and c.finish_status == "stopped" and c.successes == 1

    c, _ = controller(requested=50)
    c.start(0.0)
    assert c.tick(3.0)
    c.stop()                                        # the capture in progress is finished and kept first
    assert c.phase == PHASE_CAPTURING
    c.capture_done(OK, 3.2)
    assert c.finished and c.finish_status == "stopped" and c.successes == 1
    c.stop()                                        # stopping twice is harmless
    assert c.finish_status == "stopped"


def test_camera_loss_pauses_without_using_an_attempt():
    c, flags = controller()
    c.start(0.0)
    flags.camera_ok = False
    assert c.tick(2.5) is False and c.phase == PHASE_CAMERA_LOST
    assert c.tick(10.0) is False and c.attempts == 0
    flags.camera_ok = True
    c.tick(11.0)
    assert c.phase == PHASE_COUNTDOWN and c.countdown_value(11.0) == 3   # full countdown again
    assert not c.tick(13.9) and c.tick(14.0)


def test_controller_misuse():
    with pytest.raises(ValueError):
        CaptureController(0, 3, False, lambda: True, lambda: 0)
    c, _ = controller()
    with pytest.raises(RuntimeError):
        c.capture_done(OK, 0.0)
    c.start(0.0)
    with pytest.raises(RuntimeError):
        c.start(1.0)


# --- capture_once with a mock camera ------------------------------------------

def board_frame(seed: int) -> np.ndarray:
    """BGR test input: red rectangle (board-shaped) on a light background, with noise."""
    rng = np.random.default_rng(seed)
    img = np.full((180, 320, 3), 235, np.int16)
    img[27:153, 44:276] = (40, 30, 200)
    return np.clip(img + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)


class BoardCamera(FakeCamera):
    def read(self):
        if self.frames:
            return self.frames.pop(0)
        self.counter += 1
        return board_frame(self.counter)


@pytest.fixture
def rig(tmp_path, active_profile):
    store = AcquisitionStore(settings_from_configs(tmp_path / "data", active_profile, CAPTURE_CFG))
    session = store.create_session(SessionConfig("good", "after_tht", 3, 3.0, "GOOD_001"), CAMERA, started_at=START)
    clock = FakeClock(0.0)
    g = FrameGrabber(BoardCamera(), clock=clock, failures_before_disconnect=2, frozen_after_s=3.0)
    return store, session, clock, g


def test_capture_once_saves_a_fresh_frame(rig):
    store, session, clock, g = rig
    g.step()
    clock.t = 1.0
    g.step()
    outcome = capture_once(g, store, session, trigger_t=0.5, timeout_s=0)
    assert outcome.ok and outcome.record["frame_seq"] == 2
    assert outcome.record["camera"] == "DroidCam Video" and outcome.record["width"] == 320
    assert (store.root / outcome.record["original_file"]).is_file()
    assert session.counts["successful_captures"] == 1


def test_capture_once_records_failures(rig):
    store, session, clock, g = rig
    # camera not live yet (nothing read)
    out = capture_once(g, store, session, trigger_t=0.0, timeout_s=0)
    assert not out.ok and "camera not live" in out.error
    g.step()
    # no frame newer than the end of the countdown
    out = capture_once(g, store, session, trigger_t=clock.t, timeout_s=0)
    assert not out.ok and "no new frame" in out.error
    # same picture as the previous capture: a frozen stream must not produce a "new" image
    out = capture_once(g, store, session, trigger_t=-1.0, previous_signature=g.latest().signature, timeout_s=0)
    assert not out.ok and "identical" in out.error
    # disconnected
    g.camera.frames = [None, None]
    g.step()
    g.step()
    out = capture_once(g, store, session, trigger_t=-1.0, timeout_s=0)
    assert not out.ok and "disconnected" in out.error
    assert session.images == [] and session.counts["failed_captures"] == 4
    assert list((session.folder / "images").iterdir()) == []


def test_full_session_capture_review_keep(rig):
    """GOOD session of 3 images, reviewed after each capture: the acceptance workflow without a camera."""
    store, session, clock, g = rig
    c = CaptureController(3, 3.0, True, camera_ok=lambda: g.is_live, pending_reviews=lambda: len(session.pending()))
    g.step()
    c.start(clock.t)
    signature = None
    while not c.finished:
        clock.t += 0.5
        g.step()
        if c.tick(clock.t):
            outcome = capture_once(g, store, session, trigger_t=clock.t - 0.25, previous_signature=signature,
                                   timeout_s=0)
            signature = outcome.signature
            c.capture_done(outcome, clock.t)
            if outcome.ok:
                store.review(session, outcome.image_id, "keep")
    store.finish(session, c.finish_status)
    assert session.status == "completed" and session.counts["kept_count"] == 3
    curated = sorted(p.name for p in (store.root / "curated" / "good" / "GOOD_001" / session.id).iterdir())
    assert curated == [f"GOOD_001_TOP_{session.id}_C00{i}.png" for i in (1, 2, 3)]
    assert len(list(Path(session.folder / "images").iterdir())) == 3   # originals preserved
