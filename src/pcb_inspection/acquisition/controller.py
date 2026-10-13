"""Timed batch capture, independent of the UI: countdown -> capture -> (review) -> next capture.

The Tk application drives CaptureController with tick(now); tests drive it
with a fake clock. capture_once() takes one real capture: it waits for a frame
that arrived after the countdown ended (never a frame already shown), refuses
a picture identical to the previous capture (frozen stream), saves it through
the AcquisitionStore and records every failure without creating an image record.

A session ends after the requested number of capture attempts (successful +
failed), or when the operator presses STOP. Images already captured are kept
in both cases.

Clean collection additions (both optional):
  scene_ready   the countdown runs only while it returns True (live PCB check not RECAPTURE); otherwise the
                session WAITS FOR the PCB and starts a full countdown once it is ready again.
  retake        a capture marked retake (its capture check said RECAPTURE; it was quarantined automatically)
                does not use up a requested image; after max_consecutive_retakes in a row the session stops.
  kept          the goal is a number of KEPT images (one physical board), not of capture attempts.
BoardSequence chains such sessions board after board (GOOD_001 x 10, remove, GOOD_002 x 10, ...).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .camera import FrameGrabber, encode_png
from .session import STATUS_COMPLETED, STATUS_STOPPED, AcquisitionStore, CaptureError, Session

PHASE_READY = "READY"
PHASE_COUNTDOWN = "COUNTDOWN"
PHASE_CAPTURING = "CAPTURING"
PHASE_WAITING_REVIEW = "WAITING FOR REVIEW"
PHASE_CAMERA_LOST = "CAMERA DISCONNECTED"
PHASE_WAITING_PCB = "WAITING FOR PCB"
PHASE_FINISHED = "FINISHED"


@dataclass
class CaptureOutcome:
    ok: bool
    record: dict[str, Any] | None = None
    error: str | None = None
    signature: str | None = None
    retake: bool = False            # saved, but failed the capture check and was quarantined automatically

    @property
    def image_id(self) -> str | None:
        return self.record["image_id"] if self.record else None


class CaptureController:
    """State machine of one batch session. It decides WHEN to capture; capturing itself is done by the caller."""

    def __init__(self, requested: int, interval_s: float, review_each: bool,
                 camera_ok: Callable[[], bool], pending_reviews: Callable[[], int],
                 scene_ready: Callable[[], bool] | None = None, max_consecutive_retakes: int = 3,
                 kept: Callable[[], int] | None = None, max_attempts: int | None = None):
        """kept: when given, the session's goal is `requested` KEPT images (e.g. of one physical board, incl. earlier
        sessions of it) instead of `requested` capture attempts; max_attempts (default 3 x requested) is then the
        safety limit."""
        if requested < 1 or interval_s <= 0:
            raise ValueError("requested must be >= 1 and interval_s > 0")
        self.requested, self.interval_s, self.review_each = requested, interval_s, review_each
        self._camera_ok, self._pending_reviews = camera_ok, pending_reviews
        self._scene_ready = scene_ready or (lambda: True)
        self.max_consecutive_retakes = max_consecutive_retakes
        self._kept = kept
        self.max_attempts = max_attempts or 3 * requested
        self.phase = PHASE_READY
        self.attempts = self.successes = self.failures = self.retakes = 0
        self.consecutive_retakes = 0
        self.finish_status: str | None = None
        self.finish_reason: str | None = None
        self.last_outcome: CaptureOutcome | None = None
        self._deadline = 0.0
        self._stop_requested = False

    @property
    def finished(self) -> bool:
        return self.phase == PHASE_FINISHED

    def start(self, now: float) -> None:
        if self.phase != PHASE_READY:
            raise RuntimeError("session already started")
        self._countdown(now)

    def tick(self, now: float) -> bool:
        """Advance the state; True means: take a capture now, then call capture_done()."""
        if self.phase == PHASE_COUNTDOWN:
            if not self._camera_ok():
                self.phase = PHASE_CAMERA_LOST          # paused; the attempt is not used up
                return False
            if not self._scene_ready():
                self.phase = PHASE_WAITING_PCB          # PCB not placed correctly: no capture, attempt not used up
                return False
            if now >= self._deadline:
                self.phase = PHASE_CAPTURING
                self.attempts += 1
                return True
        elif self.phase == PHASE_CAMERA_LOST and self._camera_ok():
            self._countdown(now)                        # resume with a full countdown
        elif self.phase == PHASE_WAITING_PCB and self._camera_ok() and self._scene_ready():
            self._countdown(now)                        # PCB ready again: a full countdown
        elif self.phase == PHASE_WAITING_REVIEW and self._pending_reviews() == 0:
            if self._kept is not None and self.goal_reached():
                self._finish(STATUS_COMPLETED)          # the operator's KEEP completed the board
            else:
                self._countdown(now)
        return False

    def goal_reached(self) -> bool:
        return self._kept() >= self.requested if self._kept is not None else self.attempts >= self.requested

    @property
    def kept_count(self) -> int | None:
        return self._kept() if self._kept is not None else None

    def capture_done(self, outcome: CaptureOutcome, now: float) -> None:
        if self.phase != PHASE_CAPTURING:
            raise RuntimeError("capture_done() without a capture in progress")
        self.last_outcome = outcome
        if outcome.ok and outcome.retake:
            self.retakes += 1
            self.consecutive_retakes += 1
            self.attempts -= 1                          # quarantined automatically: taken again, not counted
        elif outcome.ok:
            self.successes += 1
            self.consecutive_retakes = 0
        else:
            self.failures += 1
        if self._stop_requested:
            self._finish(STATUS_STOPPED)
        elif outcome.retake and self.consecutive_retakes >= self.max_consecutive_retakes:
            self.finish_reason = (f"{self.consecutive_retakes} captures in a row failed the capture check; "
                                  "fix the placement / focus / light and start the session again")
            self._finish(STATUS_STOPPED)
        elif self.goal_reached():
            self._finish(STATUS_COMPLETED)
        elif self._kept is not None and self.attempts >= self.max_attempts:
            self.finish_reason = (f"{self.attempts} captures without reaching {self.requested} kept images; "
                                  "check the setup and start again")
            self._finish(STATUS_STOPPED)
        elif self.review_each and outcome.ok and not outcome.retake:
            self.phase = PHASE_WAITING_REVIEW
            self.tick(now)
        else:
            self._countdown(now)

    def stop(self) -> None:
        """Operator STOP. A capture already in progress is finished and kept first."""
        if self.phase == PHASE_CAPTURING:
            self._stop_requested = True
        elif self.phase != PHASE_FINISHED:
            self._finish(STATUS_STOPPED)

    def countdown_value(self, now: float) -> int | None:
        """Whole seconds left before the next capture (3, 2, 1), or None outside the countdown."""
        if self.phase != PHASE_COUNTDOWN:
            return None
        return max(1, math.ceil(self._deadline - now - 1e-9))

    def _countdown(self, now: float) -> None:
        self.phase = PHASE_COUNTDOWN
        self._deadline = now + self.interval_s

    def _finish(self, status: str) -> None:
        self.phase = PHASE_FINISHED
        self.finish_status = status


SEQ_PLACE, SEQ_BOARD, SEQ_REMOVE, SEQ_DONE = "PLACE BOARD", "CAPTURING BOARD", "REMOVE BOARD", "FINISHED"


class BoardSequence:
    """Automatic board-after-board collection: board N (one session, until its target of KEPT images) -> the operator
    lifts the board out -> board N+1 is placed -> its session starts by itself -> ...

        PLACE BOARD  --PCB in view-->  CAPTURING BOARD  --target reached-->  REMOVE BOARD
             ^                                                                    |
             +------------- no PCB in view for removal_s: next board number ------+

    The board ID of each physical board is the operator's: boards must be placed in ID order (label them). Removing
    the board between two IDs is required, so one board is never recorded under two IDs by staying in place.
    """

    def __init__(self, first_number: int, boards: int | None = None, removal_s: float = 1.0):
        if not 1 <= first_number <= 999 or (boards is not None and boards < 1):
            raise ValueError("first board number must be 1..999 and boards >= 1 (or None = until STOP)")
        self.number, self.boards, self.removal_s = first_number, boards, removal_s
        self.completed: list[int] = []
        self.state = SEQ_PLACE
        self.stop_reason: str | None = None
        self._absent_since: float | None = None

    def update(self, pcb_present: bool | None, now: float) -> bool:
        """pcb_present: True / False from the live PCB check, None = unknown. Returns True when the session of board
        `number` must start now."""
        if self.state == SEQ_REMOVE:
            if pcb_present is False:
                if self._absent_since is None:
                    self._absent_since = now
                if now - self._absent_since >= self.removal_s:
                    self._absent_since = None
                    if self.boards is not None and len(self.completed) >= self.boards:
                        self.state = SEQ_DONE
                    else:
                        self.number += 1
                        self.state = SEQ_PLACE
                        if self.number > 999:
                            self.finish("board number 999 reached")
            else:
                self._absent_since = None
        elif self.state == SEQ_PLACE and pcb_present:
            self.state = SEQ_BOARD
            return True
        return False

    def board_finished(self, completed: bool, reason: str | None = None) -> None:
        """The session of the current board ended: target reached (completed) -> REMOVE BOARD; otherwise the whole
        sequence stops (operator STOP, too many failed captures)."""
        if self.state != SEQ_BOARD:
            return
        if completed:
            self.completed.append(self.number)
            self.state = SEQ_REMOVE
        else:
            self.finish(reason or "stopped")

    def reopen(self) -> bool:
        """The just-completed board lost a kept image (operator REJECT LAST) while it is still in place: capture it
        again under the same ID. Only before the board was removed."""
        if self.state != SEQ_REMOVE or not self.completed or self.completed[-1] != self.number:
            return False
        self.completed.pop()
        self.state, self._absent_since = SEQ_PLACE, None
        return True

    def finish(self, reason: str) -> None:
        self.state, self.stop_reason = SEQ_DONE, reason

    @property
    def finished(self) -> bool:
        return self.state == SEQ_DONE


def capture_once(grabber: FrameGrabber, store: AcquisitionStore, session: Session, trigger_t: float,
                 previous_signature: str | None = None, timeout_s: float = 2.0,
                 encode: Callable[[Any], bytes] = encode_png) -> CaptureOutcome:
    """Take one capture: a fresh frame -> saved, verified, quality-checked image record. Failures are recorded."""

    def fail(reason: str) -> CaptureOutcome:
        store.record_failure(session, reason)
        return CaptureOutcome(False, error=reason)

    if not grabber.is_live:
        return fail(f"camera not live ({grabber.status}): {grabber.message or 'no picture'}")
    frame = grabber.wait_for_frame_after(trigger_t, timeout_s)
    if frame is None:
        return fail(f"no new frame from the camera within {timeout_s:g} s")
    if not grabber.is_live:
        return fail(f"camera not live ({grabber.status}) at capture time")
    if previous_signature is not None and frame.signature == previous_signature:
        return fail("frame identical to the previous capture (camera stream frozen?)")
    try:
        record = store.save_capture(session, encode(frame.image), frame.wall_time, grabber.info(), frame.seq)
    except (CaptureError, OSError) as e:
        return fail(str(e))
    return CaptureOutcome(True, record=record, signature=frame.signature)
