"""Image acquisition: bringing camera images into the system (camera-first, no manual metadata per image).

ingest      Experiment 0: phone photos dropped into an inbox
camera      live camera (phone via DroidCam): device list by name, frame grabber, disconnect / freeze detection
session     acquisition sessions: IDs, files, session.json, quality check, KEEP / REJECT review
controller  timed batch capture (countdown, review gating, stop, camera loss)
ui          Tkinter acquisition + review application (scripts/acquire.py)
"""
