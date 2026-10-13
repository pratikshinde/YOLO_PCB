"""Random phone-photo degradations for training copies: focus blur, exposure / saturation changes, sensor noise, JPEG."""
import cv2
import numpy as np


def degrade_random(im, rng: np.random.Generator):
    out = im
    if rng.random() < 0.6:                                   # out of focus / motion
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.6, 2.0)))
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] *= float(rng.uniform(0.55, 1.3))             # darker / brighter
    hsv[..., 1] *= float(rng.uniform(0.8, 1.3))              # paler / more saturated red
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.4:
        out = np.clip(out.astype(np.float32) + rng.normal(0, float(rng.uniform(2, 10)), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 90))])[1], cv2.IMREAD_COLOR)
    return out
