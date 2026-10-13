"""Synthetic board generator for testing PCB inspection architecture."""

import cv2
import numpy as np


def make_test_pcb(
    seed: int = 42,
    color_shift: tuple[int, int, int] = (0, 0, 0),
    component_offset: tuple[int, int] = (0, 0),
    missing_component: str | None = None,
    width_mm: float = 116.0,
    height_mm: float = 83.0,
    ppm: float = 10.0,  # 10 px/mm -> 1160 x 830 px
) -> np.ndarray:
    """Generates a detailed synthetic PCB board image with rich SIFT features."""
    w_px = round(width_mm * ppm)
    h_px = round(height_mm * ppm)
    rng = np.random.default_rng(seed)

    # 1. Base soldermask (Red substrate typical of LG Indicator)
    base_color = np.array([40 + color_shift[0], 35 + color_shift[1], 190 + color_shift[2]], dtype=np.int16)
    base_color = np.clip(base_color, 0, 255).astype(np.uint8)

    # Rich texture for SIFT features
    noise = rng.uniform(0.85, 1.15, (h_px, w_px, 1)).astype(np.float32)
    board = np.clip(base_color.astype(np.float32) * noise, 0, 255).astype(np.uint8)

    # Board border margin
    cv2.rectangle(board, (0, 0), (w_px - 1, h_px - 1), (20, 20, 100), 2)

    # 2. Add silkscreen ground planes & traces (white/gold)
    for x in range(50, w_px - 50, 80):
        cv2.line(board, (x, 40), (x, h_px - 40), (220, 220, 220), 2)
    for y in range(50, h_px - 50, 60):
        cv2.line(board, (40, y), (w_px - 40, y), (200, 200, 200), 1)

    # Diagonal fiducials and crosshairs at corners
    for cx, cy in [(80, 80), (w_px - 80, 80), (80, h_px - 80), (w_px - 80, h_px - 80)]:
        cv2.circle(board, (cx, cy), 18, (240, 240, 240), 2)
        cv2.drawMarker(board, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)

    # 3. Add components (IC chips, resistors, capacitors)
    components = [
        ("U1_MCU", 400, 250, 160, 160, "IC"),
        ("U2_REG", 700, 200, 80, 60, "IC"),
        ("C14", 250, 200, 40, 30, "CAP"),
        ("R5", 250, 300, 35, 20, "RES"),
        ("D1_LED", 850, 350, 30, 30, "LED"),
        ("C20", 500, 550, 45, 30, "CAP"),
        ("R12", 700, 500, 35, 20, "RES"),
    ]

    ox, oy = component_offset

    for refdes, cx, cy, cw, ch, comp_type in components:
        if missing_component == refdes:
            # Bare copper solder pads left behind
            pad_w = cw // 4
            cv2.rectangle(board, (cx - cw//2 + ox, cy - ch//2 + oy),
                          (cx - cw//2 + pad_w + ox, cy + ch//2 + oy), (80, 170, 210), -1)
            cv2.rectangle(board, (cx + cw//2 - pad_w + ox, cy - ch//2 + oy),
                          (cx + cw//2 + ox, cy + ch//2 + oy), (80, 170, 210), -1)
            continue

        x0, y0 = cx - cw // 2 + ox, cy - ch // 2 + oy
        x1, y1 = cx + cw // 2 + ox, cy + ch // 2 + oy

        if comp_type == "IC":
            cv2.rectangle(board, (x0, y0), (x1, y1), (30, 30, 30), -1)
            cv2.putText(board, refdes, (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            # Pins
            for px in range(x0 + 10, x1 - 10, 15):
                cv2.rectangle(board, (px, y0 - 6), (px + 6, y0), (180, 190, 200), -1)
                cv2.rectangle(board, (px, y1), (px + 6, y1 + 6), (180, 190, 200), -1)
        elif comp_type == "CAP":
            cv2.rectangle(board, (x0, y0), (x1, y1), (60, 120, 160), -1)
            cv2.rectangle(board, (x0, y0), (x0 + 8, y1), (200, 200, 200), -1)
            cv2.rectangle(board, (x1 - 8, y0), (x1, y1), (200, 200, 200), -1)
        elif comp_type == "RES":
            cv2.rectangle(board, (x0, y0), (x1, y1), (40, 40, 40), -1)
            cv2.rectangle(board, (x0, y0), (x0 + 6, y1), (210, 210, 210), -1)
            cv2.rectangle(board, (x1 - 6, y0), (x1, y1), (210, 210, 210), -1)
        elif comp_type == "LED":
            cv2.circle(board, (cx + ox, cy + oy), cw // 2, (30, 180, 50), -1)

    return board


def embed_in_scene(
    board: np.ndarray,
    scene_size: tuple[int, int] = (1400, 1800),  # (H, W)
    shift_xy: tuple[float, float] = (0, 0),
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    background_color: tuple[int, int, int] = (60, 60, 60),
) -> np.ndarray:
    """Embeds board inside a camera frame with perspective/pose transformations."""
    sh, sw = scene_size
    scene = np.full((sh, sw, 3), background_color, dtype=np.uint8)

    bh, bw = board.shape[:2]
    # Place board roughly in center
    cx, cy = sw / 2 + shift_xy[0], sh / 2 + shift_xy[1]

    # Compute transformation matrix
    M = cv2.getRotationMatrix2D((bw / 2, bh / 2), rotation_deg, scale)
    M[0, 2] += cx - bw / 2
    M[1, 2] += cy - bh / 2

    # Warp board into scene
    warped_board = cv2.warpAffine(board, M, (sw, sh), borderValue=background_color)
    board_mask = cv2.warpAffine(np.full((bh, bw), 255, dtype=np.uint8), M, (sw, sh), borderValue=0)

    # Composite onto scene
    np.copyto(scene, warped_board, where=board_mask[..., None] > 128)
    return scene
