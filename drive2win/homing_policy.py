"""MLP policy — steering-only, stuck recovery, checkpoint homing + orbit escape.

Usage:
    python 03_benchmark.py --tag v9 --weights nav_v8.npz --module drive2win.homing_policy --data data_v6.npz --seeds 42
"""
from __future__ import annotations
import numpy as np

from drive2win import nn
from drive2win.normalize import sensors_to_input

THROTTLE         = 0.9
STEER_GAIN       = 1.4
STUCK_THRESHOLD  = 15
REVERSE_FRAMES   = 10
STUCK_SPEED      = 0.3
RAY_WEDGE        = 4.0
PURE_STUCK_THR   = 50
PURE_STUCK_SPEED = 0.15

CP_HOMING_DIST   = 50.0
CP_HOMING_MAX    = 1.0
CP_HOMING_GAIN   = 5.0
CP_BRAKE_DIST    = 10.0
CP_MIN_THROTTLE  = 0.55
CP_GATE_DIST     = 5.0
CP_ORBIT_DIST    = 8.0
CP_ORBIT_FRAMES  = 25

# Columns to keep from 12-input to 9-input
KEEP_COLS = [0, 1, 2, 3, 4, 5, 9, 10, 11]


def make_policy(weights_path: str):
    w = nn.load(weights_path)
    
    prev          = np.zeros(1, dtype=np.float32)
    stuck_count   = 0
    reverse_count = 0
    prev_cp_dist  = 100.0
    orbit_frames  = 0

    def policy(state: dict) -> tuple[float, float]:
        nonlocal prev, stuck_count, reverse_count, prev_cp_dist, orbit_frames

        sensors = state["sensors"]
        speed   = sensors.get("speed", 1.0)
        rays    = sensors.get("rays", [50.0] * 8)
        front   = rays[0] if rays else 50.0
        left    = rays[6] if len(rays) > 6 else 50.0
        right   = rays[2] if len(rays) > 2 else 50.0

        wedged     = (speed < STUCK_SPEED and front < RAY_WEDGE
                      and (left < RAY_WEDGE or right < RAY_WEDGE))
        pure_stuck = speed < PURE_STUCK_SPEED

        if wedged or pure_stuck:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= (STUCK_THRESHOLD if wedged else PURE_STUCK_THR):
            reverse_count = REVERSE_FRAMES
            stuck_count   = 0

        cp_near = sensors.get("checkpoint_distance", 100.0) < CP_ORBIT_DIST
        if cp_near:
            orbit_frames += 1
        else:
            orbit_frames = 0
        if orbit_frames >= CP_ORBIT_FRAMES and reverse_count == 0:
            reverse_count = REVERSE_FRAMES * 2
            orbit_frames  = 0

        if reverse_count > 0:
            reverse_count -= 1
            steer = -0.8 if right < left else 0.8
            prev  = np.array([steer], dtype=np.float32)
            return (-1.0, steer)

        # --- model steering with 9 inputs ---
        x_raw = sensors_to_input(sensors)
        x = x_raw[KEEP_COLS]  # Select only 9 features
        raw   = nn.forward(x, w)
        prev  = raw.copy()
        steer = float(np.clip(raw[0] * STEER_GAIN, -1.0, 1.0))

        # --- checkpoint homing ---
        cp_dist      = sensors.get("checkpoint_distance", 100.0)
        heading_err  = sensors.get("heading_error", 0.0)
        old_cp_dist  = prev_cp_dist
        approaching  = cp_dist < old_cp_dist
        prev_cp_dist = cp_dist
        throttle     = THROTTLE

        if cp_dist < CP_GATE_DIST:
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            steer    = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            throttle = CP_MIN_THROTTLE

        elif cp_dist < CP_ORBIT_DIST and not approaching:
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            steer    = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            throttle = CP_MIN_THROTTLE

        elif cp_dist < CP_HOMING_DIST:
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            homing_steer = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            t     = (CP_HOMING_DIST - cp_dist) / CP_HOMING_DIST
            blend = t * CP_HOMING_MAX
            steer = steer * (1.0 - blend) + homing_steer * blend
            if cp_dist < CP_BRAKE_DIST:
                brake_t  = (cp_dist - CP_GATE_DIST) / (CP_BRAKE_DIST - CP_GATE_DIST)
                brake_t  = float(np.clip(brake_t, 0.0, 1.0))
                throttle = CP_MIN_THROTTLE + (THROTTLE - CP_MIN_THROTTLE) * brake_t
            else:
                throttle = THROTTLE

        return (throttle, steer)

    return policy
