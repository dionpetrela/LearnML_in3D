"""Hybrid policy: Arrow following (rule-based) + ML for obstacles only.

Usage:
    python 03_benchmark.py --tag v12 --weights nav_v10.npz --module drive2win.homing_policy_v12 --seeds 42 7 99
"""
from __future__ import annotations
import numpy as np

from drive2win import nn
from drive2win.normalize import sensors_to_input, clip_action

# ── Arrow following constants ───────────────────────────────────────────
STEER_SIGN      = -1.0
ARROW_THROTTLE  = 0.85
ARROW_STEER_KP  = 0.9
STRAIGHT_BOOST  = 0.15
CORNER_FACTOR   = 0.25

# ── ML obstacle blend ───────────────────────────────────────────────────
OBSTACLE_THRESH = 0.25

# ── Stuck recovery ──────────────────────────────────────────────────────
GRACE_FRAMES      = 100
STUCK_THRESH      = 60
RECOVERY_FRAMES   = 60
STUCK_SPEED       = 0.3
REORIENT_FRAMES   = 25
REORIENT_KP       = 1.5
WIGGLE_THRESH     = 150
WIGGLE_PERIOD     = 3

# ── EMA smoothing ───────────────────────────────────────────────────────
EMA_ALPHA = 0.7

# 9-input format: [speed, heading_error, cp_dist, front, front_left, front_right, ...]
# indices:         0       1            2       3        4           5
FRONT_IDX = 3
FRONT_LEFT_IDX = 4
FRONT_RIGHT_IDX = 5


def make_policy(weights_path: str):
    w = nn.load(weights_path)
    
    ctx = {
        "frame": 0,
        "stuck": 0,
        "recovery": 0,
        "reorient": 0,
        "ema_t": 0.0,
        "ema_s": 0.0,
    }

    def policy(state: dict) -> tuple[float, float]:
        ctx["frame"] += 1
        frame = ctx["frame"]
        
        sensors = state.get("sensors") or {}
        
        heading_error = float(sensors.get("heading_error", 0.0))
        speed = float(sensors.get("speed", 0.0))
        
        # ── ML forward pass ────────────────────────────────────────────
        x_raw = sensors_to_input(sensors)
        x = x_raw  # Use full 12 inputs for ML (obstacle detection)
        # But we need 9-input network. Convert 12 to 9 by selecting cols
        x_9 = np.array([x[0], x[1], x[2], x[3], x[4], x[5], x[9], x[10], x[11]], dtype=np.float32)
        ml_out = nn.forward(x_9, w)
        ml_thr = float(ml_out[0]) if ml_out.size > 0 else 0.5
        ml_steer = float(ml_out[1]) if ml_out.size > 1 else 0.0
        
        # ── Arrow following ────────────────────────────────────────────
        raw_steer = STEER_SIGN * heading_error * ARROW_STEER_KP
        arr_steer = float(np.clip(raw_steer, -1.0, 1.0))
        
        straight = float(max(0.0, 1.0 - abs(heading_error) / 0.26))
        arr_thr = float(min(1.0, ARROW_THROTTLE * (1.0 - CORNER_FACTOR * abs(arr_steer))
                            + STRAIGHT_BOOST * straight))
        
        # ── Obstacle blend using raw rays (from original sensors) ───────
        rays = sensors.get("rays", [50.0] * 8)
        front = rays[0] if rays else 50.0
        front_left = rays[1] if len(rays) > 1 else 50.0
        front_right = rays[7] if len(rays) > 7 else 50.0
        
        front_clearance = min(front, front_left, front_right) / 50.0  # normalize to [0,1]
        obstacle_factor = float(np.clip(1.0 - front_clearance / OBSTACLE_THRESH, 0.0, 1.0))
        
        # ── Blend arrow + ML ───────────────────────────────────────────
        thr = (1.0 - obstacle_factor) * arr_thr + obstacle_factor * ml_thr
        steer = (1.0 - obstacle_factor) * arr_steer + obstacle_factor * ml_steer
        
        # ── Recovery state machine ─────────────────────────────────────
        if ctx["recovery"] > 0:
            thr = -1.0
            steer = STEER_SIGN * (1.0 if heading_error >= 0 else -1.0)
            ctx["recovery"] -= 1
            if ctx["recovery"] == 0:
                ctx["ema_t"] = 0.0
                ctx["ema_s"] = 0.0
                ctx["reorient"] = REORIENT_FRAMES
        
        elif ctx["reorient"] > 0:
            snap_steer = STEER_SIGN * heading_error * REORIENT_KP
            steer = float(np.clip(snap_steer, -1.0, 1.0))
            thr = ARROW_THROTTLE * 0.6
            ctx["reorient"] -= 1
        
        elif frame > GRACE_FRAMES:
            if speed < STUCK_SPEED:
                ctx["stuck"] += 1
            else:
                ctx["stuck"] = 0
            
            if ctx["stuck"] >= WIGGLE_THRESH:
                wiggle_sign = 1.0 if (ctx["stuck"] // WIGGLE_PERIOD) % 2 == 0 else -1.0
                thr = wiggle_sign
                steer = STEER_SIGN * (1.0 if heading_error >= 0 else -1.0)
            elif ctx["stuck"] >= STUCK_THRESH:
                ctx["recovery"] = RECOVERY_FRAMES
                ctx["stuck"] = 0
        
        # ── EMA smoothing ───────────────────────────────────────────────
        ctx["ema_t"] = EMA_ALPHA * thr + (1 - EMA_ALPHA) * ctx["ema_t"]
        ctx["ema_s"] = EMA_ALPHA * steer + (1 - EMA_ALPHA) * ctx["ema_s"]
        
        return clip_action(np.array([ctx["ema_t"], ctx["ema_s"]]))
    
    return policy
