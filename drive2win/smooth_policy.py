"""Action smoothing + stuck detection + hard-coded recovery for keyboard-driven models.
   v7: Hardcoded throttle (0.9 forward, -0.7 reverse), steering-only learning.
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


def make_policy(weights_path: str, alpha: float = 0.7):
    """
    alpha: smoothing factor (0 = no smoothing, 1 = very smooth)
    """
    w = nn_mod.load(weights_path)
    prev_steering = 0.0
    
    # Stuck detection state
    position_history = []
    stuck_counter = 0
    recovery_mode = 0  # frames left in recovery mode
    
    def policy(state):
        nonlocal prev_steering, position_history, stuck_counter, recovery_mode
        
        # Hardcoded throttle values
        NORMAL_THROTTLE = 0.9
        REVERSE_THROTTLE = -0.7
        
        # Check if in recovery mode (hard-coded reverse + turn)
        if recovery_mode > 0:
            recovery_mode -= 1
            # Alternate turning direction each recovery frame for better escape
            steering = 0.8 if (recovery_mode // 5) % 2 == 0 else -0.8
            return clip_action((REVERSE_THROTTLE, steering))
        
        # Track position for stuck detection
        pos = state.get("position")
        if pos and isinstance(pos, dict) and "x" in pos and "z" in pos:
            current_pos = np.array([float(pos["x"]), float(pos["z"])])
            position_history.append(current_pos)
            if len(position_history) > 30:
                position_history.pop(0)
            
            # Check if stuck (barely moved in last 30 frames)
            if len(position_history) >= 30:
                start_pos = position_history[0]
                end_pos = position_history[-1]
                distance_moved = np.linalg.norm(end_pos - start_pos)
                
                if distance_moved < 0.5:  # stuck!
                    stuck_counter += 1
                else:
                    stuck_counter = max(0, stuck_counter - 1)
                
                # If stuck for 1.5 seconds (30 frames at 20Hz = 1.5s), trigger recovery
                if stuck_counter >= 30:
                    stuck_counter = 0
                    recovery_mode = 45  # 2.25 seconds of recovery
                    
                    # Determine which way to turn based on front rays
                    sensors = state.get("sensors", {})
                    rays = sensors.get("rays", [10.0] * 8)
                    left_space = rays[1] if len(rays) > 1 else 10.0   # ray_1_+45
                    right_space = rays[7] if len(rays) > 7 else 10.0  # ray_7_-45
                    steering = 0.9 if left_space > right_space else -0.9
                    
                    return clip_action((REVERSE_THROTTLE, steering))
        
        # Normal forward driving
        x = sensors_to_input(state["sensors"])
        raw = nn_mod.forward(x, w)
        
        # raw is steering only (shape: scalar or 1-element array)
        steering = float(raw) if np.isscalar(raw) else float(raw[0])
        
        # Apply smoothing to steering
        smoothed_steering = alpha * prev_steering + (1 - alpha) * steering
        prev_steering = smoothed_steering
        
        # Hardcoded throttle, smoothed steering
        return clip_action((NORMAL_THROTTLE, smoothed_steering))
    
    return policy