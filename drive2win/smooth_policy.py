"""Action smoothing + stuck detection + hard-coded recovery for keyboard-driven models."""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


def make_policy(weights_path: str, alpha: float = 0.7):
    """
    alpha: smoothing factor (0 = no smoothing, 1 = very smooth)
    """
    w = nn_mod.load(weights_path)
    prev_action = np.zeros(2, dtype=np.float32)
    
    # Stuck detection state
    position_history = []
    stuck_counter = 0
    recovery_mode = 0  # seconds left in recovery mode
    
    def policy(state):
        nonlocal prev_action, position_history, stuck_counter, recovery_mode
        
        # Check if in recovery mode (hard-coded reverse + turn)
        if recovery_mode > 0:
            recovery_mode -= 1
            # Reverse and turn hard
            throttle = -0.7
            steering = 0.8 if recovery_mode % 2 == 0 else -0.8  # alternate direction
            return clip_action((throttle, steering))
        
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
                    recovery_mode = 45  # 2.25 seconds of recovery (45 frames at 20Hz)
                    
                    # Determine which way to turn (based on rays or default)
                    sensors = state.get("sensors", {})
                    rays = sensors.get("rays", [10.0] * 8)
                    left_space = sum(rays[1:4])  # left side rays
                    right_space = sum(rays[4:7])  # right side rays
                    steering = 0.9 if left_space > right_space else -0.9
                    
                    throttle = -0.7
                    return clip_action((throttle, steering))
        
        # Normal smooth action
        x = sensors_to_input(state["sensors"])
        raw = nn_mod.forward(x, w)
        smoothed = alpha * prev_action + (1 - alpha) * raw
        prev_action = smoothed.copy()
        
        return clip_action(smoothed)
    
    return policy
