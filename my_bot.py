#!/usr/bin/env python3
"""Tournament bot using v15 homing policy."""
import argparse
import numpy as np

from game_client import RoomBot
from drive2win.homing_policy import make_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="main", help="Tournament room name")
    ap.add_argument("--name", default="DionPetrela", help="Your display name")
    ap.add_argument("--host", default="ml.ferit.tech", help="Server host")
    ap.add_argument("--secure", action="store_true", default=True, help="Use wss")
    ap.add_argument("--weights", default="nav_v13.npz", help="Path to weights file")
    ap.add_argument("--hz", type=float, default=20.0, help="Control rate")
    args = ap.parse_args()

    # Load your policy
    policy_fn = make_policy(args.weights)

    def controller(obs):
        # obs is the tournament observation dict
        throttle, steering = policy_fn(obs)
        return throttle, steering

    scheme = "wss" if args.secure else "ws"
    server_url = f"{scheme}://{args.host}"

    bot = RoomBot(server_url, room=args.room, name=args.name)
    standings = bot.run(controller, hz=args.hz)
    print("\n=== FINAL STANDINGS ===")
    for s in standings:
        print(f"  #{s.get('rank')} {s.get('name')}: {s.get('total_checkpoints')} checkpoints")

if __name__ == "__main__":
    main()
