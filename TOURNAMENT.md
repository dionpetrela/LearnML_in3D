# Tournament Guide

## Tournament day flow:
1. Professor announces room name
2. Run your bot: `python my_bot.py --room <room-name> --name DionPetrela`
3. Race starts when everyone ready
4. Standings print in terminal

## Observation keys your bot receives:
- speed, heading, rays, ground_friction
- navigation.distance, navigation.heading_error
- checkpoints_passed, round_index, race_phase

Your controller returns (throttle, steering) both in [-1, 1]
