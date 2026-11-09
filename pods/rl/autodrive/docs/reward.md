# Reward

The weights are in `REWARD_PRESETS` in [rl/autodrive_env.py](../rl/autodrive_env.py). Choose a preset
with `--reward`. If you don't, training uses `race_line` when a race line is given and `basic`
otherwise. Every term is given per 20 Hz control step.

| Term | `race_line` preset | `basic` preset |
| --- | --- | --- |
| Progress | `+20 × metres gained along the race line` | `+20 × metres gained along the centre line` (track `none`: forward distance) |
| Lap | `+100` per lap of the race line | `+50` per lap |
| Collision | `-30`, and the episode ends | same |
| Race line | `-2 × distance from the race line (m)`, capped at `-2` | none |
| Steer rate | `-0.1 × change in the steering action` | none |
| Hard deceleration | `-0.05 × (deceleration - 7 m/s²)` above 7 m/s², not on a crash step | none |
| Speed | none | `+0.5 × forward speed (m/s)` |

**Hard deceleration:** the car has no brake. Lifting off the throttle slows it at 6–7 m/s², so the
penalty only applies to slowdowns harder than that.

**Speed on track `none`:** driving back and forth also earns the speed term there. The Oct 24 model
was trained with it anyway and still learned to complete laps.

## Episode rules

An episode ends on any of these:

- a collision
- 10 laps
- 3,000 steps (150 s)
- the end of a rollout, because the cars are held at the spawn during each policy update

Two more rules exist, but both are off by default:

| Rule | Ends the episode, with the collision penalty, when the car… | Parameter |
| --- | --- | --- |
| Stall | gains less than `stall_min_m` in `stall_window_s` | `stall_window_s` |
| Wrong way | falls `wrong_way_m` behind its best progress | `wrong_way_m` |

The stall rule was added after a model learned to stop and avoid crashing (run `ppo_20261106_013220`).

## Random starts

The simulator ignores pose commands, so the car can't be placed on the track. For a random start, a
scripted driver follows the centre line from the spawn to a random point, slowing for hairpins, and
the episode begins there.

This lowers training throughput from about 80 to about 10 steps per second. It is therefore off by
default (`--random-start 0`). Set `--random-start 0.25`, for example, to use it for a quarter of
episodes. Track `none` can't use it.
