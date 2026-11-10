# AutoDRIVE RoboRacer PPO

Trains a PPO driving policy in the real **AutoDRIVE Simulator** (RoboRacer, ICRA 2026 compete build).
The simulators run on Windows. The trainer runs in a Docker container that uses the GPU. Each
simulator connects to its own port on the container.


## Three Models

### 1. Original model
**Model:** `ppo_20261105_135745`
**Trained on:** ICRA track, 1,798,841 steps
**Tested on:** ICRA and Bridge

**Result:** 0 laps on ICRA, with crashes at the 33 m and 39 m hairpins. On Bridge, a track it never saw in training, it completes smooth laps at a fast pace.

**Reward**:
| Term | Value |
| --- | --- |
| Progress | `+10 × metres along the centre line` |
| Lap | `+50` per lap |
| Collision | `-10`, and the episode ends |
| Centre line | Penalty for distance from the centre line, weight `0.05` |
| Steer rate | `-0.1 × change in steering` |
| Speed | weight `0` |

---

### 2. Fine-tuned Original model based on Bridge track
**Model:** `ppo_20261107_182816` continued from model 1 and fine-tuned on Bridge
**Steps:** 2,335,417 in total, about 537k of them on Bridge

**Result:** 7–8 laps per episode on Bridge, best lap 10.8 s, average lap 11.3 s.

**Reward**:
| Term | Value |
| --- | --- |
| Progress | `+20 × metres driven forward` |
| Lap | `+50` per lap, from the simulator's lap counter |
| Collision | `-30`, and the episode ends |
| Speed | `+0.5 × forward speed (m/s)` |

---

### 3. Race-line model
**Model:** `ppo_20261108_054349`
**Trained on:** ICRA track, from scratch, following the minimum-curvature race line (`track3_minimum_curvature_world.csv`)
**Steps:** 1,039,495

**Result:** 0 laps. The car crashes at the same 33m hairpins like model 1. 

**Reward**:
| Term | Value |
| --- | --- |
| Progress | `+20 × metres along the race line` |
| Lap | `+100` per lap |
| Collision | `-30`, and the episode ends |
| Race line | `-2 × distance from the race line (m)`, capped at `-2` |
| Steer rate | `-0.1 × change in steering` |
| Hard deceleration | `-0.05 × (deceleration − 7 m/s²)` above 7 m/s² |
