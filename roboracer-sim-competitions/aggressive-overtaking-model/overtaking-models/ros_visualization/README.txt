F1TENTH GYM + ROS 2 + RVIZ: STAGES 2 AND 4

START
Double-click “Run Gym with RViz.command” in the Aichen - RoboRacer workspace.
It opens the existing native Gym window and the actual ROS RViz application.
Both stages alternate until either window is closed. Playback is half speed;
the Gym physics timestep stays 0.01 s and policy decisions stay 0.05 s apart.

The launcher also accepts --stage 2, --stage 4, --speed 1, and --once.
Run from overtaking-models:
  ../gym-two-car-reactive/.venv/bin/python ros_visualization/launch.py --stage 4

WHAT YOU SEE
White: the existing locally optimized reference line.
Gray boundaries: occupied/free boundaries from Gym's actual track map.
Green car: our ego car. Orange car: the opponent.
Red paths: candidates rejected by a clearance, curvature, or maneuver gate.
Blue paths: feasible candidates.
Thick green path: the final selected representative, even when its speed
command is reduced by LiDAR or the fallback decision is to stop.
The green path shows intended geometry, not a promise that the car will
execute its entire length before replanning.

The decision text shows candidate counts, maneuver probabilities, per-maneuver
cost and clearance, the final choice, safety/commitment overrides, and actual
steering/speed commands. If an expanded path is selected, its entry distance,
padding, and merge factor are shown too. Toggle Ego LiDAR in RViz to show scans.
Use RViz's normal mouse camera controls to zoom in on an encounter.

HOW TO READ THE DECISIONS
The planner constructs smooth offset curves around the reference rather than
just sampling fixed steering angles. It tries different entry distances,
lateral clearances, and return-to-line distances when a nominal maneuver is
blocked or stalled. Up to 4 nominal + 28 additional paths can be evaluated.
It checks walls, steering curvature, and predicted vehicle clearance.

It retains the lowest-cost feasible path per side (with a small continuity
penalty), preserving the trained network's existing feature interface.
The network and learned success-value model rank maneuver choices. Feasibility
masks and commitment rules can override the raw network preference. Finally,
LiDAR braking and steering/acceleration limits affect the actual command.
“Selected” therefore means best under this pipeline, not globally optimal.
Nominal rejection labels are deliberately coarse; expanded paths report the
specific first failed check. Candidate costs include heuristic penalties,
so they are not exact lap-time predictions.

STAGES
Stage 2: saved Stage 2 ego network + expanded planner, faster than a reference-
following opponent. The opponent has no reactive overtaking behavior.
Stage 4: saved aggressive Stage 3 ego network + expanded planner, slightly
faster than the opponent. The opponent runs the original reactive Stage 2
model using its own LiDAR. No scripted yielding or forced displacement.
Both cars use the existing reference line when no maneuver is needed.

The demonstration scenarios are selected successful regression cases:
Stage 2: stage2-test-000, confirmed pass at 11.43 simulated seconds.
Stage 4: start 49 m / opponent cap 2.1 m/s, confirmed pass at 7.85 seconds.
These demonstrations are not new success-rate measurements or new training.
Both complete the existing two-second follow-through without contact.

ARCHITECTURE
One F1TENTH Gym instance owns all physics and observations. gym_source.py runs
the existing experiment harness and records the planner's actual candidates.
Atomic local JSON snapshots carry observation-only data to ros_display.py.
ROS publishes /map, /gym/walls, /clock, /ego/odom, /opponent/odom, /ego/scan,
/planner/reference, /planner/selected_path, /planner/candidates, /planner/status,
and the vehicle transforms. No ROS command is fed back to a second simulator.
The simulation remains in the existing Python environment; ROS/RViz has its
own isolated Pixi/RoboStack installation here to avoid dependency conflicts.
ROS discovery is local to this computer on domain 47.

The Gym policy remains inside the existing offline network guard. It uses no
LLM calls, API credits, or retraining. Ground-truth car poses are used only for
visualization and scoring; the controller still receives its own observation.
A monotonic ROS clock spans episodes; the label shows local episode time.
Visual recording adds overhead, so this display is not a latency benchmark.

MACOS DISPLAY NOTE
RViz's textured occupancy-grid plugin produced a GLSL palette error on this
Mac. The default display therefore uses 5,955 wall-boundary cubes taken from
the exact occupancy grid. /map is still published at its full 0.02 m resolution.
The original Map display is present but disabled. This is a rendering fallback
inside actual RViz, with the same Gym geometry and no alternate simulator.

VERIFICATION
- Five existing expanded-planner tests pass.
- Capture parity test compares the instrumented controller against the frozen
  pre-visualization source: identical features, choices, and commands on sampled
  observations for both stages.
- Both representative closed-loop runs reproduce the existing pass times and
  finish with no contact.
- validate_ros.py checks both stages, expanded candidate sets, map/wall/selected
  path messages, monotonic ROS clock, and an actual RViz topic subscription.
  Its result is saved in verification.json.
- live/results.json keeps the latest 20 demo runs; ros.log and rviz.log hold
  the native process logs. These are runtime outputs, not training data.

FILES
expanded_planner.py (parent folder): optional candidate capture, off by default.
gym_source.py: actual Gym execution and snapshot bridge.
ros_display.py: ROS publishers and RViz markers.
planner.rviz: ready-to-use RViz display configuration.
launch.py: one-command process startup and cleanup.
pixi.toml / pixi.lock: ROS dependencies and exact resolved package versions.
test_capture.py / validate_ros.py: control parity and live-topic validation.

INSTALLATION REFERENCE
https://robostack.github.io/GettingStarted.html
ROS MarkerArray source used to verify display topic configuration:
https://github.com/ros2/rviz/blob/jazzy/rviz_default_plugins/src/rviz_default_plugins/displays/marker_array/marker_array_display.cpp


LEARNED REACTIVE OPPONENT EXPERIMENT
The workspace launcher "Run Stage 4 Response Experiment.command" opens actual
F1TENTH Gym and RViz. New experiments only race the reactive Stage 2 opponent.
Nonreactive/fixed-command counterfactual demos are now disabled.

GREEN: chosen ego path. BLUE: feasible candidates. RED: rejected candidates.
CYAN: learned opponent forecast when that forecast is used for a maneuver.
ORANGE: actual simulated opponent. The text shows its measured displacement
from the shared racing line. Displacement alone does not prove yielding; use
the corrected yield_audit.json and the full experiment report for success rates.

The display loads frozen planner code and forecast weights from the selected
experiment. It refuses changed shared runtime dependencies. The display receives
state from the real Gym instance, not a separate physics engine or HTML replay.
