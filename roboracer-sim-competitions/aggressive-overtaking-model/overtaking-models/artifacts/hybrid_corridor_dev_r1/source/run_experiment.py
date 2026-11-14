"""Train/evaluate drivers on the existing AutoDRIVE map and F1TENTH physics."""
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import json
import hashlib
from pathlib import Path
import time
import numpy as np

from bootstrap import ARTIFACTS, DT
from world import make_env, reset_scenario, body_polygon, restore_base_map, make_stage2_opponent
from track_geometry import TrackGeometry, PathTracker
from maneuver_core import delta_s
from static_model import StaticModel
from moving_model import MovingModel
from interactive_model import InteractiveModel

MODELS = {1: StaticModel, 2: MovingModel, 3: InteractiveModel}
# Capture loaded source provenance once. Edits during a long-running sweep
# must not relabel already-imported code as the newer version on disk.
LOADED_SOURCE_SHA256 = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("maneuver_core.py", "perception.py", "track_geometry.py", "world.py", "run_experiment.py")}


@dataclass
class Scenario:
    stage: int = 1
    seed: int = 0
    s: float = 0.
    gap: float = 3.0
    lateral: float = 0.
    ego_speed: float = 2.4
    opponent_speed: float = 0.
    seconds: float = 18.
    blocked: bool = False
    opponent: str = "line"
    warmup_seconds: float = 0.


def latency(values):
    if not values:
        return {}
    a = np.asarray(values)
    return {"count":len(a), "median_ms":float(np.median(a)), "p95_ms":float(np.percentile(a,95)),
            "p99_ms":float(np.percentile(a,99)), "max_ms":float(a.max())}


def stage3_setup_speed(geometry, positions, maximum_speed, preview_distance=1.2):
    """Common physical setup speed suitable for both cars and their preview.

    This coordination exists only during the setup prelude; racing controllers
    receive only their own actual observation histories after the handoff.
    """
    samples=np.asarray(positions,dtype=float)[:,None]+np.linspace(0.,preview_distance,13)
    return float(min(maximum_speed,np.min(geometry.speed_at(samples,maximum_speed,2.,2.,3.))))


def episode(scenario, *, teacher=False, model_path=None, render=False, collect=False, env=None, geometry=None,
            control_dt=.01, policy_bias=None):
    decision_ticks = round(float(control_dt) / DT)
    if decision_ticks < 1 or decision_ticks > 10 or abs(decision_ticks * DT - control_dt) > 1e-8:
        raise ValueError("control_dt must be a 0.01-second multiple in [0.01, 0.10]")
    if policy_bias is not None and not teacher:
        raise ValueError("Exploration bias is training-teacher only")
    warmup_ticks = round(scenario.warmup_seconds / DT)
    if (warmup_ticks < 0 or warmup_ticks * DT >= scenario.seconds
            or abs(warmup_ticks * DT - scenario.warmup_seconds) > 1e-8
            or (warmup_ticks and scenario.stage == 1)):
        raise ValueError("Moving-car warmup must be a nonnegative physics-tick multiple below the episode deadline")
    owns = env is None
    env = make_env(render=render, seed=scenario.seed) if env is None else env
    g = TrackGeometry(env.unwrapped.track) if geometry is None else geometry
    u = env.unwrapped
    restore_base_map(env)
    # Gym's reset(seed=...) does not reseed each car's private LiDAR RNG.
    # Set these explicitly so a shared-env sweep reproduces standalone trials.
    sensor_seeds=[int(scenario.seed)+i for i in (0,1)]
    for agent,sensor_seed in zip(u.sim.agents,sensor_seeds):
        agent.seed=sensor_seed
    setup_ego_speed=min(scenario.ego_speed,scenario.opponent_speed) if scenario.stage>1 else 0.
    setup_opponent_speed=scenario.opponent_speed
    coordinated_warmup=bool(scenario.stage==3 and warmup_ticks)
    warmup_speed_commands=[]
    warmup_command_history=[]
    if coordinated_warmup:
        setup_ego_speed=setup_opponent_speed=stage3_setup_speed(
            g,[scenario.s,scenario.s+scenario.gap],min(scenario.ego_speed,scenario.opponent_speed))
    obs, meta = reset_scenario(env, g, scenario.s, scenario.gap, setup_opponent_speed,
                              ego_speed=setup_ego_speed,
                              lateral=scenario.lateral, blocked=scenario.blocked)
    warm_observations = []
    warm_minimum_gap, warm_walls, warm_collision = 100., [100.,100.], None
    warm_trackers = [PathTracker(g), PathTracker(g)]
    # A flying encounter can carry actual pre-encounter sensor history. Both
    # cars physically drive the reference during setup; no opponent truth is
    # injected into either motion tracker. Setup time and contacts still count.
    for warm_tick in range(warmup_ticks):
        if warm_tick % decision_ticks == 0:
            warm_observations.append([{k:(v.copy() if isinstance(v,np.ndarray) else v)
                                       for k,v in obs[f"agent_{i}"].items()} for i in (0,1)])
            warm_commands = [warm_trackers[i].control(obs[f"agent_{i}"],
                vmax=min(scenario.ego_speed,scenario.opponent_speed),dt=control_dt) for i in (0,1)]
            if coordinated_warmup:
                positions=[float(g.project(float(obs[f"agent_{i}"]["pose_x"]),
                    float(obs[f"agent_{i}"]["pose_y"]))[0]) for i in (0,1)]
                shared_cap=stage3_setup_speed(g,positions,min(scenario.ego_speed,scenario.opponent_speed))
                shared_speed=min(shared_cap,warm_commands[0][1],warm_commands[1][1])
                warm_commands=[(command[0],shared_speed) for command in warm_commands]
                warmup_speed_commands.append(shared_speed)
                warmup_command_history.append({"t":warm_tick*DT,"speed_cap":shared_cap,
                    "commands":[list(map(float,command)) for command in warm_commands]})
        obs,*_ = env.step(np.asarray(warm_commands))
        warm_bodies = [body_polygon(obs[f"agent_{i}"]) for i in (0,1)]
        warm_minimum_gap = min(warm_minimum_gap,warm_bodies[0].distance(warm_bodies[1]))
        for i in (0,1):
            verts=np.asarray(warm_bodies[i].exterior.coords)[:-1]
            perimeter=np.vstack([verts+q*(np.roll(verts,-1,axis=0)-verts) for q in (0,.25,.5,.75)])
            warm_walls[i]=min(warm_walls[i],float(g.clearance(perimeter).min()))
            if warm_collision is None and (obs[f"agent_{i}"]["collision"] or warm_walls[i]<.001 or warm_minimum_gap<.001):
                warm_collision=dict(agent=i,time=(warm_tick+1)*DT,type="warmup_contact")
        if warm_collision:
            break
    completed_warmup_ticks = warm_tick+1 if warmup_ticks else 0
    warmup_elapsed = completed_warmup_ticks * DT
    warmup_entry_speeds=[float(obs[f"agent_{i}"]["linear_vel_x"]) for i in (0,1)]
    driver = MODELS[scenario.stage](g, u.sim.agents[0].scan_angles, vmax=scenario.ego_speed,
                                  teacher=teacher, model_path=model_path)
    driver.command_speed = max(0., float(obs["agent_0"]["linear_vel_x"]))
    driver.steering = float(u.sim.agents[0].state[2])
    for observations in warm_observations:
        driver.sensor.update(observations[0],dt=control_dt)
    if policy_bias is not None:
        bias = np.asarray(policy_bias, float)
        if bias.shape not in ((3,), (4,)) or not np.isfinite(bias).all():
            raise ValueError("policy_bias needs 3 or 4 finite action offsets")
        driver.teacher_bias = bias
    opponent = PathTracker(g)
    if scenario.opponent == "reactive":
        angles = u.sim.agents[1].scan_angles
        opponent = make_stage2_opponent(g, angles, scenario.opponent_speed, obs["agent_1"],
                                         steering=u.sim.agents[1].state[2])
        for observations in warm_observations:
            opponent.sensor.update(observations[1],dt=control_dt)
    else:
        opponent.last_steer = float(u.sim.agents[1].state[2])
    histories, samples, scans = [], [], []
    counters = Counter()
    minimum_gap, wall_clearance = warm_minimum_gap, warm_walls[0]
    body_wall_clearances = warm_walls.copy()
    collision, passed_at, success = warm_collision, None, False
    previous = [g.project(float(obs[f"agent_{i}"]["pose_x"]),float(obs[f"agent_{i}"]["pose_y"]))[0] for i in (0,1)]
    actual_start_gap = float(delta_s(previous[0],previous[1],g.length)) if warmup_ticks else scenario.gap
    progress = [0., actual_start_gap]
    peak, speeds = [0.,0.], [[],[]]
    lead_streak = 0.
    begin = time.monotonic()
    try:
        # A pass must be confirmed before the deadline. Allow only its two
        # seconds of follow-through to extend beyond that opportunity window.
        for tick in range(round((scenario.seconds + 2.02 - warmup_elapsed)/DT)):
            t = warmup_elapsed + (tick+1)*DT
            if render:
                import pygame
                if pygame.event.get(pygame.QUIT):
                    break
            if tick % decision_ticks == 0:
                action = driver.plan(obs["agent_0"], dt=control_dt)
                if scenario.opponent_speed <= 0 or scenario.blocked:
                    other = (0.,0.)
                elif scenario.opponent == "reactive":
                    other = opponent.plan(obs["agent_1"], dt=control_dt)
                else:
                    other = opponent.control(obs["agent_1"], vmax=scenario.opponent_speed, dt=control_dt)
            if collect and tick % max(10, decision_ticks) == 0:
                te = driver.telemetry
                samples.append({"features":te["features"].tolist(), "candidates":te["candidates"],
                                "action":te["action"], "probabilities":te["probabilities"],
                                "stage":scenario.stage, "seed":scenario.seed, "t":t,
                                "ego_s":float(g.project(float(obs["agent_0"]["pose_x"]),float(obs["agent_0"]["pose_y"]))[0]),
                                "max_speed":scenario.ego_speed, "state":te["target"],
                                "safe_probabilities":te.get("safe_probabilities"),
                                "maneuver_success_probabilities":te.get("action_success_probabilities"),
                                "padding_recovery_certified":te.get("padding_recovery_certified",False),
                                "padding_released_beams":te.get("padding_released_beams",0),
                                "scenario":asdict(scenario)})
            if tick % 20 == 0:
                scans.append({"scan":np.asarray(obs["agent_0"]["scan"]).tolist(),
                              "speed":float(obs["agent_0"]["linear_vel_x"])})
            previous_velocity=float(obs["agent_0"]["linear_vel_x"])
            obs, *_ = env.step(np.array([action, other]))
            acceleration_estimate=(float(obs["agent_0"]["linear_vel_x"])-previous_velocity)/DT
            bodies = [body_polygon(obs[f"agent_{i}"]) for i in (0,1)]
            blocker = meta.get("obstacle_polygon")
            body_gap = bodies[0].distance(bodies[1]) if blocker is None else bodies[0].distance(blocker)
            minimum_gap = min(minimum_gap, body_gap)
            for i in (0,1):
                o = obs[f"agent_{i}"]
                s, d = g.project(float(o["pose_x"]),float(o["pose_y"]))
                progress[i] += float(delta_s(previous[i],s,g.length))
                previous[i] = s
                speed = float(o["linear_vel_x"])
                speeds[i].append(speed); peak[i] = max(peak[i],speed)
                verts = np.asarray(bodies[i].exterior.coords)[:-1]
                perimeter = np.vstack([verts+q*(np.roll(verts,-1,axis=0)-verts) for q in (0,.25,.5,.75)])
                body_wall_clearances[i] = min(body_wall_clearances[i], float(g.clearance(perimeter).min()))
                if i == 0:
                    wall_clearance = body_wall_clearances[0]
                    ego_offset = abs(d)
                if o["collision"] and collision is None:
                    collision = {"agent":i,"time":t,"type":"gym_collision"}
            if body_gap < .001 and collision is None:
                collision = {"agent":0,"time":t,"type":"body_contact"}
            if min(body_wall_clearances) < .001 and collision is None:
                collision = {"agent":int(np.argmin(body_wall_clearances)),"time":t,"type":"wall_contact"}
            relative = progress[0]-progress[1]
            if scenario.blocked:
                relative = progress[0]-scenario.gap
            if relative > 1.0 and ego_offset < .28 and not collision:
                lead_streak += DT
                if lead_streak >= 1.0 and (t <= scenario.seconds or success):
                    success = True
                    passed_at = t if passed_at is None else passed_at
            else:
                lead_streak = 0.
            if success and relative < 0:
                success, passed_at = False, None
            counters[driver.action] += 1
            counters["mask_override"] += int(driver.telemetry["safety_mask_override"])
            counters["commit_override"] += int(driver.telemetry["commitment_override"])
            if tick%5 == 0:
                histories.append({"t":round(t,3), "ego":[float(obs["agent_0"][k]) for k in ("pose_x","pose_y","pose_theta","linear_vel_x")],
                                  "opponent":[float(obs["agent_1"][k]) for k in ("pose_x","pose_y","pose_theta","linear_vel_x")],
                                  "action":driver.action,"probabilities":driver.telemetry["probabilities"],
                                  "maneuver_success_probabilities":driver.telemetry.get("action_success_probabilities"),
                                  "padding_recovery_certified":driver.telemetry.get("padding_recovery_certified",False),
                                  "padding_released_beams":driver.telemetry.get("padding_released_beams",0),
                                  "planned_speed":driver.telemetry.get("planned_speed"),
                                  "lidar_speed_cap":driver.telemetry.get("lidar_speed_cap"),
                                  "body_gap":float(body_gap),
                                  "pass_completion_recovery_eligible":driver.telemetry.get("pass_completion_recovery_eligible",False),
                                  "command":list(action), "relative_progress":float(relative),
                                  "opponent_command":list(other),
                                  "opponent_action":opponent.action if scenario.opponent == "reactive" else "LINE",
                                  "opponent_target":opponent.telemetry.get("target") if scenario.opponent == "reactive" else None,
                                  "acceleration":acceleration_estimate,
                                  "target":driver.telemetry["target"]})
            if render:
                if tick % 10 == 0:
                    pygame.display.set_caption(f"Stage {scenario.stage} | {driver.action} | {t:.1f}s | local inference")
                env.render()
            if collision or (success and t > passed_at + 2.) or (not success and t >= scenario.seconds):
                break
    finally:
        restore_base_map(env)
        if owns:
            env.close()
    safe_stop = bool(scenario.blocked and not collision and np.mean(speeds[0][-min(100,len(speeds[0])):])<.10)
    result = {"scenario":asdict(scenario),"teacher":teacher,"llm_calls_in_test":0,
              "sensor_seeds":sensor_seeds,
              "warmup_seconds":warmup_elapsed,"warmup_lidar_samples_per_car":len(warm_observations),
              "actual_gap_at_policy_start":actual_start_gap,
              "coordinated_stage3_warmup":coordinated_warmup,
              "warmup_initial_speed_m_s":[setup_ego_speed,setup_opponent_speed],
              "warmup_common_speed_commands_m_s":warmup_speed_commands,
              "warmup_command_history":warmup_command_history,
              "warmup_entry_speeds_m_s":warmup_entry_speeds,
              "warmup_minimum_body_gap_m":warm_minimum_gap,
              "warmup_preview_distance_m":1.2 if coordinated_warmup else None,
              "warmup_description":("Physical shared-reference setup at one common curvature-limited speed for both cars and their 1.2m previews; actual own LiDAR histories; setup time and contacts scored" if coordinated_warmup else "Physical shared-reference driving with actual own LiDAR history; time and any contacts included in episode scoring"),
              "acceleration_measurement":"finite difference of measured speed at 100Hz",
              "model_file":str(driver.model_path) if driver.model_path else None,
              "model_sha256":driver.model_sha256,
              "value_model_file":str(driver.value_model_path) if driver.value_model_path else None,
              "value_model_sha256":driver.value_model_sha256,
              "opponent_controller":"MovingModel(stage=2)" if scenario.opponent == "reactive" else "PathTracker",
              "opponent_model_file":str(opponent.model_path) if scenario.opponent == "reactive" else None,
              "opponent_model_sha256":opponent.model_sha256 if scenario.opponent == "reactive" else None,
              "opponent_value_model_file":str(opponent.value_model_path) if scenario.opponent == "reactive" and opponent.value_model_path else None,
              "opponent_value_model_sha256":opponent.value_model_sha256 if scenario.opponent == "reactive" else None,
              "opponent_uses_own_observation_only":True,
              "opponent_safety_margins":({"wall":opponent.planner.wall_margin,
                                           "car":opponent.planner.car_margin,
                                           "detour":opponent.planner.detour_clearance}
                                          if scenario.opponent == "reactive" else None),
              "opponent_planning_profile":(opponent.planner.planning_profile()
                                            if scenario.opponent == "reactive" else None),
              "opponent_padding_recovery_decisions":getattr(opponent,"padding_recovery_steps",0),
              "opponent_padding_lidar_release_decisions":getattr(opponent,"padding_lidar_release_steps",0),
              "source_sha256":dict(LOADED_SOURCE_SHA256),
              "physics_dt_seconds": DT, "control_dt_seconds":control_dt,
              "policy_bias":None if policy_bias is None else list(map(float,policy_bias)),
              "reference_description":g.reference_description,
              "reference_mode":getattr(g,"reference_mode","legacy"),
              "reference_optimization":getattr(g,"reference_optimization",{}),
              "reference_sha256":hashlib.sha256(g.xy.tobytes()).hexdigest(),
              "ego_planning_profile": {"wall_margin_m": driver.planner.wall_margin,
                                       "car_margin_m": driver.planner.car_margin,
                                       "detour_clearance_m": driver.planner.detour_clearance,
                                       "lateral_acceleration_m_s2": driver.planner.lateral_acceleration,
                                       "preview_distance_m": float(driver.planner.distances[-1]),
                                       **(driver.planner.planning_profile() if hasattr(driver.planner,"planning_profile") else {})},
              "padding_recovery_decisions":getattr(driver,"padding_recovery_steps",0),
              "padding_recovery_nominal_seconds":getattr(driver,"padding_recovery_seconds",0.),
              "padding_lidar_release_decisions":getattr(driver,"padding_lidar_release_steps",0),
              "padding_lidar_released_beams":getattr(driver,"padding_lidar_released_beams",0),
              "success":success and not collision,"safe_blocked_stop":safe_stop,
              "outcome":"collision" if collision else "pass" if success else "safe_stop" if safe_stop else "no_pass",
              "collision":collision,"passed_at":passed_at,"sim_seconds":t,
              "pass_opportunity_deadline_seconds":scenario.seconds,
              "pass_followthrough_complete":bool(success and passed_at is not None and t > passed_at + 2.),
              "wall_seconds":time.monotonic()-begin,"minimum_body_gap":minimum_gap,
              "minimum_wall_clearance":wall_clearance,"progress":progress,
              "minimum_body_wall_clearances":body_wall_clearances,
              "mean_speed":[float(np.mean(v)) for v in speeds],"peak_speed":peak,
              "action_seconds":{k:v*DT for k,v in counters.items() if k not in ("mask_override","commit_override")},
              "safety_mask_override_ticks":counters["mask_override"],
              "commitment_override_ticks":counters["commit_override"],
              "network_latency":latency(driver.network_times),"controller_latency":latency(driver.control_times),
              "perception":"ego LiDAR residual clustering and temporal tracking; own simulator localization",
              "control_architecture":"neural maneuver probabilities + geometric trajectory tracking + safety filter"}
    return result, histories, samples, scans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",type=int,choices=[1,2,3],default=1)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--position",type=float,default=0.)
    p.add_argument("--gap",type=float,default=3.)
    p.add_argument("--lateral",type=float,default=0.)
    p.add_argument("--speed",type=float,default=2.4)
    p.add_argument("--opponent-speed",type=float)
    p.add_argument("--seconds",type=float,default=20.)
    p.add_argument("--teacher",action="store_true")
    p.add_argument("--render",action="store_true")
    p.add_argument("--control-dt",type=float,default=.01)
    p.add_argument("--warmup-seconds",type=float,default=0.)
    p.add_argument("--blocked",action="store_true")
    p.add_argument("--line-opponent",action="store_true")
    p.add_argument("--model")
    p.add_argument("--output",default="latest")
    args=p.parse_args()
    ospeed=args.opponent_speed if args.opponent_speed is not None else (0. if args.stage==1 else 2. if args.stage==2 else args.speed)
    scenario=Scenario(args.stage,args.seed,args.position,args.gap,args.lateral,args.speed,ospeed,args.seconds,args.blocked,
                      "reactive" if args.stage==3 and not args.line_opponent else "line",args.warmup_seconds)
    result,history,samples,scans=episode(scenario,teacher=args.teacher,model_path=args.model,render=args.render,collect=True,control_dt=args.control_dt)
    out=ARTIFACTS/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    out.with_suffix(".trace.json").write_text(json.dumps({"result":result,"frames":history})+"\n")
    out.with_suffix(".samples.json").write_text(json.dumps(samples)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
