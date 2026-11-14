"""Export frozen Stage 2 weights and an offline, simulator-independent runtime.

Run with the existing simulator Python. Export verification uses simulated
observations only and never connects to an actuator or an external API.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'exports' / 'stage2_20261107'

def main():
    from world import make_env, reset_scenario
    from track_geometry import TrackGeometry, PathTracker
    from moving_model import MovingModel
    from tiny_network import TinyNetwork
    freeze = json.loads((HERE/'artifacts/rollout_final_20261107/FINAL_TEST_FREEZE.json').read_text())
    for name, digest in freeze['runtime_source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest, name
    package = OUT/'stage2_runtime'
    assets = package/'assets'
    assets.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in ('tiny_network.py','maneuver_core.py','moving_model.py','perception.py','track_geometry.py'):
        raw = (HERE/name).read_text()
        provenance[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for module in ('bootstrap','perception','track_geometry','tiny_network','maneuver_core'):
            raw = raw.replace(f'from {module} import ', f'from .{module} import ')
        (package/name).write_text(raw)
    (package/'bootstrap.py').write_text('from pathlib import Path\nMODELS = Path(__file__).resolve().parent / "assets"\n')
    for name in ('stage2.npz','stage2.value.npz'):
        src = HERE/'artifacts/models'/name
        assert hashlib.sha256(src.read_bytes()).hexdigest() == freeze['models'][name]
        shutil.copy2(src, assets/name)
    env = make_env(False, seed=30001)
    g = TrackGeometry(env.unwrapped.track)
    angles = env.unwrapped.sim.agents[0].scan_angles.copy()
    fields = ('resolution','origin','occupancy','distance_field','length','s','xy','yaw','kappa','_left','_right')
    np.savez_compressed(assets/'track.npz', **{k:getattr(g,k) for k in fields}, angles=angles,
                        reference_digest=g.reference_digest)
    from exports.stage2_20261107.stage2_runtime import Stage2Controller, Stage2Networks
    model = TinyNetwork.load(assets/'stage2.npz')
    value = TinyNetwork.load(assets/'stage2.value.npz')
    schema = {'maneuver_architecture':[50,32,24,3], 'value_architecture':[54,32,24,2],
        'hidden_activation':'ReLU','output_activation':'softmax','normalization':'(x - checkpoint mean) / checkpoint scale',
        'classes':list(model.classes),'feature_names':list(model.feature_names),
        'value_classes':list(value.classes),'value_feature_names':list(value.feature_names),
        'reference_digest':g.reference_digest,'lidar_beams':len(angles),
        'lidar_angle_min_rad':float(angles[0]),'lidar_angle_max_rad':float(angles[-1]),
        'source_sha256':provenance,'models_sha256':{n:freeze['models'][n] for n in ('stage2.npz','stage2.value.npz')}}
    (OUT/'MODEL_SPEC.json').write_text(json.dumps(schema,indent=2)+'\n')
    networks = Stage2Networks()
    observations=[]; commands=[]; features=[]; probabilities=[]
    max_error=0.
    for start in (2.,22.,42.):
        obs,_ = reset_scenario(env,g,start,2.,1.7,ego_speed=1.7)
        original = MovingModel(g,angles)
        exported = Stage2Controller()
        opponent = PathTracker(g)
        for tick in range(40):
            own=obs['agent_0']
            action=np.asarray(original.plan(own,dt=.05))
            exported_action=np.asarray(exported.step(own,dt=.05))
            max_error=max(max_error,float(np.max(np.abs(action-exported_action))))
            np.testing.assert_allclose(exported_action,action,rtol=0,atol=1e-10)
            x=original.telemetry['features']
            result=networks.predict(x)
            np.testing.assert_array_equal(result['maneuver_probabilities'],model.proba(x))
            value_x=np.column_stack([np.repeat(x[None,:],3,axis=0),np.eye(4,dtype=np.float32)[:3]])
            np.testing.assert_array_equal(result['pass_probabilities'],value.predict_proba(value_x)[:,1])
            observations.append([float(own[k]) for k in ('pose_x','pose_y','pose_theta','linear_vel_x')])
            commands.append(action);features.append(x);probabilities.append(result['maneuver_probabilities'])
            if start==2. and tick==0:
                np.savez_compressed(OUT/'example_observation.npz',scan=own['scan'],
                    **{k:own[k] for k in ('pose_x','pose_y','pose_theta','linear_vel_x')},
                    expected_command=action,features=x,expected_probabilities=model.proba(x))
            other=opponent.control(obs['agent_1'],vmax=1.7,dt=.05)
            for _ in range(5):
                obs,*_=env.step(np.asarray([action,other]))
    env.close()
    for bad in (np.zeros(49),np.full(50,np.nan),np.zeros((1,50))):
        try: networks.predict(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid feature input accepted')
    np.savez_compressed(OUT/'parity_vectors.npz',features=features,probabilities=probabilities,commands=commands)
    report={'checks_passed':True,'controller_comparisons':len(commands),'starting_locations':3,
        'maximum_command_difference':max_error,'network_outputs':'exact match for both heads',
        'invalid_input_checks':3,'api_calls':0,'hardware_tests':0,
        'note':'Export parity checks, not new overtaking success tests.'}
    (OUT/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'requirements.txt').write_text('numpy==2.5.4\nscipy==1.18.1\n')
    checksums={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS.json'}
    (OUT/'SHA256SUMS.json').write_text(json.dumps(checksums,indent=2)+'\n')
    # Exclude caches from the deliverable.
    import zipfile
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                z.write(p,Path(OUT.name)/p.relative_to(OUT))
    print(json.dumps({'output':str(OUT.with_suffix('.zip')),'verification':report},indent=2))

if __name__=='__main__': main()
