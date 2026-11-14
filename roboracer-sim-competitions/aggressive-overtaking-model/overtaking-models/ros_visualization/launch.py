"""Launch ROS publisher, actual RViz, and a single live F1TENTH Gym."""
import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['2','4','all'],default='all')
    p.add_argument('--speed',type=float,default=.5)
    p.add_argument('--once',action='store_true')
    p.add_argument('--show-counterfactual',action='store_true')
    p.add_argument('--response-experiment',type=Path)
    args=p.parse_args()
    if args.show_counterfactual:p.error('Nonreactive opponent demos are disabled.')
    pixi=shutil.which('pixi') or str(HERE/'bin/pixi')
    if not Path(pixi).exists():raise RuntimeError('Install Pixi and put it on PATH before launching RViz.')
    gym_python=ROOT.parent/'gym-two-car-reactive/.venv/bin/python'
    live=HERE/'live';live.mkdir(exist_ok=True)
    # Stale snapshots must never appear as a new live run.
    (live/'frame.json').unlink(missing_ok=True)
    env=os.environ.copy()
    env.update(ROS_DOMAIN_ID='47',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
    # Prevent the user's unrelated Python/Qt environments from shadowing ROS.
    for key in ('PYTHONPATH','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','ROS_LOCALHOST_ONLY'):
        env.pop(key,None)
    children=[];logs=[]
    try:
        for name,command in [
            ('ros',[str(pixi),'run','python','ros_display.py']),
            ('rviz',[str(pixi),'run','rviz2','-d','planner.rviz','--ros-args','-p','use_sim_time:=true']),
        ]:
            log=(live/(name+'.log')).open('w');logs.append(log)
            children.append(subprocess.Popen(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT))
        time.sleep(3.)
        for proc,name in zip(children,('ROS bridge','RViz')):
            if proc.poll() is not None:raise RuntimeError(f'{name} exited; see {live}')
        command=[str(gym_python),str(HERE/'gym_source.py'),'--stage',args.stage,'--speed',str(args.speed)]
        if args.once:command.append('--once')
        if args.response_experiment:command.extend(['--response-experiment',str(args.response_experiment.resolve())])
        gym=subprocess.Popen(command,cwd=ROOT,env=env);children.append(gym)
        print('Gym drives both cars; RViz displays the live ROS topics. Close either window to finish.',flush=True)
        while all(proc.poll() is None for proc in children):time.sleep(.5)
        for proc,name in zip(children,('ROS bridge','RViz','Gym')):
            if proc.poll() not in (None,0):raise RuntimeError(f'{name} exited with {proc.returncode}; see {live}')
    except KeyboardInterrupt:pass
    finally:
        for proc in reversed(children):
            if proc.poll() is None:proc.send_signal(signal.SIGINT)
        for proc in reversed(children):
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.terminate()
        for log in logs:log.close()

if __name__=='__main__':main()
