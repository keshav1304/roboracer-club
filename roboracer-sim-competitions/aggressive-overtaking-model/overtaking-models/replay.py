"""Build an offline, interactive replay from actual saved simulator traces.

Usage:
  ../gym-two-car-reactive/.venv/bin/python replay.py artifacts/*.trace.json \
      --output artifacts/replay.html

The HTML contains the occupancy image, reference, recorded poses, measured
latencies and optional training histories. It performs no vehicle simulation,
requires no server or external scripts, and makes no network requests.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bootstrap import ARTIFACTS, MODELS
from world import make_env
from track_geometry import TrackGeometry


def load_replay_data(paths, history_paths=()):
    traces = []
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text())
        if not isinstance(data.get("frames"), list) or not data["frames"]:
            raise ValueError(f"{path.name} does not contain nonempty recorded frames")
        if "scenario" not in data.get("result", {}):
            raise ValueError(f"{path.name} does not contain a scenario result")
        data["name"] = path.name.removesuffix(".trace.json").replace("_", " ")
        data["source_file"] = path.name
        model_name = Path(data["result"].get("model_file") or "").name
        if data["result"].get("teacher"):
            checkpoint = "teacher"
        elif ".early." in model_name or "early" in path.stem.lower():
            checkpoint = "early"
        elif ".mid." in model_name or "_mid" in path.stem.lower():
            checkpoint = "mid"
        elif model_name or "final" in path.stem.lower():
            checkpoint = "final"
        else:
            checkpoint = "unrecorded"
        data["checkpoint"] = checkpoint
        traces.append(data)
    env = make_env()
    try:
        geometry = TrackGeometry(env.unwrapped.track)
        image = Image.fromarray(np.flipud(np.uint8(geometry.occupancy))).convert("L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        map_data = {"image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
                    "origin": geometry.origin.tolist(), "resolution": geometry.resolution,
                    "width": image.width, "height": image.height,
                    "line": geometry.xy[::3].round(5).tolist(), "length": geometry.length,
                    "description": geometry.reference_description}
        for trace in traces:
            scenario = trace["result"]["scenario"]
            if scenario.get("blocked"):
                s = scenario["s"] + scenario["gap"]
                xy, yaw, _ = geometry.sample(s)
                left, right = geometry.widths(s)
                # This is the same static geometry formula used by world.py.
                trace["barrier"] = [*xy.tolist(), float(yaw), .58,
                                    2 * max(float(left), float(right)) + .2]
    finally:
        env.close()
    histories = []
    for path in history_paths:
        path = Path(path)
        report = json.loads(path.read_text())
        if report.get("history"):
            entry = {"name": path.name.removesuffix(".history.json"), "report": report}
            stage = report.get("metadata", {}).get("stage")
            training_path = ARTIFACTS / f"stage{stage}.training.json"
            if training_path.exists():
                training = json.loads(training_path.read_text()).get("dataset", {})
                if training.get("sources") == report.get("metadata", {}).get("label_sources"):
                    entry["dataset"] = {key: training.get(key) for key in
                                        ("rows", "sources", "action_counts", "split_counts",
                                         "labels_are_rollout_ground_truth", "equal_initial_speed_count")}
            histories.append(entry)
    final_indices = [i for i, trace in enumerate(traces) if trace["checkpoint"] == "final"]
    initial_index = min(final_indices, key=lambda i: (traces[i]["result"]["scenario"]["stage"],
                        "demo" not in traces[i]["source_file"])) if final_indices else 0
    return {"map": map_data, "traces": traces, "histories": histories, "initial_index": initial_index}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboRacer · Overtaking experiment replay</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--card:#151c30;--edge:#2b354e;--text:#eef2fc;--muted:#a1adc7;--ego:#a78bfa;--opp:#fb7185;--green:#6ee7b7;--amber:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1440px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;margin-bottom:22px}h1{margin:3px 0 7px;font-size:27px;letter-spacing:-.5px}h2{font-size:18px;margin:0 0 14px}p{margin:5px 0;color:var(--muted)}.eyebrow{font-size:11px;letter-spacing:1.6px;color:#b5a0ff;font-weight:700}.pill{border:1px solid var(--edge);border-radius:20px;padding:6px 11px;font-size:12px;white-space:nowrap}.selectors{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}select,button{background:#202940;color:var(--text);border:1px solid #35415e;border-radius:8px;padding:9px 12px;font:inherit;cursor:pointer}select{flex:1;min-width:240px}button:hover{background:#2d3854}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #c4b5fd;outline-offset:3px}button.primary{background:#7951cf;border-color:#9871f0;min-width:90px}.grid{display:grid;grid-template-columns:minmax(0,1fr) 302px;gap:16px}.card{background:var(--card);border:1px solid var(--edge);border-radius:13px;overflow:hidden}.canvasWrap{position:relative;aspect-ratio:1.7;min-height:340px}#track{width:100%;height:100%;display:block}.canvasLabel{position:absolute;top:14px;left:16px;background:#111827df;border:1px solid #56617a;padding:6px 10px;border-radius:7px;font-size:12px;pointer-events:none}.legend{display:flex;gap:20px;padding:0 18px 13px;flex-wrap:wrap;color:var(--muted);font-size:12px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px}.controls{border-top:1px solid var(--edge);padding:14px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.controls input[type=range]{flex:1;min-width:110px;accent-color:#a78bfa}.time{font-variant-numeric:tabular-nums;font-size:13px;min-width:100px}.options{display:flex;align-items:center;gap:18px;padding:0 16px 14px;color:var(--muted);font-size:12px;flex-wrap:wrap}.options label{cursor:pointer}.options input{accent-color:#a78bfa}.aside{padding:19px}.status{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;padding:7px 10px;border-radius:7px;margin-bottom:13px;background:#263246}.status.pass{color:#6ee7b7;background:#14372e}.status.collision{color:#fda4af;background:#471e2a}.status.no_pass{color:#fcd34d;background:#45371c}.status.safe_stop{color:#7dd3fc;background:#183547}.metricrow{display:flex;justify-content:space-between;margin:10px 0;gap:12px}.metricrow span{color:var(--muted);font-size:12px}.metricrow strong{font-size:14px;font-variant-numeric:tabular-nums}.divider{height:1px;background:var(--edge);margin:17px 0}.action{font-size:22px;font-weight:700;margin:0 0 10px}.prob{margin:11px 0}.probhead{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px}.bar{height:7px;background:#2b344b;border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;border-radius:5px;transition:width .1s}.small{font-size:12px;color:var(--muted)}.note{font-size:12px;color:var(--muted);line-height:1.55;padding:14px 17px;border-top:1px solid var(--edge)}.bottom{margin-top:18px;padding:20px}.learningGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:22px}.learning canvas{width:100%;height:180px;display:block}.learning h3{font-size:15px;margin:0 0 4px}.learning p{font-size:12px}.historyLegend{font-size:11px;color:var(--muted);display:flex;gap:12px;margin-top:4px}.hidden{display:none}.provenance{margin-top:18px;color:var(--muted);font-size:12px}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:15px}.summaryItem{background:#10172a;padding:12px;border-radius:8px}.summaryItem strong{display:block;font-size:20px}.summaryItem span{font-size:11px;color:var(--muted)}@media(max-width:900px){.grid{grid-template-columns:1fr}.aside{display:grid;grid-template-columns:1fr 1fr;gap:20px}.divider{display:none}.canvasWrap{min-height:270px}.summary{grid-template-columns:repeat(2,1fr)}}@media(max-width:550px){main{padding:13px}.aside{display:block}header{display:block}header .pill{display:inline-block;margin-top:10px}h1{font-size:24px}.controls{gap:7px}.canvasWrap{aspect-ratio:1.2}.options{gap:10px}}
.trainingBanner{border:1px solid #6c5427;background:#2c2418;color:#f5d9a5;padding:13px 16px;border-radius:10px;margin-bottom:17px;font-size:13px}.tableWrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;margin-top:13px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--edge)}th{color:var(--muted);font-weight:500}.stageWarning{color:#fcd34d;margin-top:12px}.sourceLine{font-size:11px;color:var(--muted);margin-top:9px}</style></head><body><main>
<header><div><div class="eyebrow">ROBORACER / LEARNED OVERTAKING</div><h1>Three stages. One track.</h1><p>Watch recorded simulation runs, inspect decisions, and compare training checkpoints.</p></div><span class="pill">Offline recorded replay · no LLM calls</span></header>
<div id="trainingBanner" class="trainingBanner"></div>
<div class="selectors"><label for="episode" class="small">Run / checkpoint</label><select id="episode" aria-label="Choose a recorded run"></select><button id="previous" title="Previous run">←</button><button id="next" title="Next run">→</button><button id="save" title="Save the current track image">Save image</button></div>
<div class="grid"><section class="card"><div class="canvasWrap"><canvas id="track" aria-label="Recorded car positions on the real imported track"></canvas><div class="canvasLabel" id="canvasLabel"></div></div><div class="legend"><span><i class="dot" style="background:var(--ego)"></i>Our car</span><span><i class="dot" style="background:var(--opp)"></i><span id="opponentLabel">Opponent</span></span><span><i class="dot" style="background:#6b7a96"></i>Reference racing line</span></div><div class="controls"><button class="primary" id="play">Play</button><button id="restart" title="Restart this replay">↺</button><input id="seek" type="range" min="0" max="100" value="0" step="1" aria-label="Replay time"><span class="time" id="time"></span><select id="rate" aria-label="Playback speed" style="flex:0;min-width:75px;padding:8px"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div><div class="options"><label><input id="follow" type="checkbox" checked> Follow the cars</label><label><input id="trail" type="checkbox" checked> Recorded paths</label><label><input id="line" type="checkbox" checked> Reference line</label><label><input id="autonext" type="checkbox"> Play next run automatically</label></div><div class="note">The cars and controls below come from saved F1TENTH simulation frames. Playback does not run a new race. The reference is a locally curvature-reduced approximation, not a certified fastest line.</div></section>
<aside class="card aside"><div><div id="status" class="status"></div><div id="runDescription" class="small"></div><div class="divider"></div><div class="small">Current decision</div><div class="action" id="action"></div><div id="probabilitySource" class="small"></div><div id="probabilities"></div><p class="small">Left means the obstacle is the right boundary; right means it is the left boundary. “No pass” can follow, stop, or cruise when the line is clear.</p></div><div><div class="divider"></div><div class="metricrow"><span>Car speed</span><strong id="speed"></strong></div><div class="metricrow"><span>Other speed</span><strong id="otherSpeed"></strong></div><div class="metricrow"><span>Steering command</span><strong id="steer"></strong></div><div class="metricrow"><span>Target speed</span><strong id="targetSpeed"></strong></div><div class="metricrow"><span>Lead (+) / behind (−)</span><strong id="lead"></strong></div><div class="metricrow"><span>Opponent decision</span><strong id="otherAction"></strong></div><div class="metricrow"><span>Opponent sees car</span><strong id="otherSees"></strong></div><div class="divider"></div><div class="small">Measured across this run</div><div class="metricrow"><span>Network p95</span><strong id="network"></strong></div><div class="metricrow"><span>Whole controller p95</span><strong id="controller"></strong></div><div class="metricrow"><span>Closest body gap</span><strong id="gap"></strong></div><p id="source" class="small"></p><p id="stageWarning" class="small stageWarning"></p></div></aside></div>
<section class="card bottom"><h2>Recorded run outcome</h2><p id="outcomeDescription"></p><div class="summary"><div class="summaryItem"><strong id="summaryTime"></strong><span>Simulated seconds</span></div><div class="summaryItem"><strong id="summaryPass"></strong><span>Pass verified at</span></div><div class="summaryItem"><strong id="summaryWall"></strong><span>Closest wall clearance</span></div><div class="summaryItem"><strong id="summaryCalls"></strong><span>LLM calls during test</span></div></div></section>
<section id="learningSection" class="card bottom hidden"><h2>Training evolution: label agreement</h2><p class="small" style="margin-bottom:18px">Curves show recorded training and validation loss. Lower loss means closer agreement with the teacher labels; successful driving is measured separately in the replays. Use the run selector to compare early, middle, and final checkpoints when included.</p><div id="learning" class="learningGrid"></div></section>
<section class="card bottom"><h2>Driving checks: actual recorded episodes</h2><p class="small">These outcomes come from closed-loop simulator runs, separately from classification accuracy. Each row is one episode; a selected demonstration is not an overall success-rate estimate.</p><div class="tableWrap"><table><thead><tr><th>Stage / run</th><th>Checkpoint</th><th>Outcome</th><th>Pass verified</th><th>Contact</th><th>Network p95</th></tr></thead><tbody id="drivingResults"></tbody></table></div></section>
<footer class="provenance">Imported AutoDRIVE default track · simulator vehicle body 0.58 × 0.31 m · positive travel direction. Reported success requires clearing the obstacle, returning near the reference, and holding a lead without contact. Neural maneuver probabilities feed geometric tracking and a safety filter; they are not certified real-world safety probabilities.</footer>
</main><script id="data" type="application/json">__DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const colors={LEFT:'#60a5fa',RIGHT:'#fb923c',WAIT:'#94a3b8',PROBE:'#fbbf24'}, labels={LEFT:'Pass left',RIGHT:'Pass right',WAIT:'Follow / hold',PROBE:'Inside probe'};
const stageNames={1:'Static obstacle',2:'Slower line-following car',3:'Equal-cap stage 2 neural rival'};
const canvas=$('track'), ctx=canvas.getContext('2d'), mapImage=new Image();
let runIndex=0,frameIndex=0,playing=false,lastTime=0,playTime=0,scale=1,tx=0,ty=0;
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';
function current(){return DATA.traces[runIndex]}
function toPx(x,y){return [tx+x*scale,ty-y*scale]}
function measure(canvas){const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);return [canvas.width,canvas.height]}
function line(points,color,width,dash=[]){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>{const q=toPx(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([])}
function car(pose,color,length=.58,width=.31){const p=toPx(pose[0],pose[1]);ctx.save();ctx.translate(...p);ctx.rotate(-pose[2]);ctx.fillStyle=color;ctx.strokeStyle='#f1f5f9';ctx.lineWidth=Math.max(1,scale*.012);ctx.beginPath();ctx.rect(-length*scale/2,-width*scale/2,length*scale,width*scale);ctx.fill();ctx.stroke();ctx.fillStyle='#192132';ctx.fillRect(length*scale*.03,-width*scale*.33,length*scale*.19,width*scale*.66);ctx.strokeStyle='#ffffff';ctx.beginPath();ctx.moveTo(length*scale*.30,0);ctx.lineTo(length*scale*.43,0);ctx.stroke();ctx.restore()}
function drawTrack(){const run=current(),f=run.frames[frameIndex],m=DATA.map;const [w,h]=measure(canvas);ctx.fillStyle='#101829';ctx.fillRect(0,0,w,h);let xmin=m.origin[0],ymin=m.origin[1],ww=m.width*m.resolution,hh=m.height*m.resolution;
if($('follow').checked){let partner=run.barrier?run.barrier:f.opponent;let dx=Math.abs(f.ego[0]-partner[0]),dy=Math.abs(f.ego[1]-partner[1]);let near=dx<10&&dy<7;ww=Math.max(8,near?dx+5:8);hh=Math.max(5,near?dy+4:5);let cx=near?(f.ego[0]+partner[0])/2:f.ego[0],cy=near?(f.ego[1]+partner[1])/2:f.ego[1];xmin=cx-ww/2;ymin=cy-hh/2}
scale=Math.min(w/ww,h/hh)*.92;tx=(w-ww*scale)/2-xmin*scale;ty=(h+hh*scale)/2+ymin*scale;let p=toPx(m.origin[0],m.origin[1]+m.height*m.resolution);ctx.globalAlpha=.86;ctx.drawImage(mapImage,p[0],p[1],m.width*m.resolution*scale,m.height*m.resolution*scale);ctx.globalAlpha=1;
if($('line').checked)line([...m.line,m.line[0]],'#6b7a96',Math.max(1,scale*.025),[scale*.16,scale*.12]);if($('trail').checked){const history=run.frames.slice(0,frameIndex+1);line(history.map(a=>a.ego),'#8b5cf6',Math.max(2,scale*.04));if(!run.barrier)line(history.map(a=>a.opponent),'#f87171',Math.max(1.5,scale*.025))}
if(run.barrier){car(run.barrier,'#fb7185',run.barrier[3],run.barrier[4])}else{car(f.opponent,'#fb7185')}car(f.ego,'#a78bfa');
let dpr=window.devicePixelRatio||1;ctx.font=`${11*dpr}px system-ui`;ctx.fillStyle='#e2e8f0';ctx.fillText('1 m',w-65*dpr,h-31*dpr);ctx.strokeStyle='#e2e8f0';ctx.lineWidth=2*dpr;ctx.beginPath();ctx.moveTo(w-20*dpr-scale,h-20*dpr);ctx.lineTo(w-20*dpr,h-20*dpr);ctx.stroke();
if(run.result.collision&&f.t>=run.result.collision.time-.06){ctx.fillStyle='#7f1d1ddd';ctx.fillRect(0,h-65*dpr,w,65*dpr);ctx.fillStyle='#fee2e2';ctx.font=`bold ${18*dpr}px system-ui`;ctx.fillText('Contact recorded — this run failed',24*dpr,h-27*dpr)}
}
function update(){const run=current(),r=run.result,f=run.frames[frameIndex],s=r.scenario;$('seek').max=run.frames.length-1;$('seek').value=frameIndex;$('time').textContent=`${fmt(f.t,1)} / ${fmt(r.sim_seconds,1)} s`;$('action').textContent=labels[f.action]||f.action;$('canvasLabel').textContent=`Stage ${s.stage} · ${stageNames[s.stage]} · ${fmt(f.t,1)} s`;
const probs=f.probabilities||f.probs||[],actions=['LEFT','RIGHT','WAIT','PROBE'];$('probabilities').replaceChildren();probs.forEach((prob,i)=>{const a=actions[i],box=document.createElement('div');box.className='prob';const head=document.createElement('div');head.className='probhead';const name=document.createElement('span'),value=document.createElement('span');name.textContent=labels[a];value.textContent=fmt(prob*100,1)+'%';head.append(name,value);const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('i');fill.style.background=colors[a];fill.style.width=Math.max(0,Math.min(100,prob*100))+'%';bar.append(fill);box.append(head,bar);$('probabilities').append(box)});
$('otherAction').textContent=f.opponent_action?(labels[f.opponent_action]||f.opponent_action):'Line / static';$('otherSees').textContent=s.stage===3?(f.opponent_target?'Yes':'No'):'—';$('speed').textContent=fmt(f.ego[3])+' m/s';$('otherSpeed').textContent=run.barrier?'Stationary':fmt(f.opponent[3])+' m/s';$('steer').textContent=fmt((f.command||[0])[0]*180/Math.PI,1)+'°';$('targetSpeed').textContent=fmt((f.command||[0,0])[1])+' m/s';$('lead').textContent=(f.relative_progress>0?'+':'')+fmt(f.relative_progress)+' m';drawTrack()}
function choose(i){runIndex=(i+DATA.traces.length)%DATA.traces.length;frameIndex=0;playTime=0;lastTime=0;$('episode').value=runIndex;const run=current(),r=run.result,s=r.scenario;$('status').className='status '+r.outcome;$('status').textContent={pass:'Clean pass verified',safe_stop:'Blocked track · safe stop',no_pass:'No completed pass',collision:'Contact · failed run'}[r.outcome]||r.outcome;
$('runDescription').textContent=`${r.teacher?'Geometry teacher':'Local neural model'} · start ${fmt(s.s,1)} m · gap ${fmt(s.gap,1)} m · seed ${s.seed}. Caps: ${fmt(s.ego_speed,1)} / ${fmt(s.opponent_speed,1)} m/s.`;$('opponentLabel').textContent=s.blocked?'Full-width blocker':s.stage===1?'Static obstacle':s.stage===3?'Stage 2 neural rival':'Line-following opponent';$('network').textContent=r.network_latency&&r.network_latency.count?fmt(r.network_latency.p95_ms,3)+' ms':'Not measured / teacher';$('controller').textContent=r.controller_latency&&r.controller_latency.count?fmt(r.controller_latency.p95_ms,3)+' ms':'Not measured';$('gap').textContent=fmt(r.minimum_body_gap,3)+' m';$('source').textContent=run.source_file;$('probabilitySource').textContent=r.teacher?'Teacher scores before safety filtering':'Raw neural probabilities before safety / commitment filtering';$('stageWarning').textContent=s.stage===3?'The inside probe is experimental. These runs do not establish that the model forced the opponent to yield.':'';$('summaryTime').textContent=fmt(r.sim_seconds,1)+' s';$('summaryPass').textContent=r.passed_at==null?'—':fmt(r.passed_at,2)+' s';$('summaryWall').textContent=fmt(r.minimum_wall_clearance,3)+' m';$('summaryCalls').textContent=r.llm_calls_in_test==null?'Unrecorded':r.llm_calls_in_test;
$('outcomeDescription').textContent=r.collision?`Failure at ${fmt(r.collision.time,2)} s: ${r.collision.type}. A lead change before contact is not counted as success.`:r.success?'The car cleared the obstacle or opponent, returned near the reference line, and held the lead without contact in this recorded episode.':r.safe_blocked_stop?'The blocked-track scenario ended with the car stopped and no contact. It did not try to squeeze through an unavailable gap.':'The episode ended without contact but no completed pass was verified. This is not counted as an overtaking success.';update()}
function toggle(){playing=!playing;if(playing&&frameIndex===current().frames.length-1){frameIndex=0;playTime=0}lastTime=0;$('play').textContent=playing?'Pause':'Play'}
function tick(now){if(playing){if(lastTime)playTime+=(now-lastTime)/1000*Number($('rate').value);lastTime=now;const frames=current().frames;while(frameIndex<frames.length-1&&frames[frameIndex+1].t<=playTime)frameIndex++;if(frameIndex===frames.length-1){if($('autonext').checked&&runIndex<DATA.traces.length-1){choose(runIndex+1)}else{playing=false;$('play').textContent='Play'}}update()}requestAnimationFrame(tick)}
function drawLearning(){const host=$('learning');host.replaceChildren();if(!DATA.histories.length)return;$('learningSection').classList.remove('hidden');DATA.histories.forEach(entry=>{let card=document.createElement('div');card.className='learning';let title=document.createElement('h3');title.textContent=entry.name.replaceAll('_',' ');let sub=document.createElement('p'),rep=entry.report;sub.textContent=`Selected epoch ${rep.selected_epoch} / ${rep.epochs} · validation label agreement ${rep.validation?fmt(rep.validation.accuracy*100,1)+'%':'not recorded'}`;let c=document.createElement('canvas');c.setAttribute('aria-label','Recorded training and validation loss');let leg=document.createElement('div');leg.className='historyLegend';leg.innerHTML='<span style="color:#a78bfa">● Training loss</span><span style="color:#6ee7b7">● Validation loss</span>';card.append(title,sub,c,leg);let source=document.createElement('p');source.className='sourceLine';const sources=rep.metadata?.label_sources||{},count=Object.values(sources).reduce((a,b)=>a+b,0),api=sources.openai_responses||0;source.textContent=`${count.toLocaleString()} original labeled snapshots · ${api.toLocaleString()} accepted LLM/API labels. `+(rep.metadata?.not_rollout_ground_truth?'Geometry proposals, not rollout ground truth. ':'');if(entry.dataset?.action_counts&&rep.metadata?.stage===3)source.textContent+=`Raw inside-probe labels: ${entry.dataset.action_counts.PROBE||0}. This does not establish learned forced yielding.`;card.append(source);host.append(card);const [w,h]=measure(c),cx=c.getContext('2d'),d=window.devicePixelRatio||1,pad=32*d,rows=rep.history,all=rows.flatMap(a=>[a.train_loss,a.validation_loss]).filter(Number.isFinite),max=Math.max(...all)*1.08||1,maxEpoch=Math.max(...rows.map(a=>a.epoch));cx.clearRect(0,0,w,h);cx.strokeStyle='#34405a';cx.lineWidth=d;cx.font=`${10*d}px system-ui`;cx.fillStyle='#a1adc7';for(let i=0;i<4;i++){let y=pad+(h-pad*2)*i/3;cx.beginPath();cx.moveTo(pad,y);cx.lineTo(w-10*d,y);cx.stroke();cx.fillText(fmt(max*(1-i/3),2),2*d,y+3*d)}for(const [key,color] of [['train_loss','#a78bfa'],['validation_loss','#6ee7b7']]){cx.strokeStyle=color;cx.lineWidth=2*d;cx.beginPath();let first=true;rows.forEach(a=>{if(!Number.isFinite(a[key]))return;let x=pad+(w-pad-12*d)*(a.epoch-1)/Math.max(1,maxEpoch-1),y=h-pad-(h-pad*2)*a[key]/max;first?cx.moveTo(x,y):cx.lineTo(x,y);first=false});cx.stroke()}cx.fillText('1',pad,h-10*d);cx.fillText('Epoch '+maxEpoch,w-65*d,h-10*d);if(rep.selected_epoch){let x=pad+(w-pad-12*d)*(rep.selected_epoch-1)/Math.max(1,maxEpoch-1);cx.setLineDash([3*d,3*d]);cx.strokeStyle='#94a3b8';cx.beginPath();cx.moveTo(x,pad);cx.lineTo(x,h-pad);cx.stroke()}})}
function drawProvenance(){const totals={};DATA.histories.forEach(entry=>{const sources=entry.report.metadata?.label_sources||{};Object.entries(sources).forEach(([source,n])=>totals[source]=(totals[source]||0)+n)});const total=Object.values(totals).reduce((a,b)=>a+b,0),api=totals.openai_responses||0,geometry=totals.geometric_bootstrap||0;$('trainingBanner').textContent=total?`Training provenance: ${total.toLocaleString()} labeled simulation snapshots · ${geometry.toLocaleString()} geometric bootstrap labels · ${api.toLocaleString()} accepted LLM/API labels. `+(api===0?'This is a local-teacher prototype; repeated LLM teaching has not been completed.':'LLM labels are advice checked against geometric feasibility, not rollout ground truth.'):'Training provenance was not bundled with these recordings; no LLM-training claim is implied.'}
function drawDrivingResults(){const table=$('drivingResults');table.replaceChildren();DATA.traces.forEach((run,i)=>{let row=document.createElement('tr'),r=run.result;const values=[`Stage ${r.scenario.stage} / ${run.name}`,run.checkpoint,r.outcome.replaceAll('_',' '),r.passed_at==null?'—':fmt(r.passed_at,2)+' s',r.collision?'Yes':'No',r.network_latency?.count?fmt(r.network_latency.p95_ms,3)+' ms':'—'];values.forEach((value,j)=>{let cell=document.createElement('td');if(j===0){let button=document.createElement('button');button.textContent=value;button.style.fontSize='12px';button.style.padding='5px 8px';button.addEventListener('click',()=>choose(i));cell.append(button)}else{cell.textContent=value}row.append(cell)});table.append(row)})}
DATA.traces.forEach((run,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`Stage ${run.result.scenario.stage} · ${run.checkpoint} · ${run.name} · ${run.result.outcome.replaceAll('_',' ')}`;$('episode').append(o)});
$('episode').addEventListener('change',()=>choose(Number($('episode').value)));$('previous').addEventListener('click',()=>choose(runIndex-1));$('next').addEventListener('click',()=>choose(runIndex+1));$('play').addEventListener('click',toggle);$('restart').addEventListener('click',()=>choose(runIndex));$('seek').addEventListener('input',()=>{frameIndex=Number($('seek').value);playTime=current().frames[frameIndex].t;lastTime=0;update()});for(const id of ['follow','trail','line'])$(id).addEventListener('change',update);$('save').addEventListener('click',()=>{let a=document.createElement('a');a.download='roboracer-replay-'+runIndex+'-'+frameIndex+'.png';a.href=canvas.toDataURL();a.click()});window.addEventListener('resize',()=>{update();drawLearning()});document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();toggle()}else if(e.code==='ArrowRight'||e.code==='ArrowLeft'){e.preventDefault();frameIndex=Math.max(0,Math.min(current().frames.length-1,frameIndex+(e.code==='ArrowRight'?1:-1)));playTime=current().frames[frameIndex].t;update()}});mapImage.onload=()=>{choose(DATA.initial_index||0);drawProvenance();drawLearning();drawDrivingResults();requestAnimationFrame(tick)};mapImage.src=DATA.map.image;
</script></body></html>'''


def make_poster(data, output):
    """Optional static overview of genuine trajectories, suitable for sharing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    traces = data["traces"]
    cols = min(3, len(traces))
    rows = int(np.ceil(len(traces) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.4 * rows), squeeze=False, dpi=150)
    m = data["map"]
    image = Image.open(io.BytesIO(base64.b64decode(m["image"].split(",")[1])))
    ox, oy = m["origin"]
    extent = [ox, ox + m["width"] * m["resolution"], oy, oy + m["height"] * m["resolution"]]
    reference = np.asarray(m["line"])
    for ax, trace in zip(axes.flat, traces):
        r, frames = trace["result"], trace["frames"]
        ego = np.asarray([f["ego"] for f in frames])
        opponent = np.asarray([f["opponent"] for f in frames])
        ax.imshow(image, cmap="gray", extent=extent, origin="upper", vmin=0, vmax=255)
        ax.plot(*reference.T, "--", c="#8899aa", lw=.7, label="Reference")
        ax.plot(*ego[:, :2].T, c="#7c3aed", lw=2., label="Our car")
        if "barrier" not in trace:
            ax.plot(*opponent[:, :2].T, c="#e11d48", lw=1.4, label="Obstacle / opponent")
            x, y, yaw = opponent[-1, :3]
            vertices = np.array([[.29,.155],[.29,-.155],[-.29,-.155],[-.29,.155]])
            rotation = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            ax.add_patch(Polygon(vertices @ rotation.T + [x,y], color="#e11d48", zorder=5))
        else:
            x, y, yaw, length, width = trace["barrier"]
            vertices = np.array([[length/2,width/2],[length/2,-width/2],[-length/2,-width/2],[-length/2,width/2]])
            rotation = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            ax.add_patch(Polygon(vertices @ rotation.T + [x,y], color="#e11d48"))
        ax.scatter(*ego[0,:2], c="#7c3aed", marker="o", s=18, zorder=4)
        ax.scatter(*ego[-1,:2], c="#7c3aed", marker="^", s=28, zorder=4)
        ax.set_aspect("equal")
        ax.set_title(f"Stage {r['scenario']['stage']}: {trace['name']}\n{r['outcome'].replace('_',' ')} · {r['sim_seconds']:.1f} simulated s", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=7, loc="lower right")
    for ax in list(axes.flat)[len(traces):]:
        ax.axis("off")
    fig.suptitle("Recorded simulator trajectories — not a new simulation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "replay.html")
    parser.add_argument("--history", action="append", type=Path, default=[])
    parser.add_argument("--poster", type=Path)
    args = parser.parse_args()
    histories = args.history or sorted(MODELS.glob("*.history.json"))
    data = load_replay_data(args.traces, histories)
    # JSON inside a script must not allow a filename or metadata string to end
    # the script element. Text labels are rendered using textContent.
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", encoded))
    if args.poster:
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        make_poster(data, args.poster)
    print(json.dumps({"replay": str(args.output.resolve()), "runs": len(data["traces"]),
                      "training_histories": len(data["histories"]),
                      "poster": str(args.poster.resolve()) if args.poster else None}))


if __name__ == "__main__":
    main()
