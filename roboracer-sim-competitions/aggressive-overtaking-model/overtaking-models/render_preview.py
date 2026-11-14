"""Render a compact GIF from actual recorded simulator poses and decisions.

No interpolation, vehicle physics, or invented positions are used here. At
2x playback and 10 fps, the renderer selects every fourth 20 Hz recorded
frame, plus the exact final recorded frame of each episode.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bootstrap import ARTIFACTS, MODELS
from replay import load_replay_data

BACKGROUND = "#0b1020"
PANEL = "#151c30"
TEXT = "#eef2fc"
MUTED = "#a1adc7"
EGO = "#a78bfa"
OPPONENT = "#fb7185"
COLORS = ["#60a5fa", "#fb923c", "#94a3b8", "#fbbf24"]
ACTION_NAMES = {"LEFT": "Pass left", "RIGHT": "Pass right", "WAIT": "Follow / hold", "PROBE": "Inside probe"}
STAGE_NAMES = {1: "Static obstacle", 2: "Slower line-following car", 3: "Equal-cap neural rival"}


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def render_frame(data, trace, index, width=900, height=500, playback_speed=2.):
    frame = trace["frames"][index]
    result = trace["result"]
    stage = result["scenario"]["stage"]
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((22, 15), f"STAGE {stage} / {STAGE_NAMES[stage]}", fill=TEXT, font=font(22, True))
    draw.text((22, 45), f"{trace['checkpoint'].capitalize()} checkpoint  |  Recorded simulator replay  |  {playback_speed:g}x playback", fill=MUTED, font=font(13))
    viewport = (18, 76, width - 260, height - 66)
    x0, y0, x1, y1 = viewport
    vw, vh = x1 - x0, y1 - y0
    ego, opponent = np.asarray(frame["ego"]), np.asarray(frame["opponent"])
    partner = np.asarray(trace.get("barrier", opponent))
    midpoint = (ego[:2] + partner[:2]) / 2
    dx, dy = np.abs(ego[:2] - partner[:2])
    view_width = max(9.4, float(dx + 4.), float(dy + 3.) * vw / vh)
    metres_per_pixel = view_width / vw
    view_height = vh * metres_per_pixel
    xmin, ymax = midpoint[0] - view_width / 2, midpoint[1] + view_height / 2

    def transform(point):
        return x0 + (float(point[0]) - xmin) / metres_per_pixel, y0 + (ymax - float(point[1])) / metres_per_pixel

    map_data = data["map"]
    ox, oy = map_data["origin"]
    res = map_data["resolution"]
    source = Image.open(io.BytesIO(base64.b64decode(map_data["image"].split(",")[1]))).convert("RGB")
    # Affine sampling of the real occupancy image; out-of-map areas stay dark.
    scale = metres_per_pixel / res
    crop = source.transform((vw, vh), Image.Transform.AFFINE,
                            (scale, 0., (xmin - ox) / res,
                             0., scale, map_data["height"] - (ymax - oy) / res),
                            resample=Image.Resampling.BILINEAR, fillcolor="#101829")
    map_draw = ImageDraw.Draw(crop)

    def local(point):
        px, py = transform(point)
        return px - x0, py - y0

    reference = map_data["line"]
    map_draw.line([local(p) for p in reference + reference[:1]], fill="#8994a8", width=1)
    history = trace["frames"][:index + 1]
    if len(history) > 1:
        map_draw.line([local(f["ego"]) for f in history], fill="#8b5cf6", width=2)
        if "barrier" not in trace:
            map_draw.line([local(f["opponent"]) for f in history], fill="#ec7185", width=2)

    def car(pose, color, length=.58, car_width=.31):
        x, y, yaw = pose[:3]
        corners = np.array([[length/2,car_width/2], [length/2,-car_width/2],
                            [-length/2,-car_width/2], [-length/2,car_width/2]])
        rotation = np.array([[np.cos(yaw),-np.sin(yaw)], [np.sin(yaw),np.cos(yaw)]])
        polygon = corners @ rotation.T + [x, y]
        map_draw.polygon([local(p) for p in polygon], fill=color, outline="#283044", width=1)
        front = np.array([x, y]) + length * .38 * np.array([np.cos(yaw), np.sin(yaw)])
        map_draw.line([local([x,y]),local(front)], fill="#ffffff", width=2)

    if "barrier" in trace:
        car(trace["barrier"], OPPONENT, trace["barrier"][3], trace["barrier"][4])
    else:
        car(opponent, OPPONENT)
    car(ego, EGO)
    image.paste(crop, (x0, y0))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=3, outline="#45516b", width=1)
    draw.rectangle((x0+10, y0+10, x0+131, y0+37), fill=PANEL)
    draw.text((x0+18, y0+16), f"{frame['t']:5.2f} / {result['sim_seconds']:.1f} s", fill=TEXT, font=font(12))
    draw.rounded_rectangle((x0+6,y1-33,x0+310,y1-5),radius=4,fill=PANEL)
    draw.ellipse((x0+12,y1-24,x0+20,y1-16),fill=EGO)
    draw.ellipse((x0+124,y1-24,x0+132,y1-16),fill=OPPONENT)
    draw.text((x0+26,y1-27),"Our car",fill=TEXT,font=font(11,True))
    draw.text((x0+138,y1-27),"Obstacle / opponent",fill=TEXT,font=font(11,True))

    sx = x1 + 20
    draw.rounded_rectangle((sx-7,y0,width-17,y1),radius=8,fill=PANEL)
    draw.text((sx+5,y0+13),"APPLIED DECISION",fill=MUTED,font=font(11,True))
    draw.text((sx+5,y0+31),ACTION_NAMES.get(frame["action"],frame["action"]),fill=TEXT,font=font(21,True))
    draw.text((sx+5,y0+61),"Teacher preferences" if result.get("teacher") else "Raw neural probabilities",fill=MUTED,font=font(11))
    for j, (probability, color) in enumerate(zip(frame["probabilities"], COLORS)):
        yy = y0 + 84 + 34*j
        name = ("Left", "Right", "Follow / hold", "Inside probe")[j]
        draw.text((sx+5,yy),name,fill=TEXT,font=font(11))
        draw.text((width-72,yy),f"{probability*100:5.1f}%",fill=TEXT,font=font(11))
        draw.rounded_rectangle((sx+5,yy+17,width-33,yy+22),radius=2,fill="#2b344b")
        right = sx+5 + max(0., min(1., probability))*(width-38-sx)
        if right > sx+5:
            draw.rounded_rectangle((sx+5,yy+17,right,yy+22),radius=2,fill=color)
    yy = y0 + 231
    command = frame["command"]
    for name, value in [("Speed",f"{ego[3]:.2f} m/s"),
                        ("Other speed",f"{opponent[3]:.2f} m/s"),
                        ("Steering",f"{np.rad2deg(command[0]):+.1f} deg"),
                        ("Network p95",("Not used" if result.get("teacher") else f"{result.get('network_latency',{}).get('p95_ms',float('nan')):.3f} ms")),
                        ("Controller p95",f"{result.get('controller_latency',{}).get('p95_ms',float('nan')):.2f} ms")]:
        draw.text((sx+5,yy),name,fill=MUTED,font=font(11))
        draw.text((sx+112,yy),value,fill=TEXT,font=font(11,True))
        yy += 20
    lead_confirmed = result.get("passed_at") is not None and frame["t"] >= result["passed_at"]
    verified = bool(result.get("success") and lead_confirmed and (frame["t"] >= result["passed_at"] + 2. or index == len(trace["frames"])-1))
    status = "PASS VERIFIED" if verified else "LEAD / FOLLOW-THROUGH" if lead_confirmed else "RUN IN PROGRESS"
    if result.get("collision") and frame["t"] >= result["collision"]["time"] - .06:
        status = "CONTACT / FAILED RUN"
    elif index == len(trace["frames"]) - 1 and not verified:
        status = "SAFE STOP" if result.get("safe_blocked_stop") else "NO COMPLETED PASS"
    color = "#6ee7b7" if verified else "#c4b5fd"
    draw.text((22,height-47),status,fill=color,font=font(14,True))
    caveat = "Forced yielding is not established." if stage == 3 else "Each car position comes from the saved simulation."
    draw.text((210,height-46),caveat,fill=MUTED,font=font(12))
    history = next((h for h in data["histories"] if h["report"].get("metadata",{}).get("stage") == stage), None)
    sources = history["report"].get("metadata",{}).get("label_sources",{}) if history else {}
    api_labels = sources.get("openai_responses",0)
    teaching = "Training-policy demonstration, not a neural test" if result.get("teacher") else f"{api_labels} accepted API training labels" if sources else "Training provenance not bundled"
    test_calls = result.get("llm_calls_in_test", "unrecorded")
    draw.text((22,height-24),f"Simulation prototype  |  {teaching}  |  LLM calls in this test: {test_calls}",fill=MUTED,font=font(11))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "three_models.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speed", type=float, default=2.)
    parser.add_argument("--width", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or args.speed <= 0 or args.width < 700:
        raise ValueError("Use fps 1–30, positive playback speed, and width at least 700.")
    data = load_replay_data(args.traces, sorted(MODELS.glob("*.history.json")))
    images, manifest = [], []
    duration = round(1000 / args.fps)
    for trace in data["traces"]:
        times = np.array([f["t"] for f in trace["frames"]])
        wanted = np.arange(times[0], times[-1], args.speed / args.fps)
        selected = np.searchsorted(times, wanted, side="right") - 1
        selected = np.unique(np.r_[selected, len(times)-1])
        frames = [render_frame(data,trace,int(i),width=args.width,playback_speed=args.speed) for i in selected]
        # Brief start/end holds make stage changes readable; poses are repeated,
        # never extrapolated or interpolated.
        images.extend([frames[0]] * 3 + frames + [frames[-1]] * 7)
        manifest.append({"source":trace["source_file"],"selected_recorded_frame_indices":selected.tolist(),
                         "outcome":trace["result"]["outcome"],"model_sha256":trace["result"].get("model_sha256")})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    paletted = [image.quantize(colors=128, method=Image.Quantize.FASTOCTREE) for image in images]
    paletted[0].save(args.output,save_all=True,append_images=paletted[1:],duration=duration,loop=0,optimize=True,disposal=2)
    images[0].save(args.output.with_suffix(".first.png"))
    images[-1].save(args.output.with_suffix(".last.png"))
    args.output.with_suffix(".manifest.json").write_text(json.dumps({"type":"recorded_replay",
        "frames":len(images),"fps":args.fps,"playback_speed":args.speed,"sources":manifest},indent=2)+"\n")
    print(json.dumps({"gif":str(args.output.resolve()),"frames":len(images),
                       "duration_seconds":len(images)/args.fps,"bytes":args.output.stat().st_size}))


if __name__ == "__main__":
    main()
