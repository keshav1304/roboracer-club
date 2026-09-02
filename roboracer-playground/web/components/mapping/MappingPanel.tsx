"use client";

import type { Gateway } from "@/lib/useGateway";
import type { Mapping, MappingStep } from "@/lib/useMapping";
import { isMapControllerAlgo } from "@/lib/algorithms";
import MapStep from "./MapStep";
import RacelineStep from "./RacelineStep";
import ControllerStep from "./ControllerStep";

interface Props {
  gw: Gateway;
  lab: Mapping;
}

interface StepDef {
  id: MappingStep;
  n: number;
  label: string;
}

const STEPS: StepDef[] = [
  { id: "map", n: 1, label: "Build Map" },
  { id: "raceline", n: 2, label: "Generate Raceline" },
  { id: "controller", n: 3, label: "Controller" },
];

export default function MappingPanel({ gw, lab }: Props) {
  const chips: Record<MappingStep, { text: string; tone: "" | "on" | "mid" }> = {
    map: gw.status.slam.active
      ? { text: gw.status.slam.lap_done ? "lap done, save" : "mapping…", tone: "on" }
      : lab.maps.length
        ? { text: `${lab.maps.length} saved`, tone: "" }
        : { text: "none yet", tone: "" },
    raceline: lab.opt.running
      ? { text: "optimizing…", tone: "on" }
      : lab.raceline
        ? { text: `editing ${lab.editorMap}`, tone: "mid" }
        : lab.racelines.length
          ? { text: `${lab.racelines.length} saved`, tone: "" }
          : { text: "none yet", tone: "" },
    controller:
      isMapControllerAlgo(gw.status.algorithm)
        ? {
            text: `lap ${gw.telemetry.lap_count + 1} · ${gw.telemetry.lap_time.toFixed(1)} s`,
            tone: "on",
          }
        : lab.selectedRaceline
          ? { text: lab.selectedRaceline, tone: "" }
          : { text: "pick a raceline", tone: "" },
  };

  return (
    <div className="stepper">
      {STEPS.map((s) => {
        const active = lab.step === s.id;
        const chip = chips[s.id];
        return (
          <div key={s.id} className={`step ${active ? "open" : ""}`}>
            <button className="step-head" onClick={() => lab.setStep(s.id)}>
              <span className={`step-num ${active ? "active" : ""}`}>
                {s.n}
              </span>
              <span className="step-label">{s.label}</span>
              <span className={`step-chip ${chip.tone}`}>{chip.text}</span>
            </button>
            {active && (
              <div className="step-content">
                {s.id === "map" && <MapStep gw={gw} lab={lab} />}
                {s.id === "raceline" && <RacelineStep gw={gw} lab={lab} />}
                {s.id === "controller" && <ControllerStep gw={gw} lab={lab} />}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
