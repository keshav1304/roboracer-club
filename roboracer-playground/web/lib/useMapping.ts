"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Gateway,
  RacelineData,
  Transform2D,
  httpBaseFromWs,
} from "@/lib/useGateway";

export type MappingStep = "map" | "raceline" | "controller";
export type EditTool = "none" | "erase" | "wall";

export interface MapMeta {
  name: string;
  created: number;
  resolution: number;
  origin: number[];
  version: number;
  width: number;
  height: number;
  map_to_world?: Transform2D | null;
}

export interface RacelineMeta {
  name: string;
  map: string;
  created: number;
  params?: Record<string, number>;
  anchors?: [number, number][] | null;
  stats?: {
    length_m: number;
    n_points: number;
    v_min: number;
    v_max: number;
    lap_time_est: number;
  };
}

export interface OptState {
  running: boolean;
  evals: number;
  budget: number;
  bestCost: number | null;
  preview: { x: number[]; y: number[] } | null;
  error: string | null;
}

const IDLE_OPT: OptState = {
  running: false,
  evals: 0,
  budget: 0,
  bestCost: null,
  preview: null,
  error: null,
};

export const FAST_PARAM_DEFAULTS: Record<string, number> = {
  vmin: 0.5,
  vmax: 6.5,
  alat: 3.5,
  smooth: 2.0,
  margin: 0.25,
  n_out: 800,
};

export const OPT_PARAM_DEFAULTS: Record<string, number> = {
  margin: 0.25,
  budget: 8000,
  n_ctrl: 80,
  vmin: 0.5,
  vmax: 6.5,
  alat: 3.5,
};

export function suggestedMapName(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `track-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`;
}

export function useMapping(gw: Gateway, gatewayUrl: string, token: string) {
  const httpBase = useMemo(() => httpBaseFromWs(gatewayUrl), [gatewayUrl]);
  const tokenQS = token ? `token=${encodeURIComponent(token)}` : "";

  const [step, setStep] = useState<MappingStep>("map");

  // --- artifact lists -----------------------------------------------------
  const [maps, setMaps] = useState<MapMeta[]>([]);
  const [racelines, setRacelines] = useState<RacelineMeta[]>([]);

  const fetchJson = useCallback(
    async (path: string) => {
      const sep = path.includes("?") ? "&" : "?";
      const res = await fetch(
        `${httpBase}${path}${tokenQS ? sep + tokenQS : ""}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    [httpBase, tokenQS]
  );

  const refreshMaps = useCallback(async () => {
    try {
      const data = await fetchJson("/maps");
      setMaps((data.maps as MapMeta[]) ?? []);
    } catch {
      /* gateway offline; lists refresh on next rev bump */
    }
  }, [fetchJson]);

  const refreshRacelines = useCallback(async () => {
    try {
      const data = await fetchJson("/racelines");
      setRacelines((data.racelines as RacelineMeta[]) ?? []);
    } catch {
      /* ignore */
    }
  }, [fetchJson]);

  useEffect(() => {
    refreshMaps();
  }, [refreshMaps, gw.mapsRev, gw.status.wsConnected]);
  useEffect(() => {
    refreshRacelines();
  }, [refreshRacelines, gw.racelinesRev, gw.status.wsConnected]);

  const mapImageUrl = useCallback(
    (name: string, version: number) =>
      `${httpBase}/maps/${name}/image.png?v=${version}${
        tokenQS ? "&" + tokenQS : ""
      }`,
    [httpBase, tokenQS]
  );

  const mapGridUrl = useCallback(
    (name: string, version: number) =>
      `${httpBase}/maps/${name}/grid?v=${version}${
        tokenQS ? "&" + tokenQS : ""
      }`,
    [httpBase, tokenQS]
  );

  // --- map step state -------------------------------------------------------
  /** Saved map opened for viewing/cleanup in the Map step. */
  const [viewedMap, setViewedMap] = useState<string | null>(null);
  const [tool, setTool] = useState<EditTool>("none");
  const [brushRadius, setBrushRadius] = useState(3);

  // --- raceline editor state ------------------------------------------------
  const [editorMap, setEditorMapState] = useState<string | null>(null);
  const [anchors, setAnchorsState] = useState<[number, number][]>([]);
  const [fastParams, setFastParams] = useState<Record<string, number>>({
    ...FAST_PARAM_DEFAULTS,
  });
  const [optParams, setOptParams] = useState<Record<string, number>>({
    ...OPT_PARAM_DEFAULTS,
  });
  const [raceline, setRaceline] = useState<RacelineData | null>(null);
  const [opt, setOpt] = useState<OptState>(IDLE_OPT);
  const [selectedAnchor, setSelectedAnchor] = useState<number | null>(null);

  const reqIdRef = useRef(0);
  const acceptedReqRef = useRef(0);
  const recomputeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setEditorMap = useCallback((name: string | null) => {
    setEditorMapState(name);
    setAnchorsState([]);
    setRaceline(null);
    setSelectedAnchor(null);
    setOpt(IDLE_OPT);
  }, []);

  /** Debounced authoritative recompute on the gateway. */
  const scheduleRecompute = useCallback(
    (map: string | null, pts: [number, number][], params: Record<string, number>) => {
      if (!map || pts.length < 4) return;
      if (recomputeTimer.current) clearTimeout(recomputeTimer.current);
      recomputeTimer.current = setTimeout(() => {
        const id = ++reqIdRef.current;
        gw.racelineUpdate(map, pts, params, id);
      }, 150);
    },
    [gw]
  );

  const setAnchors = useCallback(
    (pts: [number, number][]) => {
      setAnchorsState(pts);
      scheduleRecompute(editorMap, pts, fastParams);
    },
    [editorMap, fastParams, scheduleRecompute]
  );

  const setFastParam = useCallback(
    (name: string, value: number) => {
      setFastParams((prev) => {
        const next = { ...prev, [name]: value };
        scheduleRecompute(editorMap, anchors, next);
        return next;
      });
    },
    [editorMap, anchors, scheduleRecompute]
  );

  // Accept authoritative recompute results (latest request wins).
  useEffect(() => {
    const data = gw.racelineData;
    if (!data) return;
    const id = Number(data.req_id ?? 0);
    if (id < acceptedReqRef.current) return;
    acceptedReqRef.current = id;
    if (data.ok) setRaceline(data);
  }, [gw.racelineData]);

  // Optimizer event stream → local optimizer state.
  useEffect(() => {
    const ev = gw.optEvent;
    if (!ev) return;
    const phase = ev.event;
    if (phase === "started") {
      setOpt({ ...IDLE_OPT, running: true, error: null });
    } else if (phase === "progress") {
      setOpt((o) => ({
        ...o,
        running: true,
        evals: ev.evals ?? o.evals,
        budget: ev.budget ?? o.budget,
        bestCost: ev.best_cost ?? o.bestCost,
        preview: ev.preview ?? o.preview,
        error: null,
      }));
    } else if (phase === "done") {
      setOpt((o) => ({
        ...o,
        running: false,
        preview: null,
        evals: ev.evals ?? o.evals,
        bestCost: ev.best_cost ?? o.bestCost,
        error: null,
      }));
      if (ev.raceline?.ok) setRaceline(ev.raceline);
      if (ev.anchors?.length) setAnchorsState(ev.anchors);
    } else if (phase === "error") {
      setOpt((o) => ({
        ...o,
        running: false,
        preview: null,
        error: ev.reason ?? "Optimization failed",
      }));
    } else if (phase === "cancelled") {
      setOpt((o) => ({ ...o, running: false, preview: null }));
    }
  }, [gw.optEvent]);

  const startOptimize = useCallback(() => {
    if (!editorMap) return;
    setOpt({ ...IDLE_OPT, running: true });
    gw.optStart(editorMap, { ...optParams, ...pickVelocityParams(fastParams) });
  }, [gw, editorMap, optParams, fastParams]);

  const cancelOptimize = useCallback(() => gw.optCancel(), [gw]);

  /** Load a saved raceline into the editor (map + anchors + params). */
  const loadRacelineIntoEditor = useCallback(
    async (meta: RacelineMeta) => {
      setEditorMapState(meta.map || null);
      setSelectedAnchor(null);
      setOpt(IDLE_OPT);
      if (meta.params) {
        setFastParams((prev) => ({ ...prev, ...meta.params }));
      }
      try {
        const res = await fetchJson(`/racelines/${meta.name}`);
        const data = res.data as { s: number[]; x: number[]; y: number[]; theta: number[]; v: number[] };
        const rl: RacelineData = {
          ok: true,
          ...data,
          length_m: data.s[data.s.length - 1] ?? 0,
          lap_time_est: meta.stats?.lap_time_est ?? 0,
          margin: meta.params?.margin ?? FAST_PARAM_DEFAULTS.margin,
        };
        setRaceline(rl);
        if (meta.anchors?.length) {
          setAnchorsState(meta.anchors);
        } else {
          // Subsample the dense line into editable anchors.
          const n = Math.min(40, data.x.length);
          const pts: [number, number][] = [];
          for (let i = 0; i < n; i++) {
            const idx = Math.floor((i * data.x.length) / n);
            pts.push([data.x[idx], data.y[idx]]);
          }
          setAnchorsState(pts);
        }
      } catch {
        setRaceline(null);
        setAnchorsState(meta.anchors ?? []);
      }
    },
    [fetchJson]
  );

  // --- race step state --------------------------------------------------------
  const [selectedRaceline, setSelectedRaceline] = useState<string | null>(null);
  const [trackOverlay, setTrackOverlay] = useState<{
    meta: RacelineMeta;
    data: { s: number[]; x: number[]; y: number[]; theta: number[]; v: number[] };
  } | null>(null);

  useEffect(() => {
    let stale = false;
    if (!selectedRaceline) {
      setTrackOverlay(null);
      return;
    }
    (async () => {
      try {
        const res = await fetchJson(`/racelines/${selectedRaceline}`);
        if (!stale) setTrackOverlay(res);
      } catch {
        if (!stale) setTrackOverlay(null);
      }
    })();
    return () => {
      stale = true;
    };
  }, [selectedRaceline, fetchJson, gw.racelinesRev]);

  return {
    step,
    setStep,
    maps,
    racelines,
    refreshMaps,
    refreshRacelines,
    mapImageUrl,
    mapGridUrl,
    viewedMap,
    setViewedMap,
    tool,
    setTool,
    brushRadius,
    setBrushRadius,
    editorMap,
    setEditorMap,
    anchors,
    setAnchors,
    setAnchorsState,
    selectedAnchor,
    setSelectedAnchor,
    fastParams,
    setFastParam,
    optParams,
    setOptParams,
    raceline,
    opt,
    startOptimize,
    cancelOptimize,
    loadRacelineIntoEditor,
    selectedRaceline,
    setSelectedRaceline,
    trackOverlay,
  };
}

function pickVelocityParams(fast: Record<string, number>) {
  const { vmin, vmax, alat } = fast;
  return { vmin, vmax, alat };
}

export type Mapping = ReturnType<typeof useMapping>;
