"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { decodeOccB64 } from "@/lib/occupancyMap";

export interface AlgoDebug {
  algorithm: string;
  /** Path points [[x,y], ...] — vehicle frame for RRT; map frame for MPC. */
  path?: number[][];
  /** Tree edges [[x1,y1,x2,y2,cost], ...] in the vehicle frame. */
  tree?: number[][];
  /** Raw LiDAR occupancy cell centers [[x,y], ...] (vehicle frame). */
  occ_hits?: number[][];
  /** Inflated occupancy cell centers [[x,y], ...] (vehicle frame). */
  occ_inflated?: number[][];
  [key: string]: string | number | number[][] | undefined;
}

export interface PoseError {
  xy_m: number;
  yaw_rad: number;
  yaw_deg: number;
  true_map: { x: number; y: number; yaw: number };
  amcl: { x: number; y: number; yaw: number };
}

export interface Telemetry {
  /** Gateway sample time (monotonic seconds); used to pace charts. */
  t?: number;
  position: [number, number, number];
  yaw: number;
  /** True speed from simulator odometry twist (m/s). */
  speed: number;
  throttle: number;
  steering: number;
  lap_count: number;
  lap_time: number;
  best_lap_time: number;
  collision_count: number;
  algo_debug: AlgoDebug | null;
  /** Present while the particle filter is publishing; true odom vs estimate in map frame. */
  pose_error?: PoseError | null;
}

export interface LidarFrame {
  angle_min: number;
  angle_increment: number;
  range_max: number;
  ranges: number[];
}

export interface SlamStatus {
  active: boolean;
  drive: string | null;
  auto_stop: boolean;
  lap_done: boolean;
  started_at: number | null;
}

export interface LocalizeStatus {
  active: boolean;
  map: string | null;
  seeded: boolean;
  started_at: number | null;
  pose_fresh: boolean;
  localized: boolean;
  pose: {
    x: number;
    y: number;
    yaw: number;
    var_x: number;
    var_y: number;
    var_yaw: number;
  } | null;
  age_s?: number;
}

export interface GatewayStatus {
  wsConnected: boolean;
  simConnected: boolean;
  algorithm: string | null;
  raceline: string | null;
  slam: SlamStatus;
  localize: LocalizeStatus;
  optRunning: boolean;
}

export interface ParamsAck {
  ok: boolean;
  reason?: string;
  applied?: Record<string, number>;
}

export interface Ack {
  ok: boolean;
  op?: string;
  reason?: string;
  /** monotonically increasing local id so consumers see repeats */
  n: number;
  [key: string]: unknown;
}

export interface Transform2D {
  tx: number;
  ty: number;
  yaw: number;
  /** "map_to_world" for current saves; omitted on legacy world→map blobs. */
  frame?: string;
}

/** Live SLAM occupancy grid frame streamed during mapping. */
export interface MapFrame {
  /** ROS OccupancyGrid values: -1 unknown, 0 free, 1–100 occupied. */
  occupancy: Int8Array;
  width: number;
  height: number;
  resolution: number;
  origin: [number, number];
  map_to_world: Transform2D | null;
}

export interface RacelineData {
  ok: boolean;
  reason?: string;
  req_id?: number;
  s: number[];
  x: number[];
  y: number[];
  theta: number[];
  v: number[];
  clearance?: number[];
  min_clearance_m?: number;
  violations?: number;
  length_m: number;
  lap_time_est: number;
  margin: number;
}

export interface OptEvent {
  seq: number;
  /** Envelope is always opt_progress; phase is in `event`. */
  type?: string;
  event: "started" | "progress" | "done" | "error" | "cancelled";
  map?: string;
  evals?: number;
  budget?: number;
  best_cost?: number;
  preview?: { x: number[]; y: number[] } | null;
  raceline?: RacelineData;
  anchors?: [number, number][];
  params?: Record<string, number>;
  reason?: string;
}

export interface MapUpdatedEvent {
  name: string;
  version: number;
  undo: number;
  redo: number;
  n: number;
}

const DEFAULT_TELEMETRY: Telemetry = {
  position: [0, 0, 0],
  yaw: 0,
  speed: 0,
  throttle: 0,
  steering: 0,
  lap_count: 0,
  lap_time: 0,
  best_lap_time: 0,
  collision_count: 0,
  algo_debug: null,
};

const DEFAULT_SLAM: SlamStatus = {
  active: false,
  drive: null,
  auto_stop: false,
  lap_done: false,
  started_at: null,
};

const DEFAULT_LOCALIZE: LocalizeStatus = {
  active: false,
  map: null,
  seeded: false,
  started_at: null,
  pose_fresh: false,
  localized: false,
  pose: null,
};

const DEFAULT_STATUS: GatewayStatus = {
  wsConnected: false,
  simConnected: false,
  algorithm: null,
  raceline: null,
  slam: DEFAULT_SLAM,
  localize: DEFAULT_LOCALIZE,
  optRunning: false,
};

/** Derive the REST base (http[s]://host) from the gateway WS URL. */
export function httpBaseFromWs(wsUrl: string): string {
  try {
    const u = new URL(wsUrl);
    const proto = u.protocol === "wss:" ? "https:" : "http:";
    return `${proto}//${u.host}`;
  } catch {
    return "http://localhost:8000";
  }
}

export function useGateway(url: string, token: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<GatewayStatus>(DEFAULT_STATUS);
  const [telemetry, setTelemetry] = useState<Telemetry>(DEFAULT_TELEMETRY);
  const [lidar, setLidar] = useState<LidarFrame | null>(null);
  const [lastAck, setLastAck] = useState<ParamsAck | null>(null);
  const [trail, setTrail] = useState<[number, number][]>([]);
  const [sessionEpoch, setSessionEpoch] = useState(0);

  // Mapping streams / events
  const [mapFrame, setMapFrame] = useState<MapFrame | null>(null);
  const [optEvent, setOptEvent] = useState<OptEvent | null>(null);
  const [racelineData, setRacelineData] = useState<RacelineData | null>(null);
  const [slamAck, setSlamAck] = useState<Ack | null>(null);
  const [mapAck, setMapAck] = useState<Ack | null>(null);
  const [racelineAck, setRacelineAck] = useState<Ack | null>(null);
  const [algoAck, setAlgoAck] = useState<Ack | null>(null);
  const [mapUpdated, setMapUpdated] = useState<MapUpdatedEvent | null>(null);
  const [mapsRev, setMapsRev] = useState(0);
  const [racelinesRev, setRacelinesRev] = useState(0);

  const trailRef = useRef<[number, number][]>([]);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const simWasConnected = useRef(false);
  const ackCounter = useRef(0);

  const clearFrontendSession = useCallback(() => {
    trailRef.current = [];
    setTrail([]);
    setTelemetry(DEFAULT_TELEMETRY);
    setLidar(null);
    setLastAck(null);
    setSessionEpoch((n) => n + 1);
  }, []);

  const connect = useCallback(() => {
    if (!url) return;
    try {
      wsRef.current?.close();
    } catch {
      /* ignore */
    }
    const full = token ? `${url}?token=${encodeURIComponent(token)}` : url;
    let ws: WebSocket;
    try {
      ws = new WebSocket(full);
    } catch {
      setStatus((s) => ({ ...s, wsConnected: false }));
      return;
    }
    wsRef.current = ws;

    const stamp = <T extends Record<string, unknown>>(msg: T): Ack => ({
      ok: Boolean(msg.ok),
      ...msg,
      n: ++ackCounter.current,
    });

    ws.onopen = () => setStatus((s) => ({ ...s, wsConnected: true }));
    ws.onclose = () => {
      // Stale close from a replaced socket must not wipe the new session.
      if (wsRef.current !== ws) return;
      simWasConnected.current = false;
      setStatus({ ...DEFAULT_STATUS });
      clearFrontendSession();
      setMapFrame(null);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      reconnectTimer.current = setTimeout(() => {
        if (wsRef.current === ws) connect();
      }, 2000);
    };
    ws.onmessage = (ev) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(ev.data as string);
      } catch {
        return;
      }
      switch (msg.type) {
        case "telemetry": {
          // Per-frame flag (preferred): unlock odom as soon as the sim is live,
          // without waiting for the 1 Hz status tick. Never paint DEFAULT zeros
          // here — that pinned the green triangle at the origin on startup.
          if (typeof msg.sim_connected === "boolean") {
            const connected = Boolean(msg.sim_connected);
            if (connected !== simWasConnected.current) {
              const wasConnected = simWasConnected.current;
              simWasConnected.current = connected;
              setStatus((s) => ({ ...s, simConnected: connected }));
              if (wasConnected && !connected) {
                clearFrontendSession();
              }
            }
          }
          if (!simWasConnected.current) {
            break;
          }
          const t = msg as unknown as Telemetry;
          setTelemetry(t);
          const [x, y] = t.position;
          const prev = trailRef.current;
          const last = prev[prev.length - 1];
          // ≥0.05 m spacing, 4000-point cap ≈ 200 m of path — enough for a
          // full lap; smaller caps make the path visibly erase itself mid-lap.
          if (!last || Math.hypot(x - last[0], y - last[1]) > 0.05) {
            const next = [...prev, [x, y] as [number, number]].slice(-4000);
            trailRef.current = next;
            setTrail(next);
          }
          break;
        }
        case "lidar":
          if (simWasConnected.current) {
            setLidar(msg as unknown as LidarFrame);
          }
          break;
        case "status": {
          const simConnected = Boolean(msg.sim_connected);
          const wasConnected = simWasConnected.current;
          simWasConnected.current = simConnected;
          setStatus((s) => ({
            ...s,
            simConnected,
            algorithm: (msg.algorithm as string) ?? null,
            raceline: (msg.raceline as string) ?? null,
            slam: (msg.slam as SlamStatus) ?? DEFAULT_SLAM,
            localize: {
              ...DEFAULT_LOCALIZE,
              ...((msg.localize as LocalizeStatus) ?? {}),
            },
            optRunning: Boolean(msg.opt_running),
          }));
          // Backend signals a full wipe, or we observe the disconnect edge.
          if (msg.session_reset || (wasConnected && !simConnected)) {
            clearFrontendSession();
          }
          break;
        }
        case "params_ack":
          setLastAck(msg as unknown as ParamsAck);
          break;
        case "map_frame":
          if (msg.cleared) {
            setMapFrame(null);
          } else if (typeof msg.occ_b64 === "string") {
            try {
              const occupancy = decodeOccB64(msg.occ_b64);
              const origin = msg.origin as number[] | undefined;
              setMapFrame({
                occupancy,
                width: Number(msg.width),
                height: Number(msg.height),
                resolution: Number(msg.resolution),
                origin: [Number(origin?.[0] ?? 0), Number(origin?.[1] ?? 0)],
                map_to_world: (msg.map_to_world as Transform2D) ?? null,
              });
            } catch {
              /* malformed frame — ignore */
            }
          }
          break;
        case "opt_progress":
          setOptEvent(msg as unknown as OptEvent);
          break;
        case "raceline_data":
          setRacelineData(msg as unknown as RacelineData);
          break;
        case "slam_ack":
          setSlamAck(stamp(msg));
          if (msg.op === "save" && msg.ok) setMapsRev((n) => n + 1);
          break;
        case "map_ack":
          setMapAck(stamp(msg));
          break;
        case "raceline_ack":
          setRacelineAck(stamp(msg));
          if (msg.op === "save" && msg.ok) setRacelinesRev((n) => n + 1);
          break;
        case "algo_ack":
          setAlgoAck(stamp(msg));
          break;
        case "opt_ack":
          if (!msg.ok) setOptEvent({
            seq: -1,
            event: "error",
            reason: String(msg.reason ?? "Optimization failed to start"),
          });
          break;
        case "map_updated":
          setMapUpdated({
            name: String(msg.name),
            version: Number(msg.version),
            undo: Number(msg.undo ?? 0),
            redo: Number(msg.redo ?? 0),
            n: ++ackCounter.current,
          });
          setMapsRev((n) => n + 1);
          break;
        case "maps_changed":
          setMapsRev((n) => n + 1);
          break;
        case "racelines_changed":
          setRacelinesRev((n) => n + 1);
          break;
        default:
          break;
      }
    };
  }, [url, token, clearFrontendSession]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      const ws = wsRef.current;
      wsRef.current = null;
      ws?.close();
    };
  }, [connect]);

  const send = useCallback((obj: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }, []);

  const setParams = useCallback(
    (params: Record<string, number>) => send({ type: "set_params", params }),
    [send]
  );
  const startAlgo = useCallback(
    (algorithm: string, raceline?: string) =>
      send({ type: "start_algo", algorithm, raceline }),
    [send]
  );
  const stopAlgo = useCallback(() => send({ type: "stop_algo" }), [send]);
  const reset = useCallback(() => {
    trailRef.current = [];
    setTrail([]);
    send({ type: "reset" });
  }, [send]);

  const clearTrail = useCallback(() => {
    trailRef.current = [];
    setTrail([]);
  }, []);

  // --- Mapping commands ---
  const slamStart = useCallback(
    (drive: "manual" | "wall_follow", params?: Record<string, number>) =>
      send({ type: "slam_start", drive, auto_stop: false, params }),
    [send]
  );
  const slamStop = useCallback(() => send({ type: "slam_stop" }), [send]);
  const slamSaveMap = useCallback(
    (name: string) => send({ type: "slam_save_map", name }),
    [send]
  );
  const mapEdit = useCallback(
    (name: string, ops: { tool: string; points: number[][]; radius: number }[]) =>
      send({ type: "map_edit", name, ops }),
    [send]
  );
  const mapUndo = useCallback(
    (name: string) => send({ type: "map_undo", name }),
    [send]
  );
  const mapRedo = useCallback(
    (name: string) => send({ type: "map_redo", name }),
    [send]
  );
  const mapDelete = useCallback(
    (name: string) => send({ type: "map_delete", name }),
    [send]
  );
  const racelineUpdate = useCallback(
    (map: string, anchors: [number, number][],
     params: Record<string, number>, reqId: number) =>
      send({ type: "raceline_update", map, anchors, params, req_id: reqId }),
    [send]
  );
  const racelineSave = useCallback(
    (name: string, map: string, data: RacelineData,
     params: Record<string, number>, anchors: [number, number][]) =>
      send({
        type: "raceline_save",
        name,
        map,
        data: { s: data.s, x: data.x, y: data.y, theta: data.theta, v: data.v },
        params,
        anchors,
      }),
    [send]
  );
  const racelineDelete = useCallback(
    (name: string) => send({ type: "raceline_delete", name }),
    [send]
  );
  const optStart = useCallback(
    (map: string, params: Record<string, number>) =>
      send({ type: "opt_start", map, params }),
    [send]
  );
  const optCancel = useCallback(() => send({ type: "opt_cancel" }), [send]);

  return {
    status,
    telemetry,
    lidar,
    trail,
    lastAck,
    sessionEpoch,
    mapFrame,
    optEvent,
    racelineData,
    slamAck,
    mapAck,
    racelineAck,
    algoAck,
    mapUpdated,
    mapsRev,
    racelinesRev,
    setParams,
    startAlgo,
    stopAlgo,
    reset,
    clearTrail,
    slamStart,
    slamStop,
    slamSaveMap,
    mapEdit,
    mapUndo,
    mapRedo,
    mapDelete,
    racelineUpdate,
    racelineSave,
    racelineDelete,
    optStart,
    optCancel,
  };
}

export type Gateway = ReturnType<typeof useGateway>;
