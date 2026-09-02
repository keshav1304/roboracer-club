"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useGateway } from "@/lib/useGateway";
import { useMapping } from "@/lib/useMapping";
import {
  ALGORITHMS,
  algorithmSpec,
  algorithmsInGroup,
} from "@/lib/algorithms";
import StatusTab from "@/components/StatusTab";
import ParamsPanel from "@/components/ParamsPanel";
import LidarCanvas from "@/components/LidarCanvas";
import RobotStatus from "@/components/RobotStatus";
import StripChart from "@/components/StripChart";
import SetupModal from "@/components/SetupModal";
import MappingPanel from "@/components/mapping/MappingPanel";
import MappingCanvas from "@/components/mapping/MappingCanvas";

const DEFAULT_GATEWAY =
  process.env.NEXT_PUBLIC_ROS_GATEWAY_URL ?? "ws://localhost:8000/ws";
const SIM_ADDRESS =
  process.env.NEXT_PUBLIC_SIM_BRIDGE_ADDRESS ?? "127.0.0.1 : 4567";

type LeftTab = "reactive" | "local" | "mapping";

export default function Page() {
  const [gatewayUrl, setGatewayUrl] = useState(DEFAULT_GATEWAY);
  const [token, setToken] = useState("");
  const [tab, setTab] = useState<LeftTab>("reactive");
  const [showSetup, setShowSetup] = useState(false);
  const [selectedAlgo, setSelectedAlgo] = useState(ALGORITHMS[0].id);
  const [chartHistory, setChartHistory] = useState<number[]>([]);
  const lastChartAlgo = useRef<string | null>(null);

  // Restore saved settings; show setup once on first visit.
  useEffect(() => {
    const savedUrl = localStorage.getItem("gatewayUrl");
    const savedToken = localStorage.getItem("gatewayToken");
    if (savedUrl) setGatewayUrl(savedUrl);
    if (savedToken) setToken(savedToken);
    if (!localStorage.getItem("setupSeen")) {
      setShowSetup(true);
      localStorage.setItem("setupSeen", "1");
    }
  }, []);

  const gw = useGateway(gatewayUrl, token);
  const {
    status,
    telemetry,
    lidar,
    trail,
    lastAck,
    sessionEpoch,
    setParams,
    startAlgo,
    stopAlgo,
    reset,
    clearTrail,
  } = gw;

  const lab = useMapping(gw, gatewayUrl, token);

  const onGatewayUrlChange = useCallback((url: string) => {
    setGatewayUrl(url);
    localStorage.setItem("gatewayUrl", url);
  }, []);

  const onTokenChange = useCallback((t: string) => {
    setToken(t);
    localStorage.setItem("gatewayToken", t);
  }, []);

  const running = status.algorithm != null;

  // While an algorithm runs, show that one; otherwise preview the selection.
  const activeSpec = algorithmSpec(status.algorithm ?? selectedAlgo);

  // Feed the algorithm's strip chart from algo_debug; reset on algo change
  // or when the sim/gateway session is wiped.
  const chart = activeSpec.chart;
  useEffect(() => {
    setChartHistory([]);
    lastChartAlgo.current = status.algorithm;
  }, [sessionEpoch]);

  useEffect(() => {
    const algo = status.algorithm;
    if (algo !== lastChartAlgo.current) {
      lastChartAlgo.current = algo;
      setChartHistory([]);
    }
    if (!chart || !status.simConnected) return;
    const debug = telemetry.algo_debug;
    if (!debug || debug.algorithm !== activeSpec.id) return;
    const raw = debug[chart.debugKey];
    if (typeof raw !== "number") return;
    const sample = raw * (chart.scale ?? 1);
    setChartHistory((prev) => {
      const next = [...prev, sample];
      return next.length > 200 ? next.slice(-200) : next;
    });
  }, [
    telemetry.algo_debug,
    status.algorithm,
    status.simConnected,
    chart,
    activeSpec.id,
  ]);

  const reactiveAlgos = algorithmsInGroup("reactive");
  const localAlgos = algorithmsInGroup("local");

  // Keep the dropdown selection in the active tab's group.
  useEffect(() => {
    if (tab === "mapping") return;
    const group = tab === "reactive" ? "reactive" : "local";
    const list = algorithmsInGroup(group);
    if (!list.some((a) => a.id === selectedAlgo)) {
      setSelectedAlgo(list[0]?.id ?? ALGORITHMS[0].id);
    }
  }, [tab, selectedAlgo]);

  const renderAlgoPanel = (algos: typeof reactiveAlgos) => {
    const selectValue =
      status.algorithm && algos.some((a) => a.id === status.algorithm)
        ? status.algorithm
        : status.algorithm && !algos.some((a) => a.id === status.algorithm)
          ? status.algorithm
          : selectedAlgo;
    const runningOutsideGroup =
      status.algorithm != null &&
      !algos.some((a) => a.id === status.algorithm);

    return (
      <div>
        <div className="field-label" style={{ marginTop: 0 }}>
          Select algorithm
        </div>
        <select
          className="select"
          value={selectValue}
          disabled={running}
          onChange={(e) => setSelectedAlgo(e.target.value)}
        >
          {algos.map((a) => (
            <option key={a.id} value={a.id}>
              {a.label}
            </option>
          ))}
          {runningOutsideGroup && status.algorithm && (
            <option value={status.algorithm}>
              {algorithmSpec(status.algorithm).label}
            </option>
          )}
        </select>
        <div className="btn-row" style={{ marginTop: 10 }}>
          {!running ? (
            <button
              className="btn primary"
              disabled={!status.wsConnected || !status.simConnected}
              onClick={() => startAlgo(selectedAlgo)}
            >
              Start
            </button>
          ) : (
            <button className="btn danger" onClick={stopAlgo}>
              Stop
            </button>
          )}
          <button
            className="btn"
            disabled={!status.wsConnected || !status.simConnected}
            onClick={reset}
          >
            Reset
          </button>
        </div>
        <ParamsPanel
          algorithm={activeSpec}
          onSetParams={setParams}
          lastAck={lastAck}
          disabled={!running || !status.wsConnected}
        />
      </div>
    );
  };

  return (
    <div className="app">
      <header className="header">
        <h1 className="header-brand">
          <img
            className="header-logo"
            src="/roboracer-logo.png"
            alt=""
            width={42}
            height={28}
          />
          RoboRacer Playground
        </h1>
        <div className="header-actions">
          <button className="btn small" onClick={() => setShowSetup(true)}>
            Setup
          </button>
          <input
            className="text"
            style={{ width: 150 }}
            type="password"
            placeholder="Token"
            value={token}
            onChange={(e) => onTokenChange(e.target.value)}
            spellCheck={false}
          />
        </div>
      </header>

      <aside className="left panel">
        <div className="tabs">
          <button
            className={`tab ${tab === "reactive" ? "active" : ""}`}
            onClick={() => setTab("reactive")}
          >
            Reactive
          </button>
          <button
            className={`tab ${tab === "local" ? "active" : ""}`}
            onClick={() => setTab("local")}
          >
            Local planning
          </button>
          <button
            className={`tab ${tab === "mapping" ? "active" : ""}`}
            onClick={() => setTab("mapping")}
          >
            Map-based
          </button>
        </div>

        {tab === "reactive"
          ? renderAlgoPanel(reactiveAlgos)
          : tab === "local"
            ? renderAlgoPanel(localAlgos)
            : (
              <MappingPanel gw={gw} lab={lab} />
            )}
      </aside>

      <main className="main">
        {tab === "mapping" ? (
          <MappingCanvas gw={gw} lab={lab} />
        ) : (
          <LidarCanvas
            lidar={lidar}
            telemetry={telemetry}
            trail={trail}
            vizFeatures={activeSpec.vizFeatures}
            onClearTrail={clearTrail}
            bottomExtra={
              chart ? (
                <StripChart
                  values={chartHistory}
                  label={chart.label}
                  subtitle={chart.subtitle}
                  unit={chart.unit}
                  warnAbs={chart.warnAbs}
                  minRange={chart.minRange}
                  decimals={chart.decimals}
                />
              ) : undefined
            }
          />
        )}
      </main>

      <aside className="right">
        <div
          className="panel"
          style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}
        >
          <div className="field-label" style={{ marginTop: 0 }}>
            Vehicle Status
          </div>
          <RobotStatus
            telemetry={telemetry}
            metrics={activeSpec.statusMetrics}
          />
        </div>
        <div
          className="panel"
          style={{ flex: "0 0 auto", maxHeight: "42%", overflow: "auto" }}
        >
          <div className="field-label" style={{ marginTop: 0 }}>
            Connection
          </div>
          <StatusTab
            status={status}
            gatewayUrl={gatewayUrl}
            simAddress={SIM_ADDRESS}
            onGatewayUrlChange={onGatewayUrlChange}
            onShowSetup={() => setShowSetup(true)}
          />
        </div>
      </aside>

      {showSetup && (
        <SetupModal
          simAddress={SIM_ADDRESS}
          onClose={() => setShowSetup(false)}
        />
      )}
    </div>
  );
}
