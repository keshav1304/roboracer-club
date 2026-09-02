"use client";

import { useState } from "react";
import type { GatewayStatus } from "@/lib/useGateway";
import { algorithmSpec } from "@/lib/algorithms";

interface Props {
  status: GatewayStatus;
  gatewayUrl: string;
  simAddress: string;
  onGatewayUrlChange: (url: string) => void;
  onShowSetup: () => void;
}

export default function StatusTab({
  status,
  gatewayUrl,
  simAddress,
  onGatewayUrlChange,
  onShowSetup,
}: Props) {
  const [editUrl, setEditUrl] = useState(gatewayUrl);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(simAddress);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  const algorithmLabel = status.algorithm
    ? algorithmSpec(status.algorithm).label
    : "Stopped";

  return (
    <div>
      <div className="status-rows">
        <div className="status-row">
          <span className={`dot ${status.wsConnected ? "on" : ""}`} />
          Gateway
          <span className="hint">
            {status.wsConnected ? "Connected" : "Disconnected"}
          </span>
        </div>
        <div className="status-row">
          <span className={`dot ${status.simConnected ? "on" : ""}`} />
          Simulator
          <span className="hint">
            {status.simConnected ? "Streaming" : "Not connected"}
          </span>
        </div>
        <div className="status-row">
          <span className={`dot ${status.algorithm ? "on" : ""}`} />
          Algorithm
          <span className="hint">{algorithmLabel}</span>
        </div>
      </div>

      <div className="field-label">Gateway URL</div>
      <input
        className="text"
        value={editUrl}
        onChange={(e) => setEditUrl(e.target.value)}
        onBlur={() => onGatewayUrlChange(editUrl)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onGatewayUrlChange(editUrl);
        }}
        placeholder="ws://localhost:8000/ws"
        spellCheck={false}
      />

      <div className="field-label">Simulator address</div>
      <div className="copy-row">
        <span className="mono">{simAddress}</span>
        <button className="btn small" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <div style={{ marginTop: 16 }}>
        <button className="btn" style={{ width: "100%" }} onClick={onShowSetup}>
          Setup instructions
        </button>
      </div>
    </div>
  );
}
