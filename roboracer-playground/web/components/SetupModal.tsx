"use client";

import { useState } from "react";

interface Props {
  simAddress: string;
  onClose: () => void;
}

export default function SetupModal({ simAddress, onClose }: Props) {
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

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Connect the simulator</h2>
        <ol>
          <li>Launch AutoDRIVE Simulator (practice build).</li>
          <li>
            In the menu panel, set IP and port to
            <div className="copy-row" style={{ marginTop: 6 }}>
              <span className="mono">{simAddress}</span>
              <button className="btn small" onClick={copy}>
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
          </li>
          <li>
            Press <span className="mono">Connection</span>. The simulator badge
            here turns green when data flows.
          </li>
          <li>
            Start the algorithm in this app, then set the sim to{" "}
            <span className="mono">Autonomous</span> driving mode.
          </li>
        </ol>
        <div className="note">
          If the connection fails on campus Wi-Fi, port 4567 may be blocked —
          try a hotspot or run the backend locally.
        </div>
        <div className="modal-actions">
          <button className="btn primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
