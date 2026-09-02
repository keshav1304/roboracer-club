"use client";

import { useState } from "react";

interface Props {
  title: string;
  description?: string;
  suggestion: string;
  existingNames: string[];
  busy: boolean;
  error: string | null;
  saveLabel?: string;
  onSave: (name: string) => void;
  onClose: () => void;
}

const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

/** Small modal asking for an artifact name (map / raceline). */
export default function NameModal({
  title,
  description,
  suggestion,
  existingNames,
  busy,
  error,
  saveLabel = "Save",
  onSave,
  onClose,
}: Props) {
  const [name, setName] = useState(suggestion);

  const valid = NAME_RE.test(name);
  const exists = existingNames.includes(name);

  return (
    <div className="modal-overlay" onClick={busy ? undefined : onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        {description && <div className="note" style={{ marginTop: 0, marginBottom: 12 }}>{description}</div>}
        <input
          className="text"
          value={name}
          autoFocus
          onChange={(e) =>
            setName(
              e.target.value
                .toLowerCase()
                .replace(/\s+/g, "-")
                .replace(/[^a-z0-9_-]/g, "")
            )
          }
          onKeyDown={(e) => {
            if (e.key === "Enter" && valid && !busy) onSave(name);
            if (e.key === "Escape" && !busy) onClose();
          }}
          placeholder={suggestion}
          spellCheck={false}
        />
        {!valid && name.length > 0 && (
          <div className="ack fail">
            Use lowercase letters, digits, “-” or “_”.
          </div>
        )}
        {exists && valid && (
          <div className="ack fail">
            “{name}” already exists. Saving will overwrite it.
          </div>
        )}
        {error && <div className="ack fail">{error}</div>}
        <div className="modal-actions" style={{ gap: 8 }}>
          <button className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="btn primary"
            disabled={!valid || busy}
            onClick={() => onSave(name)}
          >
            {busy ? "Saving…" : exists ? "Overwrite" : saveLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
