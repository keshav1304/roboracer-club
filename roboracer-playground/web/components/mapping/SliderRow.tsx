"use client";

interface Props {
  label: string;
  info?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  disabled?: boolean;
  decimals?: number;
  onChange: (v: number) => void;
}

/** Labeled slider matching the ParamsPanel visual language. */
export default function SliderRow({
  label,
  info,
  min,
  max,
  step,
  value,
  disabled,
  decimals,
  onChange,
}: Props) {
  const d = decimals ?? (step < 0.01 ? 3 : step < 1 ? 2 : 0);
  return (
    <div className="slider-group">
      <div className="slider-label">
        <span className="slider-name">
          {label}
          {info && (
            <span className="info-tip" tabIndex={0} aria-label={info}>
              <span className="info-icon" aria-hidden>
                i
              </span>
              <span className="info-bubble" role="tooltip">
                {info}
              </span>
            </span>
          )}
        </span>
        <span className="val">{value.toFixed(d)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}
