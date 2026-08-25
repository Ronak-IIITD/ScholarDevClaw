"use client";

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  unit?: string;
}

export function MetricCard({ label, value, sub, unit }: MetricCardProps) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">
        {value}
        {unit && <small> {unit}</small>}
      </div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}