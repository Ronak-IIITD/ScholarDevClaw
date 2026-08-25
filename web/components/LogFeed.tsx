"use client";

interface LogFeedProps {
  lines: string[];
  maxLines?: number;
  style?: React.CSSProperties;
}

export function LogFeed({ lines, maxLines = 200, style }: LogFeedProps) {
  const displayLines = lines.slice(-maxLines);

  return (
    <div className="log" role="log" aria-live="polite" aria-label="Pipeline log" style={style}>
      {displayLines.map((line, i) => {
        let className = "";
        if (line.includes("ERROR") || line.includes("Failed") || line.includes("failed")) className = "err";
        else if (line.includes("OK") || line.includes("passed") || line.includes("completed")) className = "ok";
        else if (line.includes("RUN") || line.includes("Running") || line.includes("Starting")) className = "run";
        else if (line.startsWith("[") && line.includes("]")) className = "t";

        return (
          <div key={i} className={className}>
            {line}
          </div>
        );
      })}
    </div>
  );
}