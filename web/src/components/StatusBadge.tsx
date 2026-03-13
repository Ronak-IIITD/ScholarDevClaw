interface StatusBadgeProps {
  status: string;
  size?: "sm" | "md";
}

const COLORS: Record<string, string> = {
  idle: "bg-gray-700 text-gray-300",
  running: "bg-yellow-900/60 text-yellow-300",
  completed: "bg-green-900/60 text-green-300",
  failed: "bg-red-900/60 text-red-300",
};

export default function StatusBadge({ status, size = "sm" }: StatusBadgeProps) {
  const color = COLORS[status] ?? COLORS.idle;
  const sizeClass = size === "md" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium ${color} ${sizeClass}`}
    >
      {status === "running" && (
        <span className="h-1.5 w-1.5 rounded-full bg-yellow-400 animate-pulse-dot" />
      )}
      {status === "completed" && (
        <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
      )}
      {status === "failed" && (
        <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
      )}
      {status}
    </span>
  );
}
