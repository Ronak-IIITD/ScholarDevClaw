interface ActionableErrorPanelProps {
  error: string | null;
}

function suggestionsFromError(error: string): string[] {
  const text = error.toLowerCase();
  if (text.includes("not found") || text.includes("repository")) {
    return [
      "Verify the repository path exists and is readable.",
      "Confirm the path is inside allowed server directories.",
    ];
  }
  if (text.includes("unauthorized") || text.includes("auth") || text.includes("token")) {
    return [
      "Check API/WebSocket token configuration.",
      "Refresh credentials and retry the run.",
    ];
  }
  if (text.includes("validate") || text.includes("benchmark")) {
    return [
      "Retry with Skip validation enabled to inspect generation output.",
      "Check local benchmark/runtime dependencies before re-running.",
    ];
  }
  return [
    "Review the failing step in the timeline.",
    "Retry the run with a narrower spec selection.",
  ];
}

export default function ActionableErrorPanel({ error }: ActionableErrorPanelProps) {
  if (!error) return null;

  const suggestions = suggestionsFromError(error);

  return (
    <section
      role="alert"
      className="rounded-xl border border-red-800 bg-red-950/30 p-4"
      aria-live="assertive"
    >
      <h3 className="text-sm font-semibold text-red-200">Actionable error</h3>
      <p className="mt-2 text-sm text-red-300">{error}</p>
      <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-red-200">
        {suggestions.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}
