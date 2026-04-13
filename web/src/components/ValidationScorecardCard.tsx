type Dict = Record<string, unknown>;

interface ValidationScorecardCardProps {
  latestValidateStepData: Dict | null;
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export default function ValidationScorecardCard({
  latestValidateStepData,
}: ValidationScorecardCardProps) {
  if (!latestValidateStepData) {
    return (
      <section className="rounded-xl border border-gray-800 bg-gray-900/70 p-4">
        <h3 className="text-sm font-semibold text-gray-200">Validation scorecard</h3>
        <p className="mt-3 text-sm text-gray-500">No validation data yet.</p>
      </section>
    );
  }

  const scorecard =
    (latestValidateStepData.scorecard as Dict | undefined) ?? latestValidateStepData;
  const summary = String(scorecard.summary ?? latestValidateStepData.summary ?? "unknown");
  const stage = String(latestValidateStepData.stage ?? scorecard.stage ?? "unknown");
  const passed =
    typeof latestValidateStepData.passed === "boolean"
      ? latestValidateStepData.passed
      : summary === "pass";
  const deltas = (scorecard.deltas as Dict | undefined) ?? {};
  const checks = asList(scorecard.checks ?? latestValidateStepData.checks).slice(0, 5);
  const highlights = asList(scorecard.highlights ?? latestValidateStepData.highlights).slice(0, 4);

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-900/70 p-4">
      <h3 className="text-sm font-semibold text-gray-200">Validation scorecard</h3>
      <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-xs text-gray-500">Summary</p>
          <p className="font-medium text-gray-200">{summary}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Stage</p>
          <p className="font-medium text-gray-200">{stage}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Passed</p>
          <p className={`font-medium ${passed ? "text-green-300" : "text-red-300"}`}>
            {passed ? "yes" : "no"}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Speedup</p>
          <p className="font-medium text-gray-200">
            {typeof deltas.speedup === "number" ? `${deltas.speedup.toFixed(3)}x` : "n/a"}
          </p>
        </div>
      </div>

      <div className="mt-3 text-sm">
        <p className="text-xs text-gray-500">Loss change</p>
        <p className="font-medium text-gray-200">
          {typeof deltas.loss_change_pct === "number"
            ? `${deltas.loss_change_pct.toFixed(3)}%`
            : "n/a"}
        </p>
      </div>

      {checks.length > 0 && (
        <div className="mt-4">
          <p className="text-xs text-gray-500">Checks</p>
          <ul className="mt-1 space-y-1 text-sm text-gray-300">
            {checks.map((check, i) => {
              const item = (check as Dict) ?? {};
              return (
                <li key={`check-${i}`}>
                  {String(item.name ?? "check")} • {String(item.status ?? "unknown")}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {highlights.length > 0 && (
        <div className="mt-4">
          <p className="text-xs text-gray-500">Highlights</p>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-gray-300">
            {highlights.map((highlight, i) => (
              <li key={`hl-${i}`}>{String(highlight)}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
