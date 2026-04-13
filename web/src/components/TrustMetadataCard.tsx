type Dict = Record<string, unknown>;

interface TrustMetadataCardProps {
  latestGenerateStepData: Dict | null;
  latestValidateStepData?: Dict | null;
}

function listFrom(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => String(v)).filter(Boolean);
}

export default function TrustMetadataCard({
  latestGenerateStepData,
  latestValidateStepData = null,
}: TrustMetadataCardProps) {
  if (!latestGenerateStepData && !latestValidateStepData) {
    return (
      <section className="rounded-xl border border-gray-800 bg-gray-900/70 p-4">
        <h3 className="text-sm font-semibold text-gray-200">Trust metadata</h3>
        <p className="mt-3 text-sm text-gray-500">No generation metadata available yet.</p>
      </section>
    );
  }

  const gen = latestGenerateStepData ?? {};
  const val = latestValidateStepData ?? {};
  const previewFiles = listFrom(gen.preview_files ?? gen.file_names).slice(0, 6);

  const diffFiles = listFrom(
    (val.diff_evidence as Dict | undefined)?.files_changed ??
      (val.diff_evidence as Dict | undefined)?.files_new
  ).slice(0, 4);
  const hunkPreview = listFrom(val.representative_hunks).slice(0, 3);

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-900/70 p-4">
      <h3 className="text-sm font-semibold text-gray-200">Trust metadata</h3>

      <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-xs text-gray-500">New files</p>
          <p className="font-medium text-gray-200">{String(gen.new_files ?? 0)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Transformations</p>
          <p className="font-medium text-gray-200">{String(gen.transformations ?? 0)}</p>
        </div>
      </div>

      {typeof gen.output_dir === "string" && gen.output_dir && (
        <div className="mt-3">
          <p className="text-xs text-gray-500">Output directory</p>
          <p className="truncate font-mono text-xs text-gray-300">{gen.output_dir}</p>
        </div>
      )}

      {previewFiles.length > 0 && (
        <div className="mt-4">
          <p className="text-xs text-gray-500">Preview files</p>
          <ul className="mt-1 space-y-1 text-xs text-gray-300">
            {previewFiles.map((file) => (
              <li key={file} className="truncate font-mono">
                {file}
              </li>
            ))}
          </ul>
        </div>
      )}

      {diffFiles.length > 0 && (
        <div className="mt-4">
          <p className="text-xs text-gray-500">Diff files</p>
          <ul className="mt-1 space-y-1 text-xs text-gray-300">
            {diffFiles.map((file) => (
              <li key={file} className="truncate font-mono">
                {file}
              </li>
            ))}
          </ul>
        </div>
      )}

      {hunkPreview.length > 0 && (
        <div className="mt-4">
          <p className="text-xs text-gray-500">Representative hunks</p>
          <ul className="mt-1 space-y-1 text-xs text-gray-300">
            {hunkPreview.map((hunk, i) => (
              <li key={`h-${i}`}>{hunk}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
