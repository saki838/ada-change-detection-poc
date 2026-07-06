import { useEffect, useState } from "react";
import { getRuns } from "../api.js";

const PAGE = 10;

function fmtDate(s) {
  if (!s) return "—";
  const d = new Date(s);
  return isNaN(d.getTime()) ? s : d.toLocaleString();
}

export default function RunList({ refreshKey, onSelectRun }) {
  const [runs, setRuns] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    getRuns(PAGE, offset)
      .then((data) => {
        if (cancelled) return;
        setRuns(Array.isArray(data.runs) ? data.runs : []);
        setTotal(data.total ?? 0);
      })
      .catch(() => {
        if (!cancelled) setError("failed to load runs");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey, offset]);

  function select(run) {
    // Normalize to the shape MapView expects (run_id). Runs may carry `id` or `run_id`.
    const runId = run.run_id ?? run.id;
    onSelectRun?.({ ...run, run_id: runId });
  }

  const canPrev = offset > 0;
  const canNext = offset + PAGE < total;

  return (
    <div className="card runs-card">
      <div className="runs-head">
        <h2>Past runs</h2>
        <span className="muted">{total} total</span>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>name</th>
              <th>status</th>
              <th>mode</th>
              <th>detections</th>
              <th>total area (m²)</th>
              <th>created</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 && !loading && (
              <tr>
                <td colSpan={6} className="muted">
                  No runs yet.
                </td>
              </tr>
            )}
            {runs.map((r) => {
              const id = r.run_id ?? r.id;
              return (
                <tr key={id} className="run-row" onClick={() => select(r)} title="Load on map">
                  <td>{r.name || `run #${id}`}</td>
                  <td>
                    <span className={`status status-${r.status}`}>{r.status}</span>
                  </td>
                  <td>{r.mode}</td>
                  <td>{r.num_detections ?? 0}</td>
                  <td>{r.total_area_m2 != null ? Number(r.total_area_m2).toFixed(1) : "—"}</td>
                  <td>{fmtDate(r.created_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <button disabled={!canPrev} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
          ← Prev
        </button>
        <span className="muted">
          {total === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE, total)}
        </span>
        <button disabled={!canNext} onClick={() => setOffset(offset + PAGE)}>
          Next →
        </button>
      </div>
    </div>
  );
}
