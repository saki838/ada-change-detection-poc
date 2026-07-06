import { useState } from "react";
import { detect } from "../api.js";

const ACCEPT = "image/tiff,image/png,image/jpeg,.tif,.tiff";

export default function UploadPair({ onComplete }) {
  const [t1File, setT1File] = useState(null);
  const [t2File, setT2File] = useState(null);
  const [mode, setMode] = useState("ml");
  const [pixelSizeM, setPixelSizeM] = useState(10.0);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const missing = !t1File || !t2File;

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    if (missing) {
      setError("t1/t2 required");
      return;
    }
    setBusy(true);
    try {
      const res = await detect({
        t1: t1File,
        t2: t2File,
        mode,
        // Only override the server default when the user changed it.
        pixelSizeM: Number(pixelSizeM) === 10.0 ? undefined : pixelSizeM,
        name: name || undefined
      });
      if (onComplete) onComplete(res);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(detail || "detection failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card upload-card" onSubmit={onSubmit}>
      <h2>New detection</h2>

      <label>
        T1 (before)
        <input type="file" accept={ACCEPT} onChange={(e) => setT1File(e.target.files?.[0] || null)} />
      </label>
      <label>
        T2 (after)
        <input type="file" accept={ACCEPT} onChange={(e) => setT2File(e.target.files?.[0] || null)} />
      </label>

      <div className="row">
        <label>
          Mode
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="ml">ml</option>
            <option value="diff">diff</option>
          </select>
        </label>
        <label>
          Pixel size (m)
          <input
            type="number"
            step="0.1"
            min="0"
            value={pixelSizeM}
            onChange={(e) => setPixelSizeM(e.target.value)}
          />
        </label>
      </div>

      <label>
        Name (optional)
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="run label" />
      </label>

      {error && <div className="error">{error}</div>}

      <button type="submit" disabled={busy || missing}>
        {busy ? "Detecting…" : "Run detection"}
      </button>
    </form>
  );
}
