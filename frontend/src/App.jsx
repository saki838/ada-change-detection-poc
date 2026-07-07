import { useState } from "react";
import { useAuth } from "./auth.jsx";
import Login from "./components/Login.jsx";
import UploadPair from "./components/UploadPair.jsx";
import MapView from "./components/MapView.jsx";
import RunList from "./components/RunList.jsx";

export default function App() {
  const { user, logout } = useAuth();
  const [currentRun, setCurrentRun] = useState(null);
  const [runsRefreshKey, setRunsRefreshKey] = useState(0);

  if (!user) {
    return <Login />;
  }

  function handleComplete(detectResponse) {
    // detectResponse: {run_id, status, mode, num_detections, total_area_m2, ...}
    setCurrentRun(detectResponse);
    setRunsRefreshKey((k) => k + 1);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">ADA · Change Detection</div>
        <div className="header-right">
          <span className="muted">
            {user.username}
            {user.role ? ` (${user.role})` : ""}
          </span>
          <button className="ghost" onClick={logout}>
            Log out
          </button>
        </div>
      </header>

      <main className="app-main">
        <aside className="col-left">
          <UploadPair onComplete={handleComplete} />
        </aside>
        <section className="col-center">
          <MapView run={currentRun} />
        </section>
      </main>

      <section className="app-bottom">
        <RunList refreshKey={runsRefreshKey} onSelectRun={setCurrentRun} />
      </section>
    </div>
  );
}
