import { useState } from "react";
import { login as apiLogin, me as apiMe } from "../api.js";
import { useAuth } from "../auth.js";

export default function Login() {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const { access_token } = await apiLogin(username, password);
      // Token must be persisted before /auth/me so the interceptor can attach it.
      login(access_token, null);
      const user = await apiMe();
      login(access_token, user);
    } catch (err) {
      if (err?.response?.status === 401) setError("invalid credentials");
      else setError(err?.response?.data?.detail || "login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="login-wrap">
      <form className="card login-card" onSubmit={onSubmit}>
        <h1>ADA Change Detection</h1>
        <p className="muted">Sign in to run encroachment detection.</p>
        <label>
          Username
          <input
            type="text"
            value={username}
            autoComplete="username"
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            autoComplete="current-password"
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        {error && <div className="error">{error}</div>}
        <button type="submit" disabled={submitting}>
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
