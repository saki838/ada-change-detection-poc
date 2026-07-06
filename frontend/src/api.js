import axios from "axios";
import { getToken, setToken } from "./auth.js";

// Single axios instance. baseURL "/api" is forwarded to the gateway (:8000) by the
// Vite dev proxy and by nginx in the prod build. The browser only ever talks to /api.
const client = axios.create({ baseURL: "/api" });

// Request interceptor: attach the bearer token when present.
client.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor: on 401 clear the token and bounce to login.
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err?.response?.status === 401) {
      setToken(null);
      // Avoid redirect loop if we're already handling a login submit failure.
      const url = err.config?.url || "";
      if (!url.includes("/auth/login")) {
        if (typeof window !== "undefined") window.location.assign("/");
      }
    }
    return Promise.reject(err);
  }
);

// POST /api/auth/login -> {access_token, token_type, expires_in}
export async function login(username, password) {
  const { data } = await client.post("/auth/login", { username, password });
  return data;
}

// GET /api/auth/me -> {id, username, role}
export async function me() {
  const { data } = await client.get("/auth/me");
  return data;
}

// POST /api/detect (multipart) -> 201 {run_id, status, mode, num_detections,
//   total_area_m2, mask_png_url, detections_url, created_at}
export async function detect({ t1, t2, pixelSizeM, mode = "ml", name } = {}) {
  const form = new FormData();
  form.append("t1", t1);
  form.append("t2", t2);
  if (mode) form.append("mode", mode);
  // Only send pixel_size_m when the user overrides the default (10.0).
  if (pixelSizeM !== undefined && pixelSizeM !== null && pixelSizeM !== "") {
    form.append("pixel_size_m", String(pixelSizeM));
  }
  if (name) form.append("name", name);
  // Let axios set the multipart boundary.
  const { data } = await client.post("/detect", form);
  return data;
}

// GET /api/runs?limit=&offset= -> {runs:[...], total}
export async function getRuns(limit = 50, offset = 0) {
  const { data } = await client.get("/runs", { params: { limit, offset } });
  return data;
}
// Alias to match the requested `listRuns` name.
export const listRuns = getRuns;

// GET /api/runs/{runId}/detections -> GeoJSON FeatureCollection
export async function getDetections(runId) {
  const { data } = await client.get(`/runs/${runId}/detections`);
  return data;
}

// Relative mask URL. The mask needs the bearer header, so callers fetch it as a blob
// (see maskBlobUrl) rather than dropping this straight into <img src>.
export function maskUrl(runId) {
  return `/api/runs/${runId}/mask.png`;
}
// Alias to match the requested `getMaskUrl` name.
export const getMaskUrl = maskUrl;

// Fetch the mask PNG (auth-required) as a blob and return an object URL for overlays.
export async function maskBlobUrl(runId) {
  const res = await client.get(`/runs/${runId}/mask.png`, { responseType: "blob" });
  return URL.createObjectURL(res.data);
}

export default client;
