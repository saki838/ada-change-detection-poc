import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: browser -> Vite (5173) -> gateway (8000). All app traffic goes to /api/*.
// Under docker-compose the gateway is reachable via service DNS `gateway`; for a bare
// `npm run dev` on the host set VITE_PROXY_TARGET=http://localhost:8000.
const target = process.env.VITE_PROXY_TARGET || "http://gateway:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target,
        changeOrigin: true
      }
    }
  }
});
