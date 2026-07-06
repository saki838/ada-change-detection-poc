import React from "react";
import { createRoot } from "react-dom/client";
import { AuthProvider } from "./auth.js";
import App from "./App.jsx";
import "leaflet/dist/leaflet.css";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>
);
