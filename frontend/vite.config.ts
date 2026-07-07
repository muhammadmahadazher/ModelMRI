import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` on :5173 proxies API/WS to the Python backend on :5900.
// Build: emits into the Python package so `modelmri serve` ships the app.
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: {
    outDir: "../modelmri/static/app",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5900",
      "/ws": { target: "ws://127.0.0.1:5900", ws: true },
    },
  },
});
