import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev: `npm run dev` on :5173 proxies API/WS to the Python backend on :5900.
// Build: emits into the Python package so `modelmri serve` ships the app.
// VITE_DEMO=1 builds the static GitHub Pages demo (pre-baked responses,
// no backend); the default build is the app the Python package serves.
const demo = process.env.VITE_DEMO === "1";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: demo ? "./" : "/app/",
  // public/ holds only the baked demo payloads — never ship them inside the
  // pip package, which has a live backend and no use for recorded responses.
  publicDir: demo ? "public" : false,
  build: {
    outDir: demo ? "../demo-dist" : "../modelmri/static/app",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5900",
      "/ws": { target: "ws://127.0.0.1:5900", ws: true },
    },
  },
});
