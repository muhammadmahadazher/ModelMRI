import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev: `npm run dev` on :5173 proxies API/WS to the Python backend on :5900.
//
// Three build targets, selected with `vite build --mode demo|viewer`:
//   (default)  the app the Python package serves at /app/
//   demo       the GitHub Pages tour — pre-baked real responses, no backend
//   viewer     the .mri reader — the same app with the API answered from a
//              file the user drops, so someone who was sent an analysis can
//              read it with nothing installed
//
// A mode rather than an inline environment variable, because
// `VITE_DEMO=1 npm run build` is not valid syntax in PowerShell or cmd — so
// that form only ever worked on a POSIX shell, and the `build:demo` script
// the build helper invoked did not exist at all.
export default defineConfig(({ mode }) => {
  const demo = mode === "demo";
  const viewer = mode === "viewer";

  return {
    // Substituted at build time so the flags survive into the bundle
    // regardless of how the shell passes (or fails to pass) an env var.
    define: {
      "import.meta.env.VITE_DEMO": JSON.stringify(demo ? "1" : "0"),
      "import.meta.env.VITE_VIEWER": JSON.stringify(viewer ? "1" : "0"),
    },
    plugins: [react(), tailwindcss()],
    // Relative for the static targets so they work from GitHub Pages, from a
    // subdirectory, and from file:// alike.
    base: demo || viewer ? "./" : "/app/",
    // public/ holds only the baked demo payloads — never ship them inside the
    // pip package, which has a live backend and no use for recorded
    // responses. The viewer has no payloads: its data arrives by drag & drop.
    publicDir: demo ? "public" : false,
    build: {
      outDir: viewer
        ? "../viewer-dist"
        : demo
          ? "../demo-dist"
          : "../modelmri/static/app",
      emptyOutDir: true,
    },
    server: {
      proxy: {
        "/api": "http://127.0.0.1:5900",
        "/ws": { target: "ws://127.0.0.1:5900", ws: true },
      },
    },
  };
});
