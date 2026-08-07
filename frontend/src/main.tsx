import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { DEMO, demoFetch } from "./demo";
import "./styles.css";

// Demo build (GitHub Pages): serve pre-baked real responses instead of the
// API. Patching fetch once keeps every call site identical to the real app,
// so the demo can never drift from the product.
if (DEMO) {
  const real = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.startsWith("/api/")) {
      const body = init?.body ? JSON.parse(init.body as string) : undefined;
      // a beat of latency so streaming/loading states are still visible
      await new Promise((r) => setTimeout(r, 120));
      const payload = await demoFetch(url, body);
      if (payload === undefined) {
        return new Response(JSON.stringify({ error: "not available in the demo" }), {
          status: 409,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return real(input as RequestInfo, init);
  };
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
