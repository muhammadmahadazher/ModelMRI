// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { DEMO, demoFetch } from "./demo";
import { VIEWER, viewerFetch } from "./viewer";
import "./styles.css";

// Viewer build: the API is a `.mri` the user dropped, parsed in this page.
// Same patch-fetch trick as the demo below, for the same reason — every call
// site stays identical to the real app, so the viewer cannot drift from the
// product. Nothing is uploaded; there is nowhere to upload it to.
if (VIEWER) {
  const real = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    if (!url.startsWith("/api/")) return real(input as RequestInfo, init);
    const isJson = typeof init?.body === "string";
    const { status, payload } = await viewerFetch(
      url,
      isJson ? JSON.parse(init!.body as string) : undefined,
      isJson ? null : ((init?.body as BodyInit | null) ?? null),
    );
    return new Response(JSON.stringify(payload), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  };
}

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
      const answer = await demoFetch(url, body);
      if (answer === undefined) {
        // Nothing reachable should land here — tests/demo_check.py fails the
        // build on any endpoint api.ts can call and demo.ts does not answer.
        return new Response(JSON.stringify({ error: "not available in the demo" }), {
          status: 409,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify(answer.payload), {
        status: answer.status,
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
