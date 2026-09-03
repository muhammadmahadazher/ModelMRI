// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { defineConfig } from "vite";
import type { Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// ------------------------------------------------------------ head per target
//
// One index.html feeds three builds, so whatever its head says, it says three
// times. That shipped two wrong pages:
//
//   /viewer/  declared rel="canonical" pointing at the demo — "this page is a
//             duplicate of that one, drop it" — while sitemap.xml listed it as
//             a URL in its own right. Two contradictory instructions to the
//             same crawler.
//   /app/     carried a canonical URL, Open Graph cards and schema.org data
//             into the pip package, where the page is served from localhost
//             and every one of those claims is unverifiable by anyone.
//
// Both blocks are delimited in index.html and replaced here. The authored
// version is the demo's, so opening the file directly still shows something
// true.
const SITE = "https://muhammadmahadazher.github.io/ModelMRI";

const NAME =
  "An open-source, local-first tool for looking inside a language model: " +
  "attention maps, causal attention-head ranking by ablation, " +
  "sparse-autoencoder features, steering, and a flight recorder for agent " +
  "runs. Everything runs on your own machine.";

/** The links every no-JavaScript fallback should offer. */
const LINKS = `<p>
          <a href="https://github.com/muhammadmahadazher/ModelMRI">Source on GitHub</a> ·
          <a href="${SITE}/docs/">Documentation</a> ·
          <a href="https://pypi.org/project/modelmri/">PyPI</a>
        </p>`;

function viewerHead(): string {
  // No Open Graph and no structured data: this page is a file reader, and a
  // second SoftwareApplication declaration for the same software would just
  // compete with the one on the demo. It keeps a self-canonical so the URL
  // stays indexable on its own terms, which is what sitemap.xml already says.
  return `<title>Open a .mri analysis — ModelMRI viewer</title>
    <meta
      name="description"
      content="Open a .mri file someone sent you and read the attention, tokens and generation it records. The file is read in your browser — nothing is uploaded, and nothing needs to be installed."
    />
    <link rel="canonical" href="${SITE}/viewer/" />`;
}

function appHead(): string {
  // Served from localhost by the pip package. A canonical URL, a social card
  // and a price of zero are claims about a public web page; this is not one.
  return `<title>ModelMRI</title>
    <meta name="description" content="ModelMRI — attention, features and agent traces for a local model." />`;
}

function noscriptFor(target: Target): string {
  const closing =
    target === "viewer"
      ? "This page needs JavaScript to read a .mri file."
      : target === "app"
        ? "This page needs JavaScript. If you are seeing this, the bundled app failed to load — reinstall with <code>pip install --force-reinstall modelmri</code>."
        : "This page needs JavaScript to run the interactive demo.";
  const intro =
    target === "viewer"
      ? "Opens a <code>.mri</code> analysis file in your browser. The file is read locally and never uploaded."
      : NAME;
  return `<noscript>
      <main>
        <h1>ModelMRI</h1>
        <p>${intro}</p>
        <p><code>pip install modelmri</code> then <code>modelmri serve</code></p>
        ${LINKS}
        <p>${closing}</p>
      </main>
    </noscript>`;
}

type Target = "app" | "demo" | "viewer";

/**
 * Replace exactly one delimited region, or fail the build.
 *
 * Silence is the failure mode that matters here: if someone edits index.html
 * and the markers stop matching, a quiet no-op ships the demo's canonical URL
 * on the viewer again and nothing on the page says so. An exception during
 * `vite build` is the only version of this anyone will notice.
 */
function replaceRegion(
  html: string,
  region: string,
  replacement: string,
  target: Target,
): string {
  const pattern = new RegExp(`<!-- ${region}:start[\\s\\S]*?${region}:end -->`, "g");
  const found = html.match(pattern);
  if (found?.length !== 1) {
    throw new Error(
      `modelmri-head (${target}): expected exactly one "${region}" region in ` +
        `index.html, found ${found?.length ?? 0}. The markers were edited or ` +
        `removed — see the comment at the top of frontend/index.html.`,
    );
  }
  return html.replace(
    pattern,
    `<!-- ${region}:start (generated for ${target}) -->\n    ${replacement}\n    <!-- ${region}:end -->`,
  );
}

function headPlugin(target: Target): Plugin {
  return {
    name: "modelmri-head",
    transformIndexHtml(html) {
      if (target !== "demo") {
        html = replaceRegion(
          html,
          "head",
          target === "viewer" ? viewerHead() : appHead(),
          target,
        );
      }
      // The demo's noscript is regenerated too, so all three come from one
      // function rather than one being authored and two generated.
      return replaceRegion(html, "noscript", noscriptFor(target), target);
    },
  };
}

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
  const target: Target = demo ? "demo" : viewer ? "viewer" : "app";

  return {
    // Substituted at build time so the flags survive into the bundle
    // regardless of how the shell passes (or fails to pass) an env var.
    define: {
      "import.meta.env.VITE_DEMO": JSON.stringify(demo ? "1" : "0"),
      "import.meta.env.VITE_VIEWER": JSON.stringify(viewer ? "1" : "0"),
    },
    plugins: [react(), tailwindcss(), headPlugin(target)],
    // Relative for the static targets so they work from GitHub Pages, from a
    // subdirectory, and from file:// alike.
    base: demo || viewer ? "./" : "/app/",
    // public/ holds only the baked demo payloads — never ship them inside the
    // pip package, which has a live backend and no use for recorded
    // responses. The viewer has no payloads: its data arrives by drag & drop.
    publicDir: demo ? "public" : false,
    build: {
      // The viewer emits INTO the package, not to a sibling dist directory.
      // It has two consumers — `modelmri open`, which serves it from the
      // installed package, and the Pages deploy, which copies it to /viewer/
      // — and two build outputs is two things that can disagree about what
      // the viewer is. The deploy copies from here.
      outDir: viewer
        ? "../modelmri/static/viewer"
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
