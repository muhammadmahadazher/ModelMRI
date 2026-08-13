import { useEffect, useMemo, useState } from "react";
import {
  compareQuantisation,
  GgufPlan,
  GgufReport,
  GgufTensor,
  loadGguf,
  planGguf,
  QuantBehaviour,
} from "./api";

const bytes = (n: number) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
};

type SortKey = "name" | "type_name" | "elements" | "bytes" | "offset";

/** What is inside a GGUF, without loading it.
 *
 *  Every other local runner shows a quantisation label and a file size. The
 *  label is a preset name — it says Q4_K and then a third of the bytes are
 *  somewhere else entirely. This shows where they actually went, computed per
 *  tensor from the file's own table.
 *
 *  When any tensor uses a ggml type the reader does not know, the byte totals
 *  are withheld rather than averaged over the parts that were recognised. The
 *  parameter count survives, because element counts come from the tensor
 *  shapes and do not depend on the quantisation type at all.
 */
export default function GgufReader({
  report,
  onClose,
  onLoaded,
}: {
  report: GgufReport;
  onClose: () => void;
  /** Called after the file has become the live model, so the rest of the app
   *  can refresh its status. Optional — the panel is useful read-only. */
  onLoaded?: () => void;
}) {
  const [sort, setSort] = useState<SortKey>("bytes");
  const [desc, setDesc] = useState(true);
  const [filter, setFilter] = useState("");
  // Every hook above every early return. React error #310 has shipped twice
  // in this codebase from a `return` sneaking above a useState.
  const [plan, setPlan] = useState<GgufPlan | null>(null);
  const [planErr, setPlanErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadErr, setLoadErr] = useState("");
  // The behaviour half of the damage report. Every hook stays above the single
  // early-return-free render below; React error #310 has shipped twice here.
  const [original, setOriginal] = useState("");
  const [dPrompt, setDPrompt] = useState("The capital of France is");
  const [diff, setDiff] = useState<QuantBehaviour | null>(null);
  const [diffBusy, setDiffBusy] = useState(false);
  const [diffErr, setDiffErr] = useState("");
  const s = report.summary;

  // Asked as soon as the file is open, because the answer is the reason to
  // open it: the header is a few hundred kilobytes and the verdict decides
  // whether the next click is worth thirty seconds or is doomed.
  useEffect(() => {
    let live = true;
    setPlan(null);
    setPlanErr("");
    planGguf(report.path)
      .then((p) => live && setPlan(p))
      .catch((e) => live && setPlanErr(String(e?.message || e)));
    return () => {
      live = false;
    };
  }, [report.path]);

  const tensors = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const rows = q
      ? report.tensors.filter((t) => t.name.toLowerCase().includes(q))
      : report.tensors.slice();
    rows.sort((a, b) => {
      const av = a[sort];
      const bv = b[sort];
      // Nulls last whichever way the column is sorted — an unknown size is not
      // a small size, and letting it sort as 0 would put the tensors the
      // reader could not measure at one end as though they were tiny.
      if (av === null) return 1;
      if (bv === null) return -1;
      const cmp = typeof av === "string" ? av.localeCompare(bv as string) : av - (bv as number);
      return desc ? -cmp : cmp;
    });
    return rows;
  }, [report.tensors, sort, desc, filter]);

  const head = (key: SortKey, label: string) => (
    <th
      className={sort === key ? "sorted" : ""}
      onClick={() => {
        if (sort === key) setDesc(!desc);
        else {
          setSort(key);
          setDesc(true);
        }
      }}
      title={`Sort by ${label}`}
    >
      {label}
      {sort === key && <span className="caret">{desc ? " ▾" : " ▴"}</span>}
    </th>
  );

  return (
    <div className="gguf">
      <div className="row">
        <strong>{s.name || report.path.split(/[\\/]/).pop()}</strong>
        <span className="meta">
          {s.architecture || "architecture not stated"} · GGUF v{report.version} ·{" "}
          {report.tensor_count} tensors
        </span>
        <span className="spacer" />
        <button className="ghost sm" onClick={onClose}>
          ← back to the list
        </button>
      </div>

      {/* The headline. Parameters are always exact; bytes and bpw are withheld
          when anything could not be sized, rather than averaged over what was
          recognised and printed as the file's own figure. */}
      <div className="gguf-head">
        <span className="gguf-stat">
          <b>{s.parameters.toLocaleString()}</b>
          <span className="meta">parameters</span>
        </span>
        <span className="gguf-stat">
          <b>{s.tensor_bytes != null ? bytes(s.tensor_bytes) : "—"}</b>
          <span className="meta">tensor bytes</span>
        </span>
        <span className="gguf-stat gguf-bpw">
          <b>{s.effective_bpw != null ? `${s.effective_bpw}` : "—"}</b>
          <span className="meta">effective bits/weight</span>
        </span>
        {s.context_length != null && (
          <span className="gguf-stat">
            <b>{s.context_length.toLocaleString()}</b>
            <span className="meta">context</span>
          </span>
        )}
        {s.head_count != null && (
          <span className="gguf-stat">
            <b>
              {s.head_count}
              {s.head_count_kv != null && `/${s.head_count_kv}`}
            </b>
            <span className="meta">heads (q/kv)</span>
          </span>
        )}
        {s.tokenizer && (
          <span className="gguf-stat">
            <b>{s.tokenizer}</b>
            <span className="meta">tokeniser</span>
          </span>
        )}
      </div>

      {s.why_unmeasured && (
        <div className="hint err refusal">{s.why_unmeasured}</div>
      )}

      {/* What loading it would cost. The one number nobody expects: a 4-bit
          GGUF does not load as a 4-bit model. Transformers has no kernels for
          these types, so it dequantises everything on the way in and the file
          size stops predicting anything. Measured on this repo's machine, a
          0.397 GB Q4_K_M file became 1.192 GB of bfloat16 tensors after a
          2.384 GB predicted transit (sampled RSS 2.30 GB, -3.5%). Saying that
          before the click is the whole feature. */}
      {planErr && <div className="hint err refusal">{planErr}</div>}
      {plan && (
        <div className={`gguf-plan v-${plan.verdict.replace(/ /g, "-")}`}>
          <div className="gguf-plan-bars">
            <span className="gguf-stat">
              <b>{bytes(plan.file_bytes)}</b>
              <span className="meta">on disk</span>
            </span>
            <span className="gguf-arrow" aria-hidden="true">
              →
            </span>
            <span className="gguf-stat">
              <b>{bytes(plan.resident_bytes)}</b>
              <span className="meta">loaded at {plan.dtype}</span>
            </span>
            <span className="gguf-stat">
              <b>{bytes(plan.peak_host_bytes)}</b>
              <span className="meta">peak host RAM</span>
            </span>
            {plan.expansion != null && (
              <span className="gguf-stat gguf-bpw">
                <b>{plan.expansion.toFixed(2)}×</b>
                <span className="meta">bigger than the file</span>
              </span>
            )}
          </div>
          <div className="row">
            {/* The verdict is about loading this file. When it is already the
                live model that question is about a SECOND copy beside the
                first — true, doomed, and not what anyone is asking. */}
            <span
              className={`pill ${
                plan.already_loaded || plan.verdict === "fits" ? "ok" : "warn"
              }`}
            >
              {plan.already_loaded ? "loaded" : plan.verdict}
            </span>
            <span className="meta">
              {plan.already_loaded
                ? "this file is the model currently loaded — every panel below is reading it"
                : plan.why}
            </span>
            <span className="spacer" />
            <button
              // `cta`, not `primary` -- there is no `.primary` rule in
              // styles.css and there never was, so this rendered as a raw
              // browser-default button. Same class of miss as the `.d-attention`
              // selector that shipped once before: a class name invented at
              // the call site and never checked against the stylesheet.
              className="cta sm"
              hidden={plan.already_loaded}
              disabled={busy || plan.verdict === "will not fit"}
              title={
                plan.verdict === "will not fit"
                  ? plan.why
                  : "Dequantise this file into a full torch module — every panel, not just chat"
              }
              onClick={() => {
                setBusy(true);
                setLoadErr("");
                // confirm is passed only for a tight fit. "will not fit" is
                // arithmetic, and the button is disabled for it rather than
                // offering an override that cannot work.
                loadGguf(report.path, undefined, plan.verdict === "tight")
                  .then(() => onLoaded?.())
                  .catch((e) => setLoadErr(String(e?.message || e)))
                  .finally(() => setBusy(false));
              }}
            >
              {busy
                ? "dequantising…"
                : plan.verdict === "tight"
                  ? "load anyway"
                  : "load for introspection"}
            </button>
          </div>
          {loadErr && <div className="hint err refusal">{loadErr}</div>}
          {plan.notes.map((n) => (
            <div className="hint" key={n}>
              {n}
            </div>
          ))}
        </div>
      )}

      {/* Where the bits went. This is the part the quantisation label hides. */}
      {Object.keys(s.by_type).length > 0 && (
        <>
          <div className="meta cand-head">
            by quantisation type
            {!s.by_type_covers_whole_file && " — covers only part of the file"}
          </div>
          <table className="gguf-table">
            <thead>
              <tr>
                <th>type</th>
                <th>bits/weight</th>
                <th>tensors</th>
                <th>elements</th>
                <th>bytes</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s.by_type)
                .sort((a, b) => b[1].bytes - a[1].bytes)
                .map(([name, row]) => (
                  <tr key={name} className={name === s.dominant_type ? "sel" : ""}>
                    <td>
                      <code>{name}</code>
                      {name === s.dominant_type && (
                        <span className="meta"> · dominant</span>
                      )}
                    </td>
                    <td>{row.bpw ?? "—"}</td>
                    <td>{row.tensors}</td>
                    <td>{row.elements.toLocaleString()}</td>
                    <td>{bytes(row.bytes)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </>
      )}

      {s.higher_precision_tensors.length > 0 && (
        <div className="hint">
          <strong>Above the headline:</strong>{" "}
          {s.higher_precision_tensors
            .slice(0, 6)
            .map((o) => `${o.name} (${o.type}, ${o.bpw})`)
            .join(", ")}
          . These sit above <code>{s.dominant_type}</code> and are excluded from
          it — which is why a file labelled for its dominant type can read much
          higher overall.
        </div>
      )}

      {/* The behaviour half of #36. The weight half says how far the numbers
          moved; this says whether the model still answers the same, which is
          the question people actually have. A tensor can move a long way in
          RMS and change no answer. */}
      <div className="meta cand-head" style={{ marginTop: 16 }}>
        what the quantisation cost — behaviour
      </div>
      <div className="gguf-diff">
        <div className="row">
          <input
            className="sm"
            value={original}
            placeholder="the full-precision original — a hub id or a folder"
            onChange={(e) => setOriginal(e.target.value)}
            spellCheck={false}
            style={{ flex: "2 1 260px" }}
          />
          <input
            className="sm"
            value={dPrompt}
            placeholder="prompt"
            onChange={(e) => setDPrompt(e.target.value)}
            spellCheck={false}
            style={{ flex: "1 1 180px" }}
          />
          <button
            className="cta sm"
            disabled={diffBusy || !original.trim() || !dPrompt.trim()}
            title="Loads both models one after the other and unloads whatever is currently held to make room"
            onClick={() => {
              setDiffBusy(true);
              setDiffErr("");
              setDiff(null);
              compareQuantisation(report.path, original.trim(), dPrompt)
                .then((r) => {
                  setDiff(r);
                  // Both models are gone by the time this returns and the
                  // previously-held one was unloaded to make room, so the rest
                  // of the app is now looking at nothing.
                  onLoaded?.();
                })
                .catch((e) => setDiffErr(String(e?.message || e)))
                .finally(() => setDiffBusy(false));
            }}
          >
            {diffBusy ? "measuring…" : "measure the damage"}
          </button>
        </div>
        <div className="hint">
          Loads both models <strong>one after the other</strong> — they never
          sit in memory together — and unloads whatever is currently held to
          make room. One prompt is one sample: this describes the prompt, not
          the model.
        </div>
        {diffErr && <div className="hint err refusal">{diffErr}</div>}
        {diff && <Damage d={diff} />}
      </div>

      <div className="row" style={{ marginTop: 14 }}>
        <span className="meta cand-head">tensors</span>
        <input
          className="sm"
          value={filter}
          placeholder="filter by name — try  blk.0  or  attn"
          onChange={(e) => setFilter(e.target.value)}
          spellCheck={false}
        />
        <span className="meta">
          {tensors.length} of {report.tensors.length}
        </span>
      </div>

      <div className="gguf-scroll">
        <table className="gguf-table">
          <thead>
            <tr>
              {head("name", "name")}
              {head("type_name", "type")}
              <th>dims</th>
              {head("elements", "elements")}
              {head("bytes", "bytes")}
              <th>bits/weight</th>
              {head("offset", "offset")}
            </tr>
          </thead>
          <tbody>
            {tensors.map((t: GgufTensor) => (
              <tr key={t.name} className={t.bytes === null ? "err" : ""}>
                <td>
                  <code>{t.name}</code>
                </td>
                <td>{t.type_name}</td>
                <td className="meta">{t.dims.join(" × ")}</td>
                <td>{t.elements.toLocaleString()}</td>
                {/* Not 0. An unknown ggml type has an unknown size, and a
                    dash is the only honest cell. */}
                <td>{t.bytes != null ? bytes(t.bytes) : "unknown type"}</td>
                <td>{t.bpw ?? "—"}</td>
                <td className="meta">{t.offset.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <details className="gguf-meta">
        <summary>
          every metadata key ({Object.keys(report.metadata).length})
        </summary>
        <table className="gguf-table">
          <tbody>
            {Object.entries(report.metadata).map(([k, v]) => (
              <tr key={k}>
                <td>
                  <code>{k}</code>
                </td>
                <td className="gguf-val">
                  {typeof v === "object" && v !== null
                    ? // Long arrays arrive truncated with a stated count — a
                      // tokeniser vocabulary is often 128,000 strings.
                      JSON.stringify(v).slice(0, 300)
                    : String(v)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <div className="hint">{s.means}</div>
    </div>
  );
}

/** The behaviour damage report.
 *
 *  The flip count is split into contested and decisive rather than shown as
 *  one number. Measured on SmolLM2-135M Q4_K_M against its original, "The
 *  capital of France is": one flip, and it sat at a 0.038 margin — the
 *  original ranked ',' at 0.322 against ' is' at 0.319. Reporting that beside
 *  a flip at a 0.9 margin, as a single "1 token changed", would overstate the
 *  damage by describing a broken tie as a changed mind.
 */
function Damage({ d }: { d: QuantBehaviour }) {
  const s = d.summary;
  const worst = s.worst_layer;
  const peak = Math.max(
    ...(d.attention?.map((r) => r.mean_abs_diff) ?? [1]),
    Number.MIN_VALUE,
  );
  return (
    <div className="gguf-damage">
      <div className="gguf-head">
        <span className="gguf-stat">
          <b>{s.median_kl.toFixed(4)}</b>
          <span className="meta">median KL (nats)</span>
        </span>
        <span className="gguf-stat">
          <b>{s.max_kl.toFixed(4)}</b>
          <span className="meta">worst, at {s.max_kl_at.token.trim() || "▁"}</span>
        </span>
        <span className="gguf-stat gguf-bpw">
          <b>
            {s.decisive_flips}
            <span className="meta"> / {s.positions}</span>
          </b>
          <span className="meta">answers actually changed</span>
        </span>
        <span className="gguf-stat">
          <b>{s.contested_flips}</b>
          <span className="meta">ties broken (margin &lt; 0.05)</span>
        </span>
        {worst && (
          <span className="gguf-stat">
            <b>L{worst.layer}</b>
            <span className="meta">most divergent attention</span>
          </span>
        )}
      </div>

      {d.flips.length > 0 && (
        <table className="gguf-table">
          <thead>
            <tr>
              <th>after</th>
              <th>original said</th>
              <th>quantised said</th>
              <th>its margin</th>
              <th>KL</th>
            </tr>
          </thead>
          <tbody>
            {d.flips.map((p) => (
              <tr key={p.index} className={p.contested ? "" : "err"}>
                <td>
                  <code>{p.token}</code>
                </td>
                <td>
                  <code>{p.top_b}</code> {p.p_b.toFixed(3)}
                </td>
                <td>
                  <code>{p.top_a}</code> {p.p_a.toFixed(3)}
                </td>
                <td>
                  {p.margin_b.toFixed(4)}
                  <span className="meta">
                    {p.contested ? " · a tie" : " · decisive"}
                  </span>
                </td>
                <td>{p.kl.toFixed(5)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Per layer, so damage can be located in depth rather than only
          totalled. Bars are relative to the worst layer in this run. */}
      {d.attention && (
        <div className="gguf-layers">
          {d.attention.map((r) => (
            <span
              key={r.layer}
              className="gguf-layer"
              title={`layer ${r.layer}: ${r.mean_abs_diff.toExponential(2)} mean |Δ|`}
              style={{ ["--h" as string]: `${(r.mean_abs_diff / peak) * 100}%` }}
            />
          ))}
        </div>
      )}

      {d.notes.map((n) => (
        <div className="hint" key={n}>
          {n}
        </div>
      ))}
      <div className="hint">{s.means}</div>
    </div>
  );
}
