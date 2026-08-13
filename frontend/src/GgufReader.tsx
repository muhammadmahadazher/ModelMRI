import { useMemo, useState } from "react";
import { GgufReport, GgufTensor } from "./api";

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
}: {
  report: GgufReport;
  onClose: () => void;
}) {
  const [sort, setSort] = useState<SortKey>("bytes");
  const [desc, setDesc] = useState(true);
  const [filter, setFilter] = useState("");
  const s = report.summary;

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
