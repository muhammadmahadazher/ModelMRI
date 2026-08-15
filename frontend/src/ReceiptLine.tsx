import { useState } from "react";
import { Receipt } from "./api";

/**
 * What produced the number above this line.
 *
 * Collapsed to one line by default and expandable to the full record. The
 * ratio matters: provenance that costs a panel a third of its height gets
 * removed by the next person who needs the space, and provenance nobody can
 * see is not provenance. One line, then everything.
 *
 * TYPE-DEFENSIVE ON PURPOSE. This renders receipts from a `.mri` a stranger
 * sent as well as ones this process just produced. Python's `receipts.parse`
 * has already refused the malformed shapes, but the viewer build must not
 * depend on Python having run — the zero-install viewer reads the file in the
 * browser, which is the one reader that always has somebody else's bytes.
 */
export default function ReceiptLine({ receipt }: { receipt?: Receipt | null }) {
  const [open, setOpen] = useState(false);
  if (!receipt || typeof receipt !== "object") return null;

  const text = (value: unknown): string | null =>
    typeof value === "string" && value.trim() ? value : null;

  const model = text(receipt.model);
  const revision = text(receipt.revision);
  const dtype = text(receipt.dtype);
  const device = text(receipt.device);
  const attn = text(receipt.attn_implementation);
  // A seed of 0 is a real seed and must not be hidden by a truthiness test;
  // null is "this measurement was not seeded", which is a different fact.
  const seed = typeof receipt.seed === "number" ? receipt.seed : null;

  // Rows for the expanded view. A fact that could not be established shows
  // the reason the writer recorded rather than being dropped — "no revision"
  // and "several revisions were cached so naming one would be a guess" send
  // the reader to different places.
  const rows: [string, string][] = [];
  const push = (label: string, value: string | null, why?: unknown) => {
    if (value) rows.push([label, value]);
    else if (text(why)) rows.push([label, `not established — ${text(why)}`]);
  };

  push("model", model);
  push("revision", revision, receipt.revision_note);
  push("dtype", dtype);
  push("device", device);
  push("attention", attn);
  // NOT "not established". An absent seed is a fact that WAS established —
  // the measurement is deterministic and there was no draw to seed. Reusing
  // the failed-lookup wording here would report a known thing as unknown,
  // which is the same error in the opposite direction from the one the rest
  // of this file is careful about.
  if (seed !== null) rows.push(["seed", String(seed)]);
  else rows.push(["seed", "none — this measurement was not seeded"]);
  push("tokenizer", text(receipt.tokenizer_sha256), receipt.tokenizer_note);
  push("prompt", text(receipt.prompt_sha256));
  push(
    "prompt tokens",
    typeof receipt.n_prompt_tokens === "number"
      ? String(receipt.n_prompt_tokens)
      : null,
  );
  push("modelmri", text(receipt.tool_version));
  push("measured", text(receipt.measured_at));

  const request = receipt.request;
  const args =
    request && typeof request === "object" && !Array.isArray(request)
      ? Object.entries(request).filter(([, v]) => v !== null && v !== undefined)
      : [];

  // The one-line form. Deliberately the facts that change an answer: two
  // receipts differing in any of these describe measurements that cannot be
  // compared, which is the question this line exists to answer at a glance.
  const glance = [
    model ? `${model}${revision ? `@${revision.slice(0, 8)}` : ""}` : null,
    dtype,
    device,
    attn ? `attn=${attn}` : null,
    seed === null ? null : `seed=${seed}`,
  ].filter(Boolean);

  if (!glance.length && !rows.length) return null;

  return (
    <div className={`receipt${open ? " open" : ""}`}>
      <button
        type="button"
        className="receipt-line"
        onClick={() => setOpen((was) => !was)}
        aria-expanded={open}
      >
        <span className="receipt-tag">measured by</span>
        <span className="receipt-glance">
          {glance.length ? glance.join(" · ") : "setup not recorded"}
        </span>
        {!revision && (
          /* Called out rather than left as an absence. A finding whose
             revision is unknown cannot be re-run against the same weights,
             and that is the single fact most worth noticing here. */
          <span className="receipt-warn" title={text(receipt.revision_note) ?? ""}>
            no revision
          </span>
        )}
        <span className="receipt-caret">{open ? "⌃" : "⌄"}</span>
      </button>

      {open && (
        <dl className="receipt-full">
          {rows.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
          {args.length > 0 && (
            <div className="receipt-args">
              <dt>request</dt>
              <dd>
                {args.map(([k, v]) => (
                  <code key={k}>
                    {k}={typeof v === "object" ? JSON.stringify(v) : String(v)}
                  </code>
                ))}
              </dd>
            </div>
          )}
        </dl>
      )}
    </div>
  );
}
