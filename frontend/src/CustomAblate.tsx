import { CSSProperties, useEffect, useState } from "react";
import { measured } from "./measured";
import { Ablation, ablateCustom, AblationSite, errorText } from "./api";

/**
 * What actually matters in the network you trained.
 *
 * The panel this sits inside is DESCRIPTIVE: it says what each layer emitted,
 * which units are dead, what went non-finite. All of that can be true of a
 * layer the answer does not depend on. This asks the causal question instead
 * — and it is the one surface in this project nothing in the category will
 * ever cover, because every platform it competes with is a fixed catalogue of
 * transformers and none of them will look at the CNN somebody trained last
 * week.
 *
 * Three things it draws that a bar chart of effect sizes would not:
 *
 *   - The CONTROL, per row, as a marker on the same bar. A site that did not
 *     clear it is drawn as not clearing it rather than as a slightly shorter
 *     bar of the same colour.
 *   - UNTESTED as its own state. Sites past the controlled cap carry a score
 *     and no verdict, and that is drawn as no verdict — not as a failure.
 *   - The MULTIPLE COMPARISON, in the same breath as the count of what
 *     cleared. Each site is compared against the strongest of its draws, so
 *     one site in nine clears having done nothing.
 */

/** The row's state, in the order the states have to be decided. */
function verdict(site: AblationSite): { text: string; cls: string } {
  if (site.beats_control === null)
    return { text: "not tested", cls: "ab-untested" };
  if (site.beats_control) return { text: "beats its control", cls: "ab-clears" };
  return { text: "a random edit did as much", cls: "ab-null" };
}

export default function CustomAblate({ epoch }: { epoch: number }) {
  const [kind, setKind] = useState<"layers" | "inputs">("layers");
  const [data, setData] = useState<Ablation | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // A sweep belongs to the model that was loaded when it ran. The custom
  // panel remounts on a new load, but the sweep is also invalidated by a new
  // forward pass changing nothing about it — so this clears on the panel's
  // own epoch rather than trying to be clever.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch]);

  async function run(which: "layers" | "inputs") {
    setKind(which);
    setBusy(true);
    setErr("");
    setData(null);
    try {
      setData(await ablateCustom(which));
    } catch (e) {
      // The refusals ARE the feature here: an adapter that does not declare
      // TASK and one with no sample_inputs() both get a message saying what
      // to add and why, rather than a metric picked for them.
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  // Scaled against the strongest bar. Effects are in the task's own unit and
  // have no fixed maximum, so a percentage-of-anything width would be
  // arithmetic about nothing.
  const widest = data
    ? data.sites.reduce((m, s) => Math.max(m, s.effect, s.control_max ?? 0), 0) || 1
    : 1;
  const cleared = data ? data.sites.filter((s) => s.beats_control) : [];

  return (
    <div className="custom-ablate">
      <div className="row">
        <span className="meta">
          what <b>matters</b> — the map above says what each layer emitted,
          which can be true of a layer the answer does not depend on
        </span>
      </div>

      <div className="row" style={{ margin: "8px 0" }}>
        <button
          className={`pill sm ${kind === "layers" ? "on" : ""}`}
          onClick={() => void run("layers")}
          disabled={busy}
        >
          {busy && kind === "layers" ? "sweeping…" : "ablate each layer"}
        </button>
        <button
          className={`pill sm ${kind === "inputs" ? "on" : ""}`}
          onClick={() => void run("inputs")}
          disabled={busy}
        >
          {busy && kind === "inputs" ? "sweeping…" : "occlude each input region"}
        </button>
        <span className="meta">
          needs an adapter with <code>TASK</code> and{" "}
          <code>sample_inputs()</code>
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          <p className="meta ab-head">
            {data.n_sites} sites over {data.n_samples} of your samples ·{" "}
            {data.passes} forward passes, {data.seconds}s · scores are{" "}
            <b>{data.unit}</b>
          </p>

          {/* The count and what it is worth, together. Separating them is how
              a reader ends up treating a site that cleared by a hair as the
              same kind of thing as one that cleared by 590x. */}
          <div
            className={`ab-verdict ${
              cleared.length && cleared.length > data.expected_false_positives
                ? "ok"
                : "none"
            }`}
          >
            {cleared.length === 0 ? (
              <>
                <b>Nothing beat its control.</b> A random edit of the same size
                in the same place did as much or more everywhere tested. On an
                untrained or fragile model that is the honest answer rather
                than a failure of the sweep.
              </>
            ) : (
              <>
                <b>
                  {cleared.length} of {data.n_controlled}
                </b>{" "}
                tested sites beat every control draw — against{" "}
                <b>{data.expected_false_positives.toFixed(1)}</b> that would
                clear by chance, since each site is compared with the strongest
                of its draws. Read the margin, not the flag.
              </>
            )}
          </div>

          <ol className="ab-rows stagger">
            {data.sites.map((site, i) => {
              const v = verdict(site);
              return (
                <li
                  key={site.name}
                  className={v.cls}
                  style={{ "--i": i } as CSSProperties}
                >
                  <span className="mid ab-name" title={site.name}>
                    {site.name}
                  </span>
                  <span className="ab-track">
                    <span
                      className="ab-bar"
                      style={{ width: `${(site.effect / widest) * 100}%` }}
                    />
                    {/* The control as a MARKER on the same bar, not a second
                        bar: it is the line this one had to cross, and two
                        bars invite reading it as a second measurement. */}
                    {site.control_max !== null && (
                      <span
                        className="ab-ctl"
                        style={{ left: `${(site.control_max / widest) * 100}%` }}
                        title={`the strongest of ${site.control_draws} control draws reached ${site.control_max}`}
                      />
                    )}
                  </span>
                  <span className="mid ab-val">{measured(site.effect, 4)}</span>
                  <span className="meta ab-verd">{v.text}</span>
                </li>
              );
            })}
          </ol>

          <p className="meta ab-means">{data.means}</p>
        </>
      )}
    </div>
  );
}
