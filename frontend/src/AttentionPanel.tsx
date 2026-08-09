import { useEffect, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  Ablation,
  AttentionData,
  AttentionDiff,
  errorText,
  exportSession,
  getAttention,
  getAttentionDiff,
  getAttentionMeta,
  rankHeads,
} from "./api";
import ArcCanvas from "./ArcCanvas";

export default function AttentionPanel({
  epoch,
  replay,
}: {
  epoch: number;
  replay?: boolean;
}) {
  const scanRef = useScanOnData(epoch);
  const [layers, setLayers] = useState(0);
  const [heads, setHeads] = useState(0);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [data, setData] = useState<AttentionData | null>(null);
  const [info, setInfo] = useState("");

  /** An instrument says where it is *within a range*, and pads so the row
   *  does not reflow as the number crosses ten. "L 09/18" beats "layer 9". */
  const dial = (label: string, i: number, n: number) => {
    const w = String(Math.max(n - 1, 0)).length;
    return `${label} ${String(i).padStart(w, "0")}/${String(Math.max(n - 1, 0))}`;
  };

  const [err, setErr] = useState("");
  const [ranked, setRanked] = useState<Ablation | null>(null);
  const [ranking, setRanking] = useState(false);
  // A comparison against the same generation with one head removed. Null
  // means we are showing the run itself rather than a difference.
  const [diff, setDiff] = useState<AttentionDiff | null>(null);

  /** Show what removing one head changes — at the first layer where it can.
   *
   *  Removing a head zeroes its OUTPUT, so the layer it lives in is computed
   *  from an unchanged input and its attention is bit-identical. Wired
   *  naively, this button compared the viewed layer against an ablation in
   *  that same layer and was therefore guaranteed to show nothing at all.
   *  The first layer that can differ is the next one, so go there.
   */
  async function compare(cutLayer: number, cutHead: number) {
    const at = Math.min(cutLayer + 1, layers - 1);
    setErr("");
    setLayer(at);
    try {
      const result = await getAttentionDiff(
        at,
        head,
        "live",
        `ablate:${cutLayer}.${cutHead}`,
      );
      setDiff(result);
    } catch (e) {
      setErr(errorText(e));
      setDiff(null);
    }
  }

  async function rank() {
    setRanking(true);
    setErr("");
    try {
      const result = await rankHeads(layer, "zero");
      setRanked(result);
      // Open the head that moved the answer most — the whole point is to
      // stop the user picking blind.
      if (result.ranked.length) setHead(result.ranked[0].head);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setRanking(false);
    }
  }

  // A ranking is about one layer's heads under one generation. Changing
  // either makes it an answer to a question nobody asked.
  useEffect(() => {
    setRanked(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layer, epoch]);

  // A difference is about one head of one generation. Anything that changes
  // either makes it a picture of something else.
  useEffect(() => {
    setDiff(null);
  }, [epoch]);

  useEffect(() => {
    let live = true;
    void (async () => {
      const meta = await getAttentionMeta().catch(() => null);
      if (!live || !meta?.available) return;
      setErr("");
      setLayers(meta.n_layers!);
      setHeads(meta.n_heads!);
      setLayer(Math.floor(meta.n_layers! / 2));
      setHead(0);
      setInfo(
        `${meta.n_layers}L × ${meta.n_heads}H · ${meta.n_tokens} tok`,
      );
    })();
    return () => {
      live = false;
    };
  }, [epoch]);

  useEffect(() => {
    if (layers === 0) return;
    let live = true;
    const t = performance.now();
    // The first fetch for a generation runs a full output_attentions forward
    // pass server-side and can take seconds, so say so rather than showing
    // controls above an empty space.
    setInfo(`${dial("L", layer, layers)} · ${dial("H", head, heads)} · computing…`);
    void getAttention(layer, head)
      .then((d) => {
        if (!live) return;
        setErr("");
        setData(d);
        setInfo(
          `${dial("L", d.layer, layers)} · ${dial("H", d.head, heads)} · ` +
            `${d.tokens.length} tok · ${((performance.now() - t) / 1000).toFixed(2)}s`,
        );
      })
      .catch((e) => {
        if (!live) return;
        setErr(errorText(e));
        setInfo("");
      });
    return () => {
      live = false;
    };
  }, [layers, layer, head, epoch]);

  if (layers === 0) return null;

  const options = (n: number) =>
    Array.from({ length: n }, (_, i) => (
      <option key={i} value={i}>
        {i}
      </option>
    ));

  /** Head options, ordered by a ranking when there is one.
   *
   *  The dropdown is 144 numbers and no reason to pick any of them. Once
   *  heads are ranked, the label carries the score and the order carries the
   *  answer — so the first entry is the head worth opening.
   */
  const headOptions = () => {
    if (!ranked) return options(heads);
    const byHead = new Map(
      ranked.ranked.filter((r) => r.layer === layer).map((r) => [r.head, r]),
    );
    return [...byHead.entries()]
      .sort((a, b) => b[1].kl - a[1].kl)
      .map(([h, score]) => (
        <option key={h} value={h}>
          {h} · KL {score.kl < ranked.noise_floor_kl ? "—" : score.kl.toFixed(3)}
          {score.flips_top ? " · flips" : ""}
        </option>
      ));
  };

  return (
    <div ref={scanRef} className="panel attn">
      <div className="sect">
        <span className="dot d-attn" />
        <h2 className="h-attn">ATTENTION — WHERE EACH TOKEN LOOKED</h2>
        <span className="rule" />
      </div>
      <div className="row" style={{ margin: "10px 0" }}>
        <label className="meta">layer</label>
        <select value={layer} onChange={(e) => setLayer(Number(e.target.value))}>
          {options(layers)}
        </select>
        <label className="meta">head</label>
        <select value={head} onChange={(e) => setHead(Number(e.target.value))}>
          {headOptions()}
        </select>
        <span className="meta">{info}</span>
        {/* Needs a forward pass per head, so it needs a model — a recording
            does not carry one. */}
        {!replay && (
          <button
            className="ghost sm"
            onClick={() => void rank()}
            disabled={ranking}
            title="Zero each head in this layer and measure how far the answer moves"
          >
            {ranking ? "ranking…" : "Rank heads"}
          </button>
        )}
        {!replay && <ShareButton layer={layer} head={head} />}
        <span className="spacer" />
        {/* Arc thickness encodes weight, which cannot be read without a
            key. Three stops is enough to calibrate the eye. */}
        <span className="weight-key" aria-hidden="true">
          <span><i style={{ height: 1 }} />0.05</span>
          <span><i style={{ height: 3 }} />0.20</span>
          <span><i style={{ height: 6 }} />0.50</span>
        </span>
      </div>
      {ranked && (
        <div className="ranking">
          <div className="ranking-head">
            <strong>
              Removing one head from layer {layer}, and measuring how far the
              answer to {JSON.stringify(ranked.target_token)} moves
            </strong>
            <span className="meta">
              {ranked.passes} forward passes · {ranked.elapsed_s}s ·{" "}
              {ranked.baseline}-ablation
            </span>
          </div>
          <ol className="ranking-list">
            {ranked.ranked
              .filter((r) => r.layer === layer)
              .slice(0, 5)
              .map((r) => {
                const noise = r.kl <= ranked.noise_floor_kl;
                return (
                  <li key={r.head} className={noise ? "faint" : ""}>
                    <button className="ghost sm" onClick={() => setHead(r.head)}>
                      H{r.head}
                    </button>
                    <span className="mid">
                      {noise ? "below the noise floor" : `KL ${r.kl.toFixed(3)}`}
                    </span>
                    <span className="meta">
                      p({JSON.stringify(ranked.target_token)}){" "}
                      {r.p_top_before.toFixed(3)} → {r.p_top_after.toFixed(3)}
                      {r.flips_top && " · changes the top token"}
                    </span>
                    <span className="spacer" />
                    <button
                      className="ghost sm"
                      onClick={() => void compare(r.layer, r.head)}
                      title={`Show what changes at layer ${r.layer + 1} when L${r.layer} H${r.head} is removed`}
                    >
                      what changes?
                    </button>
                  </li>
                );
              })}
          </ol>
          {/* The caveat travels with the numbers. An ordered list reads as
              truth, and two of these scores are not what a reader assumes. */}
          <div className="hint">
            These are <em>not</em> each head's share of the prediction — they do
            not add up, and are not meant to. Each says only: removing this one
            head, on its own, moves the answer this much. The ranking also
            depends on what a removed head is replaced with; this used{" "}
            <code>{ranked.baseline}</code>, and on some layers the mean baseline
            gives a different order.
          </div>
        </div>
      )}
      {err ? (
        <div className="hint err">
          Could not compute this attention map — {err}. Generate again, or pick
          a different layer.
        </div>
      ) : (
        <>
          {diff ? (
            <>
              <div className="diffbar">
                <strong>
                  Layer {diff.layer}, head {diff.head} — what changed when{" "}
                  {diff.b.replace("ablate:", "L").replace(".", " H")} was removed
                </strong>
                <span className="meta">
                  {diff.moved} of {diff.cells} cells moved · largest{" "}
                  {diff.max_abs.toFixed(3)}
                </span>
                <span className="spacer" />
                <span className="diff-key" aria-hidden="true">
                  <i className="up" /> attends more <i className="down" /> less
                </span>
                <button className="ghost sm" onClick={() => setDiff(null)}>
                  Back to the map
                </button>
              </div>
              {diff.note ? (
                <div className="hint">{diff.note}</div>
              ) : (
                <ArcCanvas tokens={diff.tokens} matrix={diff.matrix} signed />
              )}
              <div className="hint">
                A difference between two forward passes over the{" "}
                <em>same</em> tokens — so position {"i"} is the same token on
                both sides. Comparing two separate generations would not be:
                different sampling, or a different chat template, shifts every
                index and the subtraction becomes fiction.
              </div>
            </>
          ) : (
            <>
              {data && <ArcCanvas tokens={data.tokens} matrix={data.matrix} />}
              <div className="hint">
                hover or focus a token → arcs show what it attended to · click
                or Enter to pin · arc thickness = attention weight
                {replay && " · recorded, not live"}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

/** Save this analysis as a `.mri`, optionally with a note.
 *
 *  The note is the reason the format exists. "L14 H3 moves the subject token"
 *  is what you want to say when you send it; without somewhere to put that,
 *  the file arrives as a heat map with no claim attached and the recipient
 *  has to guess what they are meant to be seeing.
 */
function ShareButton({ layer, head }: { layer: number; head: number }) {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const save = async () => {
    setBusy(true);
    setErr("");
    try {
      const { blob, filename } = await exportSession(layer, head, note);
      // Object URL rather than a data: URI — a session is megabytes, and a
      // data: URI that large is refused by the browser without saying why.
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      setOpen(false);
      setNote("");
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button
        className="ghost sm"
        onClick={() => setOpen(true)}
        title="Save this analysis as a .mri anyone can open without the model"
      >
        Share this view
      </button>
    );
  }

  return (
    <span className="share-row">
      <input
        autoFocus
        className="share-note"
        placeholder="what did you find? (optional)"
        value={note}
        maxLength={200}
        onChange={(e) => setNote(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") void save();
          if (e.key === "Escape") setOpen(false);
        }}
      />
      <button className="ghost sm" onClick={() => void save()} disabled={busy}>
        {busy ? "packing…" : "Save .mri"}
      </button>
      <button className="ghost sm" onClick={() => setOpen(false)}>
        Cancel
      </button>
      {err && <span className="hint err">{err}</span>}
    </span>
  );
}
