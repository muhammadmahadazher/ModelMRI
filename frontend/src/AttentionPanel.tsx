import { useEffect, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  AttentionData,
  errorText,
  getAttention,
  getAttentionMeta,
} from "./api";
import ArcCanvas from "./ArcCanvas";

export default function AttentionPanel({ epoch }: { epoch: number }) {
  const scanRef = useScanOnData(epoch);
  const [layers, setLayers] = useState(0);
  const [heads, setHeads] = useState(0);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [data, setData] = useState<AttentionData | null>(null);
  const [info, setInfo] = useState("");

  const [err, setErr] = useState("");

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
        `${meta.n_layers} layers · ${meta.n_heads} heads · ${meta.n_tokens} tokens`,
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
    setInfo(`layer ${layer} · head ${head} · computing…`);
    void getAttention(layer, head)
      .then((d) => {
        if (!live) return;
        setErr("");
        setData(d);
        setInfo(
          `layer ${d.layer} · head ${d.head} · ${d.tokens.length} tokens · ` +
            `${((performance.now() - t) / 1000).toFixed(2)}s`,
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
          {options(heads)}
        </select>
        <span className="meta">{info}</span>
      </div>
      {err ? (
        <div className="hint err">
          Could not compute this attention map — {err}. Generate again, or pick
          a different layer.
        </div>
      ) : (
        <>
          {data && <ArcCanvas tokens={data.tokens} matrix={data.matrix} />}
          <div className="hint">
            hover or focus a token → arcs show what it attended to · click or
            Enter to pin · arc thickness = attention weight
          </div>
        </>
      )}
    </div>
  );
}
