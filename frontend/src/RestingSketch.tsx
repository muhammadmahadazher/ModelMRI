/** What a panel will show you, sketched, while it is still empty.
 *
 *  A resting panel was copy and a button inside a 46ch measure, which is the
 *  right measure for prose and left three fifths of every panel blank. The
 *  space is now used for a small diagram of the SHAPE of that panel's output —
 *  layer bars, a frame with regions, a run of steps.
 *
 *  It is deliberately not data. Nothing here is measured, nothing is derived
 *  from a model, and the strokes are muted and unlabelled precisely so it
 *  cannot be mistaken for a reading — a decorative chart carrying invented
 *  numbers would be the worst thing this project could put on screen. What it
 *  carries instead is the answer to "what will I get if I press this", which
 *  is a real question a resting panel otherwise answers only in words.
 *
 *  The motion is one slow pass keyed off nothing. It stops after settling
 *  rather than looping, because an element that moves forever is one the eye
 *  learns to discard, and `prefers-reduced-motion` removes it entirely
 *  through the global rule in styles.css.
 */
export type SketchKind = "custom" | "vla" | "agent" | "patch" | "image";

/** Deterministic pseudo-random, so the sketch is the same on every render and
 *  a re-render never looks like new information. */
function bars(seed: number, n: number): number[] {
    let x = seed;
    return Array.from({ length: n }, () => {
        x = (x * 1103515245 + 12345) & 0x7fffffff;
        return 0.25 + (x / 0x7fffffff) * 0.72;
    });
}

export default function RestingSketch({ kind }: { kind: SketchKind }) {
    return (
        <div className={`sketch sk-${kind}`} aria-hidden="true">
            <svg viewBox="0 0 220 120" preserveAspectRatio="xMidYMid meet">
                {kind === "custom" && <Layers />}
                {kind === "vla" && <Frame />}
                {kind === "agent" && <Timeline />}
                {kind === "patch" && <Grid />}
                {kind === "image" && <StepsByWord />}
            </svg>
            <span className="sketch-cap">
                {kind === "custom" && "one forward pass, layer by layer"}
                {kind === "vla" && "what the policy looked at, per frame"}
                {kind === "agent" && "every step, nested, on one timeline"}
                {kind === "patch" && "every layer against every token"}
                {kind === "image" && "every word, at every denoising step"}
            </span>
        </div>
    );
}

/** A stack of per-layer bars: shapes and statistics down a network. */
function Layers() {
    const vals = bars(7, 11);
    return (
        <g>
            {vals.map((v, i) => (
                <rect
                    key={i}
                    className="sk-bar"
                    x={12 + i * 18}
                    y={104 - v * 84}
                    width={11}
                    height={v * 84}
                    rx={2}
                    style={{ animationDelay: `${i * 55}ms` }}
                />
            ))}
            <line className="sk-axis" x1="8" y1="105" x2="212" y2="105" />
        </g>
    );
}

/** A video frame with the regions a policy attends to. */
function Frame() {
    return (
        <g>
            <rect className="sk-frame" x="14" y="12" width="192" height="92" rx="4" />
            {[
                [62, 44, 15],
                [108, 66, 22],
                [154, 38, 11],
            ].map(([cx, cy, r], i) => (
                <circle
                    key={i}
                    className="sk-blob"
                    cx={cx}
                    cy={cy}
                    r={r}
                    style={{ animationDelay: `${i * 220}ms` }}
                />
            ))}
            <line className="sk-axis" x1="14" y1="112" x2="206" y2="112" />
            {[0, 1, 2, 3, 4, 5].map((i) => (
                <line
                    key={i}
                    className="sk-tick"
                    x1={14 + i * 38}
                    y1="108"
                    x2={14 + i * 38}
                    y2="116"
                />
            ))}
        </g>
    );
}

/** Nested steps on one clock. */
function Timeline() {
    const rows: [number, number, number][] = [
        [0, 14, 196],
        [1, 30, 120],
        [2, 46, 62],
        [2, 118, 38],
        [1, 160, 46],
    ];
    return (
        <g>
            {rows.map(([depth, x, w], i) => (
                <rect
                    key={i}
                    className="sk-step"
                    x={12 + x}
                    y={16 + i * 18}
                    width={w}
                    height={10}
                    rx={3}
                    style={{ animationDelay: `${i * 90}ms`, opacity: 1 - depth * 0.18 }}
                />
            ))}
            <line className="sk-axis" x1="8" y1="112" x2="212" y2="112" />
        </g>
    );
}

/** Words across, denoising steps down — the shape of a cross-attention run.
 *
 *  Deliberately the same cell vocabulary as `Grid` and a different geometry,
 *  because the two panels answer different questions in the same picture: that
 *  one is layers against tokens, this one is steps against words. A wide short
 *  grid also carries the one fact the caption cannot: there are far more steps
 *  than words, which is why the step axis is kept rather than averaged away.
 */
function StepsByWord() {
    const cells = bars(41, 56);
    return (
        <g>
            {cells.map((v, i) => (
                <rect
                    key={i}
                    className="sk-cell"
                    x={30 + (i % 8) * 22}
                    y={12 + Math.floor(i / 8) * 14}
                    width={19}
                    height={11}
                    rx={1}
                    style={{
                        animationDelay: `${Math.floor(i / 8) * 80}ms`,
                        opacity: 0.16 + v * 0.58,
                    }}
                />
            ))}
            {/* The step axis, unlabelled like everything else here. */}
            <line className="sk-axis" x1="22" y1="10" x2="22" y2="110" />
            {[0, 1, 2, 3, 4, 5, 6].map((i) => (
                <line
                    key={i}
                    className="sk-tick"
                    x1="18"
                    y1={17 + i * 14}
                    x2="26"
                    y2={17 + i * 14}
                />
            ))}
        </g>
    );
}

/** A layer-by-position grid, some cells hot. */
function Grid() {
    const cells = bars(23, 48);
    return (
        <g>
            {cells.map((v, i) => (
                <rect
                    key={i}
                    className={`sk-cell ${v > 0.82 ? "hot" : ""}`}
                    x={14 + (i % 12) * 16}
                    y={16 + Math.floor(i / 12) * 22}
                    width={14}
                    height={20}
                    rx={1}
                    style={{
                        animationDelay: `${Math.floor(i / 12) * 70}ms`,
                        opacity: 0.18 + v * 0.55,
                    }}
                />
            ))}
        </g>
    );
}
