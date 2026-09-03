// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
export type SketchKind =
    | "custom"
    | "vla"
    | "agent"
    | "patch"
    | "image"
    | "vision";

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
                {kind === "vision" && <Regions />}
            </svg>
            <span className="sketch-cap">
                {kind === "custom" && "one forward pass, layer by layer"}
                {kind === "vla" && "what the policy looked at, per frame"}
                {kind === "agent" && "every step, nested, on one timeline"}
                {kind === "patch" && "every layer against every token"}
                {kind === "image" && "every word, at every denoising step"}
                {/* No causal verb on purpose. Attention says where a model
                    LOOKED and an occlusion sweep says what an answer rested
                    on, and those are different claims — the panel below is
                    careful to separate them, so its picture must not merge
                    them in five words. */}
                {kind === "vision" && "what it says, and where in the picture"}
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

/** WHERE IN THE PICTURE — a frame, its patches, its candidate regions, and
 *  one of them closing on the place the field concentrates.
 *
 *  This exists because the vision section and the text-to-image section were
 *  drawing the SAME sketch: `StepsByWord`, a run over TIME with a step axis
 *  down the side. That is the right picture for a diffusion model and the
 *  wrong one here, and being wrong is the smaller half of the problem — a
 *  reader who has just been shown a step axis expects a step axis, and a
 *  vision panel does not have one. Pixels go in and a label, a box or a mask
 *  comes out; the question the panel is FOR is which part of the frame the
 *  answer is about. So: no time in this drawing at all.
 *
 *  Four things, each carrying one word of that sentence:
 *
 *    the frame      — a picture is the subject, not a prompt
 *    the lattice    — the model sees it as patches, not as a photograph
 *    two dashed     — several regions are proposed
 *    one bracketed  — one of them is what gets read
 *
 *  NOT DATA, per the rule at the top of this file. The concentration is a
 *  smooth falloff around a fixed point plus a little seeded grain. The grain
 *  is deliberate and not decoration: a clean radial gradient looks like a
 *  RENDERED HEAT MAP, which is precisely the thing a resting panel must not
 *  be mistaken for, and a lattice of slightly uneven tiles cannot be read as
 *  one. Nothing is labelled, nothing is a number, and the corner marks sit
 *  where they sit because the composition wanted them there.
 */

/** The lattice, in the 220x120 box every sketch here is drawn in. */
const V_COLS = 12;
const V_ROWS = 6;
const V_CELL = 13;
const V_PITCH = 16;
const V_X0 = 17;
const V_Y0 = 14;
/** The picture the lattice is OF. Inset from the lattice by a few units on
 *  every side: drawn flush, it stopped reading as the edge of a photograph
 *  and started reading as the outer rule of a table. */
const V_FRAME = { x: 10, y: 8, w: 200, h: 104 };
/** Where the field concentrates, and a quieter second place it does not. */
const V_PEAK: readonly [number, number] = [152, 58];
const V_NEAR: readonly [number, number] = [58, 46];
/** The region that gets read. Deliberately NOT aligned to the lattice: a box
 *  is continuous and a patch grid is not, and drawing them flush would claim
 *  an agreement between the two that no detector has. */
const V_LOCK = { x: 119, y: 31, w: 66, h: 54 };
/** Each corner mark is two arms of this length, so its whole path measures
 *  twice it — which is the number `.sk-bracket`'s dash draws on. Changing
 *  this without changing that leaves the marks half drawn. */
const V_ARM = 12;
/** Regions the picture PROPOSES, and the delay each fades in on. They arrive
 *  before the field does, because that is the order of the sentence: here are
 *  the candidates, and here is where the answer turned out to live. */
const V_PROPOSED: readonly (readonly [number, number, number, number, number])[] = [
    [30, 24, 56, 44, 80],
    [52, 72, 58, 30, 160],
];

function falloff(dx: number, dy: number, sigma: number): number {
    return Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
}

function Regions() {
    const grain = bars(97, V_COLS * V_ROWS);
    const { x, y, w, h } = V_LOCK;
    /** Four L-shaped paths, clockwise from the top left. */
    const corners = [
        `M${x} ${y + V_ARM}L${x} ${y}L${x + V_ARM} ${y}`,
        `M${x + w - V_ARM} ${y}L${x + w} ${y}L${x + w} ${y + V_ARM}`,
        `M${x + w} ${y + h - V_ARM}L${x + w} ${y + h}L${x + w - V_ARM} ${y + h}`,
        `M${x + V_ARM} ${y + h}L${x} ${y + h}L${x} ${y + h - V_ARM}`,
    ];
    return (
        <g>
            <rect
                className="sk-frame"
                x={V_FRAME.x}
                y={V_FRAME.y}
                width={V_FRAME.w}
                height={V_FRAME.h}
                rx={4}
            />

            {grain.map((g, i) => {
                const col = i % V_COLS;
                const row = Math.floor(i / V_COLS);
                const cx = V_X0 + col * V_PITCH + V_CELL / 2;
                const cy = V_Y0 + row * V_PITCH + V_CELL / 2;
                const hot = falloff(cx - V_PEAK[0], cy - V_PEAK[1], 28);
                const warm = falloff(cx - V_NEAR[0], cy - V_NEAR[1], 22);
                // Floored well above nothing: the lattice is structure and
                // has to stay visible where the field is cold, or the drawing
                // becomes two blobs on an empty card.
                const v = Math.min(
                    0.86,
                    Math.max(0.08, 0.1 + 0.72 * hot + 0.24 * warm + 0.11 * (g - 0.6)),
                );
                // The pass runs INWARD — the edges arrive first and the peak
                // last — so the motion is the field concentrating rather than
                // a wipe. It settles; it does not loop.
                const d = Math.hypot(cx - V_PEAK[0], cy - V_PEAK[1]);
                return (
                    <rect
                        key={i}
                        className="sk-cell"
                        x={V_X0 + col * V_PITCH}
                        y={V_Y0 + row * V_PITCH}
                        width={V_CELL}
                        height={V_CELL}
                        rx={1}
                        style={{
                            animationDelay: `${Math.round(Math.max(0, 360 - d * 2.6))}ms`,
                            opacity: v,
                        }}
                    />
                );
            })}

            {/* Proposed, not chosen. Dashed and quiet so the bracketed one is
                unmistakably the one being read. */}
            {V_PROPOSED.map(([bx, by, bw, bh, delay], i) => (
                <rect
                    key={i}
                    className="sk-region"
                    x={bx}
                    y={by}
                    width={bw}
                    height={bh}
                    rx={2}
                    style={{ animationDelay: `${delay}ms`, opacity: 0.55 }}
                />
            ))}

            {/* Corner marks rather than a closed box: a closed rectangle over
                a lattice of squares reads as one more, larger square. */}
            {corners.map((d, i) => (
                <path
                    key={i}
                    className="sk-bracket"
                    d={d}
                    style={{ animationDelay: `${520 + i * 50}ms` }}
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
