import type { ReactNode } from "react";
import type { TraceStep } from "./api";

/**
 * The kinds a recorded step may be, and how each one looks.
 *
 * A leaf module, for the same reason `modelmri/step_kinds.py` is one on the
 * other side: two panels need this list — the timeline colours and labels by
 * it, the rubric editor offers it in a picker — and whichever of them owned it
 * the other would have to import a whole panel to get at four strings.
 * `RubricPanel` had the alternative for a while: its own hand-written array of
 * the six kinds, a plain literal TypeScript cannot check, sitting one screen
 * away from the map it was supposed to agree with.
 *
 * ## Colour ran out before the kinds did
 *
 * This palette has eight hues, deliberately — `styles.css` records twice, in
 * its own words, that every remaining arc of the wheel sits within a difference
 * nobody can tell apart on a 7px dot, so a ninth hue would be a distinction
 * that only exists in the stylesheet. Six of the eight were already spoken for
 * by the original six kinds. Four new kinds and three free hues means one pair
 * shares, and the pair is `retrieval` and `rerank` — the two that are the same
 * family, both working over a document set, moss being the hue this system
 * already uses for provenance.
 *
 * A shared hue is only honest if the difference is carried somewhere else, and
 * that somewhere is the glyph: one shape per kind, in the legend, on the
 * inspector chip, and on the timeline bar itself wherever the bar came out wide
 * enough to hold a whole one. The shapes are drawn in the same schematic
 * language as `RestingSketch` — rectangles, lines and circles that diagram what
 * the step DID — rather than as illustrated icons, which stop being readable at
 * 12px anyway.
 */

/**
 * Every kind the store accepts.
 *
 * An alias for `TraceStep["kind"]` now that the wire type names all ten, and
 * deliberately still written as an alias rather than as ten fresh literals:
 * the wire type is the one the server's own payloads are typed by, so a second
 * list here would be the copy that goes stale. It was briefly the union of
 * that type and the four new names, for the window in which the wire type was
 * still the original six.
 */
export type StepKind = TraceStep["kind"];

/**
 * Kind -> hue token. `Record` and not an object literal: a kind added to
 * `StepKind` without a colour here is a compile error, which is the only
 * mechanism in this file that cannot be forgotten.
 */
const KIND_COLOR: Record<StepKind, string> = {
  llm_call: "var(--color-agent)",
  tool_call: "var(--color-attn)",
  subagent: "var(--color-feat)",
  mcp_call: "var(--color-vla)",
  user_turn: "var(--color-mute)",
  error: "var(--color-pop)",
  // Moss is "where an answer came from" everywhere else in this app — it is
  // the grounding panel's hue, and grounding asks whether an answer is in the
  // document at all. Retrieval is that question one step earlier, and rerank
  // is retrieval's second pass, so they read as one family on the timeline and
  // are told apart by their glyphs.
  retrieval: "var(--color-ground)",
  rerank: "var(--color-ground)",
  // Teal already means "the reference you measure against". An embedding is a
  // coordinate in exactly such a space.
  embedding: "var(--color-custom)",
  // Magenta is reserved for a readout whose output is a sentence rather than a
  // number. A guardrail's output is a policy name and a verdict, not a measure.
  guardrail: "var(--color-scope)",
};

/**
 * Every kind, in the order the legend and the pickers show them: the six that
 * were always here, then the retrieval-shaped four with the two that share a
 * hue adjacent, so the sharing reads as deliberate rather than as a mistake.
 *
 * Derived from the colour map rather than written out again — that map is
 * completeness-checked by its own type, and this is how the legend has always
 * been built.
 */
export const STEP_KINDS = Object.keys(KIND_COLOR) as StepKind[];

/**
 * A kind this build has no colour for renders MUTED, not invisible.
 *
 * `.tl-block` is `all: unset`, so an undefined background paints nothing: a
 * step that is on the timeline, occupies its lane, and cannot be seen or
 * clicked. That is reachable — a `.mri` bundle does not validate kinds
 * (`session.py` refuses only an empty one), so an older viewer opening a newer
 * bundle meets exactly this. Grey with the kind's own name beside it is the
 * honest rendering of "recorded, and this build does not know what it is";
 * refusing the bundle outright would be strictly worse for the reader, who is
 * holding a file somebody sent them.
 */
export function kindColor(kind: string): string {
  return KIND_COLOR[kind as StepKind] ?? "var(--color-mute)";
}

/**
 * Is this a kind this build has a colour and a shape for?
 *
 * Because grey is not "unknown" here — grey is `user_turn`, with its own row
 * in the legend. The fallback above fixed an invisible bar and stopped one
 * step short of a distinguishable one: on the timeline the glyph that carries
 * the difference is hidden on any bar under 22px, so a narrow bar of an
 * unrecognised kind is pixel-identical to a narrow user turn, and the legend
 * beside it confirms the wrong reading. Only the tooltip disagrees.
 *
 * So the caller asks this and marks the bar (`.tl-block.unknown-kind` hatches
 * it, the same idiom the stylesheet already uses for "this is not what it
 * looks like"), which survives at any width and costs no hue.
 */
export function isKnownKind(kind: string): boolean {
  return STEP_KINDS.includes(kind as StepKind);
}

/**
 * One shape per kind, on a 12-unit grid.
 *
 * Stroked rather than filled, in `currentColor`, so one drawing serves the
 * legend (in the kind's hue), the chip (in the chip's) and the bar (knocked out
 * in the panel's ground) — and so both themes are handled by the token that
 * feeds `color`, with nothing here to keep in step per palette.
 */
const KIND_GLYPH: Record<StepKind, ReactNode> = {
  // A speech bubble: something was said back.
  llm_call: (
    <>
      <rect x="1.4" y="2" width="9.2" height="6" rx="1.4" />
      <path d="M4 8v2.4L6.6 8" />
    </>
  ),
  // A mechanism: a bolt head with its four flats.
  tool_call: (
    <>
      <rect x="4.2" y="4.2" width="3.6" height="3.6" rx="0.8" />
      <path d="M6 1.4v2.8M6 7.8v2.8M1.4 6h2.8M7.8 6h2.8" />
    </>
  ),
  // A trunk with two children hanging off it.
  subagent: (
    <>
      <path d="M2.6 1.6v6.2a1.6 1.6 0 0 0 1.6 1.6h1.2" />
      <path d="M2.6 4.6h3.2" />
      <circle cx="7.2" cy="4.6" r="1.4" />
      <circle cx="7.2" cy="9.4" r="1.4" />
    </>
  ),
  // A diamond: a hop through a protocol rather than a direct call.
  mcp_call: <path d="M6 1.4 10.6 6 6 10.6 1.4 6z" />,
  // A person.
  user_turn: (
    <>
      <circle cx="6" cy="3.8" r="1.9" />
      <path d="M2 10.4a4 4 0 0 1 8 0" />
    </>
  ),
  // The one glyph that is a convention rather than a diagram, because this one
  // has to be legible before it is read.
  error: (
    <>
      <path d="M6 1.6 10.9 10.2H1.1z" />
      <path d="M6 5.2v2.1M6 8.9v.1" />
    </>
  ),
  // A stack of documents with one drawn out of it.
  retrieval: (
    <>
      <path d="M1.4 2.8h4.4M1.4 6h4.4M1.4 9.2h4.4" />
      <path d="M7.2 6h3.4M9 4.3 10.7 6 9 7.7" />
    </>
  ),
  // The same stack, reordered: two arrows pointing opposite ways.
  rerank: (
    <>
      <path d="M3.4 10.2V2.2M1.8 3.8 3.4 2.2 5 3.8" />
      <path d="M8.6 1.8v8M7 8.4l1.6 1.6L10.2 8.4" />
    </>
  ),
  // Points in a space: two axes and a scatter.
  embedding: (
    <>
      <path d="M1.8 1.6v8.6h8.6" />
      <circle cx="4.6" cy="7.4" r="0.9" />
      <circle cx="6.8" cy="4.8" r="0.9" />
      <circle cx="9" cy="6.6" r="0.9" />
    </>
  ),
  // A shield.
  guardrail: (
    <path d="M6 1.4 10.4 3v3.4c0 2.4-1.9 3.8-4.4 4.4-2.5-.6-4.4-2-4.4-4.4V3z" />
  ),
};

/** Drawn dashed on purpose: an outline with nothing in it is what "this build
 *  has no shape for that kind" looks like, and it cannot be mistaken for one of
 *  the ten above. */
const UNKNOWN_GLYPH = (
  <rect x="1.8" y="1.8" width="8.4" height="8.4" rx="1.2" strokeDasharray="2 1.6" />
);

/**
 * The kind's shape, at 12px.
 *
 * `aria-hidden` throughout: every place this is drawn already prints the kind's
 * name in text beside it, and a second announcement of the same word is noise
 * to a screen reader rather than help.
 */
export function KindGlyph({ kind, color }: { kind: string; color?: string }) {
  return (
    <svg
      className="kind-glyph"
      viewBox="0 0 12 12"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.3}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={color ? { color } : undefined}
    >
      {KIND_GLYPH[kind as StepKind] ?? UNKNOWN_GLYPH}
    </svg>
  );
}
