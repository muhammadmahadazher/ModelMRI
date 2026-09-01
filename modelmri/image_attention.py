"""Which words the image is looking at, and when.

A diffusion model attends to the prompt at every denoising step, through
cross-attention blocks at several resolutions. Early steps decide layout; late
steps decide texture. A single averaged map hides that completely, which is
why the step axis is kept rather than collapsed.

DAAM published this for Stable Diffusion specifically and is research code.
Nothing local, general and installable does it across families — so the rule
here is the one `imaging.py` sets: read the model, never assume it.

## Why attention is only half of it

This project's own standard says attention is the weak, correlational version.
`patch.py` exists on the text side because a heatmap is not a cause, and the
same is true here — a word can be attended to and change nothing.

So this module ships the interventional counterpart beside the map:
`knockout` removes one prompt token, regenerates at the SAME seed, and
measures what actually moved. The seed is doing the work; without it the
difference is sampling and the number is noise wearing a label.

## The capture

`diffusers` exposes `set_attn_processor`, so the probabilities can be captured
where they are computed rather than recomputed from hidden states. That
matters for the same reason `runtime.py` forces eager attention on the text
side: SDPA never materialises the matrix, and a map reconstructed afterwards
is a different quantity from the one the model used.

Only CROSS-attention is captured. Self-attention among latents is a different
question (which pixels look at which pixels) and mixing them into one map
would average two things that are not the same thing.

## Memory

An attention map is (heads x pixels x tokens) per block per step. At 50 steps
across a dozen blocks that is gigabytes if it is all kept. So maps are reduced
to (pixels x tokens) on the spot, mean over heads, and accumulated per step —
and the reduction is stated because a mean over heads is a choice that hides
head-level disagreement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from . import fmt
from .errors import BadRequest, Refusal
from .image_steps import public_model_name

log = logging.getLogger(__name__)

# The most denoising steps a capture will run. Each is a full UNet pass and
# every one is captured, so this is a bound on work AND on memory. What was
# dropped is always reported.
MAX_STEPS = 50

# The most prompt tokens a map will carry. CLIP pads to 77 and the padding
# carries real attention mass, which is a genuine finding rather than noise --
# but a map with 77 columns of which 60 are `<pad>` is unreadable, so the
# padded tail is reported separately rather than plotted.
MAX_TOKENS = 77


class NotSupported(Refusal):
    """This architecture does not have what the measurement needs."""


@dataclass
class StepMap:
    """One denoising step's cross-attention, already reduced."""

    step: int
    # Sigma / timestep the scheduler was on. Carried because "step 12" means
    # nothing across schedulers with different step counts.
    timestep: float = 0.0
    # (tokens,) — attention mass per prompt token, summed over pixels and
    # averaged over heads and blocks. The per-pixel map is kept separately
    # only for the tokens a caller asks for, because all of them is gigabytes.
    per_token: list[float] = field(default_factory=list)
    # How many attention blocks contributed. Reported because a model with
    # more cross-attention blocks is not more attentive, it just has more.
    blocks: int = 0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestep": self.timestep,
            "per_token": self.per_token,
            "blocks": self.blocks,
        }


@dataclass
class AttentionRun:
    """Every step's map, plus what the run cannot claim."""

    tokens: list[str] = field(default_factory=list)
    steps: list[StepMap] = field(default_factory=list)
    seed: int | None = None
    model: str = ""
    revision: str = ""
    # Tokens past the real prompt. CLIP pads to 77 and the padding attracts
    # real mass; reported so a reader is not told the model is fascinated by
    # `<pad>`.
    padding_from: int = 0
    steps_requested: int = 0
    resolutions: list[int] = field(default_factory=list)
    #: How wide the denoiser's conditioning actually was, read off the maps.
    #: Not the tokenizer's length and not `MAX_TOKENS`: PixArt-Alpha is 120 and
    #: Sigma is 300, while `tokenizer.model_max_length` for their T5 is 512.
    conditioning_width: int = 0
    #: Columns that were measured and have no label to put on them, because
    #: `_tokenize` caps labels at `MAX_TOKENS`. Reported rather than dropped
    #: quietly — this is a cap on what can be SHOWN, not on what was measured.
    columns_unlabelled: int = 0

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "steps": [s.to_dict() for s in self.steps],
            "seed": self.seed,
            "model": self.model,
            "revision": self.revision,
            "padding_from": self.padding_from,
            "steps_requested": self.steps_requested,
            "steps_measured": len(self.steps),
            "resolutions": self.resolutions,
            "conditioning_width": self.conditioning_width,
            "columns_unlabelled": self.columns_unlabelled,
            "means": self.means(),
        }

    def means(self) -> str:
        dropped = (
            f" {self.steps_requested - len(self.steps)} of the requested steps "
            f"were not captured."
            if self.steps_requested > len(self.steps)
            else ""
        )
        pad = (
            f" Tokens from index {self.padding_from} are padding, not your "
            f"prompt — they carry real attention mass and are reported "
            f"separately rather than plotted as words."
            if self.padding_from and self.padding_from < len(self.tokens)
            else ""
        )
        seeded = (
            f" Seed {self.seed}."
            if self.seed is not None
            else " No seed was fixed, so another run gives another trajectory."
        )
        unlabelled = (
            f" This denoiser's conditioning is {self.conditioning_width} "
            f"columns wide and only {len(self.tokens)} of them could be "
            f"labelled, so {self.columns_unlabelled} measured column(s) are "
            f"not plotted — a limit on what can be SHOWN, not on what was "
            f"measured."
            if self.columns_unlabelled
            else ""
        )
        return (
            f"Cross-attention from {len(self.steps)} denoising steps of "
            f"{self.model or 'this model'}, averaged over heads and over the "
            f"{len(self.resolutions)} attention resolutions "
            f"({', '.join(str(r) for r in self.resolutions)}).{seeded}{dropped}"
            f"{pad}{unlabelled}\n\n"
            f"ATTENTION IS NOT A CAUSE. A token can be attended to and change "
            f"nothing in the image. `knockout` removes a token and regenerates "
            f"at the same seed, which is the measurement that can say what a "
            f"word did."
        )


def capture(
    pipe,
    prompt: str,
    *,
    steps: int = 20,
    seed: int | None = None,
    height: int | None = None,
    width: int | None = None,
    on_step=None,
    # The repo id the caller loaded. `pipe.name_or_path` is the snapshot
    # DIRECTORY when diffusers resolved from cache, which is the normal case
    # here — and a response is no place for somebody's drive letter, least of
    # all one that ships inside a `.mri` they send to a colleague.
    model_name: str = "",
):
    """Run the pipeline once, keeping every step's cross-attention.

    `pipe` is a loaded diffusers pipeline. This module never loads one: the
    caller decides what to hold in memory, and `imaging.detect` has already
    said whether the architecture has cross-attention at all.
    """
    import torch

    denoiser = _denoiser_of(pipe)
    if denoiser is None:
        raise NotSupported(
            "this pipeline has no `unet` or `transformer`, so there is no "
            "denoiser to capture attention from."
        )
    if not hasattr(denoiser, "set_attn_processor"):
        raise NotSupported(
            f"`{type(denoiser).__name__}` does not expose `set_attn_processor`, "
            f"which is how the probabilities are captured where they are "
            f"computed. Reconstructing them from hidden states afterwards "
            f"would be a different quantity from the one the model used."
        )
    # BEFORE anything is installed and long before anything is generated.
    #
    # `imaging` calls SD3, Flux, AuraFlow and CogVideoX DiT-shaped, which is
    # true, and DiT-shaped carried `cross_attention` on the strength of the
    # family name. None of them has an `attn2`, so `_wrap` came back with
    # nothing wrapped, the run finished with an empty store, and the refusal
    # at the bottom of this function — written for a genuine gap in coverage —
    # arrived after the reader had paid for up to fifty denoising passes and
    # read as a fault in their model.
    #
    # The fact that settles it is a walk of `named_children`. It costs
    # nothing and it is available here, so it is asked here. The one below
    # stays: it is the honest answer for a denoiser that HAS capture sites
    # and still produced no map, which is a different finding.
    seen = capture_sites(denoiser)
    if seen is not None and not seen.sites:
        raise NotSupported(
            f"{no_capture_site_sentence(seen)} Nothing was generated: this "
            f"was read off the denoiser's own attention blocks before a "
            f"single denoising pass was spent."
        )
    if steps < 1:
        raise BadRequest("a run needs at least one denoising step.")
    requested = int(steps)
    steps = min(requested, MAX_STEPS)

    tokens, padding_from = _tokenize(pipe, prompt)
    if not tokens:
        raise NotSupported(
            "this pipeline has no tokenizer this can read, so the columns of "
            "the map could not be labelled — and an attention map whose "
            "columns are integers is not a map of words."
        )

    # No width is passed: the maps say how wide the conditioning is, and the
    # labels are fitted to them at report time. See `_Collector.n_tokens`.
    store = _Collector()
    original = denoiser.attn_processors
    denoiser.set_attn_processor(_wrap(original, store))

    generator = None
    if seed is not None:
        # CPU generator, moved by the pipeline. CUDA's stream differs for the
        # same seed, so "seed 7" would not be the same image on a machine
        # without a GPU -- and a seed that means different things on different
        # machines is not a seed.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def _tick(_pipe, step_index, timestep, kwargs):
        store.close_step(int(step_index), float(timestep))
        if on_step is not None:
            on_step(int(step_index), steps)
        return kwargs

    try:
        with torch.inference_mode():
            size = {}
            if height:
                size["height"] = int(height)
            if width:
                size["width"] = int(width)
            pipe(
                prompt,
                num_inference_steps=steps,
                generator=generator,
                callback_on_step_end=_tick,
                **size,
            )
    finally:
        # Always restored. A pipeline left with capturing processors attached
        # would keep allocating maps for every later generation in the process
        # -- a memory leak whose symptom is a slow OOM nobody connects to a
        # panel they opened once.
        denoiser.set_attn_processor(original)

    if not store.steps:
        raise NotSupported(
            "the run finished without capturing a single cross-attention map. "
            "This denoiser may attend to its conditioning somewhere this does "
            "not reach, which is a gap in coverage rather than a property of "
            "the model."
        )

    # The labels are fitted to the maps, not the other way round. `_tokenize`
    # caps its labels at `MAX_TOKENS`, and the captured width is whatever the
    # denoiser's conditioning actually is — 120 on PixArt-Alpha, 300 on Sigma.
    # Where the maps are wider than the labels, the extra columns are REAL
    # measurements with no word to put on them, so they are dropped from the
    # per-token vector and the number of them is reported. Silently keeping
    # them would put unlabelled numbers in a map whose columns are supposed to
    # be words; silently dropping them would be a cap nobody was told about.
    width = store.n_tokens or len(tokens)
    columns_cut = max(0, width - len(tokens))
    labels = tokens[:width]
    if columns_cut:
        for step in store.steps:
            step.per_token = step.per_token[: len(labels)]

    return AttentionRun(
        tokens=labels,
        steps=store.steps,
        seed=seed,
        model=public_model_name(pipe, model_name),
        padding_from=padding_from,
        steps_requested=requested,
        resolutions=sorted(store.resolutions),
        conditioning_width=width,
        columns_unlabelled=columns_cut,
    )


def _denoiser_of(pipe):
    """The module that denoises, whatever this pipeline calls it."""
    for name in ("unet", "transformer", "denoiser"):
        found = getattr(pipe, name, None)
        if found is not None:
            return found
    return None


def _tokenize(pipe, prompt: str) -> tuple[list[str], int]:
    """The prompt's tokens as strings, and where the padding starts.

    `(["a","cat"], 2)` — everything from index 2 is `<pad>`. The padded tail
    attracts real attention mass, which is a genuine finding and a terrible
    chart, so it travels as an index rather than as sixty blank columns.
    """
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return [], 0
    try:
        encoded = tokenizer(
            prompt,
            padding="max_length",
            max_length=getattr(tokenizer, "model_max_length", MAX_TOKENS),
            truncation=True,
            return_tensors=None,
        )
        ids = encoded["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        return [], 0

    real = len(tokenizer(prompt, truncation=True)["input_ids"])
    return [str(t) for t in tokens[:MAX_TOKENS]], min(real, len(tokens))


class _Collector:
    """Accumulates one step's maps, reduced on the spot.

    Reduced immediately and deliberately. A full map is
    (heads x pixels x tokens) per block per step; at 50 steps over a dozen
    blocks that is gigabytes, and holding it to average later would mean
    holding all of it at once. So each map is summed over pixels and averaged
    over heads as it arrives, and only the (tokens,) vector survives.

    The reduction is a CHOICE and is stated in `means()`: a mean over heads
    hides head-level disagreement, exactly as it would on the text side.
    """

    def __init__(self, n_tokens: int | None = None) -> None:
        #: The conditioning width, LEARNED from the first map rather than
        #: taken from the tokenizer.
        #:
        #: This was `len(tokens)`, and `_tokenize` slices its labels to
        #: `MAX_TOKENS = 77`. So the label cap became the acceptance test for
        #: every captured map: on a pipeline whose conditioning is wider —
        #: PixArt-Alpha is 120, PixArt-Sigma 300, both advertised as having
        #: cross-attention — every block of every step was silently dropped,
        #: `steps` finished empty, and `capture` raised "the run finished
        #: without capturing a single cross-attention map. This denoiser may
        #: attend to its conditioning somewhere this does not reach."
        #:
        #: Which is a claim about the model, made after the reader has paid
        #: for a full generation, about a model that attends exactly where
        #: this looks. A tokenizer-derived count cannot be the gate:
        #: `tokenizer.model_max_length` (T5: 512) is not the pipeline's
        #: `max_sequence_length` (300) either.
        self.n_tokens = n_tokens
        self.steps: list[StepMap] = []
        self.resolutions: set[int] = set()
        self._running = None
        self._blocks = 0

    def add(self, probs, tokens_axis: int) -> None:
        """One block's attention probabilities, already (batch*heads, q, k)."""
        import torch

        if probs.ndim != 3:
            return
        width = int(probs.shape[tokens_axis])
        if self.n_tokens is None:
            self.n_tokens = width
        elif width != self.n_tokens:
            # A genuine inconsistency — two different conditioning widths in
            # one run — and still a reason to skip. This is the check the
            # original was reaching for; it was just comparing against the
            # wrong number.
            return
        self.resolutions.add(int(probs.shape[1]))
        # Sum over query positions (pixels), mean over heads. float32 because
        # a bf16 sum over 4096 pixels loses the small values, which are
        # exactly the ones a "this word did nothing" reading depends on.
        reduced = probs.to(torch.float32).sum(dim=1).mean(dim=0)
        self._running = reduced if self._running is None else self._running + reduced
        self._blocks += 1

    def close_step(self, index: int, timestep: float) -> None:
        if self._running is None:
            return
        values = (self._running / max(1, self._blocks)).detach().cpu().tolist()
        self.steps.append(
            StepMap(
                step=index,
                timestep=timestep,
                per_token=[float(v) for v in values],
                blocks=self._blocks,
            )
        )
        self._running = None
        self._blocks = 0


def is_capture_site(name: str) -> bool:
    """Whether this processor key names a block the capture records from.

    THE one rule, and it is a function because two callers need it. `_wrap`
    asks it while installing; `capture_sites` asks it before anything is
    installed, so that a denoiser with no such block can be refused for free
    rather than after a generation. A list of names re-typed on the other
    side of a wire is the defect this project's own tests exist to catch.

    `attn2` is diffusers' consistent name for the block whose KEYS are the
    conditioning, across UNet and DiT alike. Self-attention among latents
    answers a different question — which pixels look at which pixels — and
    averaging the two into one map would combine two quantities that are not
    the same quantity.
    """
    return ".attn2." in name or name.endswith("attn2.processor")


@dataclass(frozen=True)
class CaptureSites:
    """What a walk of one denoiser's attention blocks found."""

    #: The denoiser's class, because that is what any sentence about this is
    #: about. Not the repo id: two checkpoints of the same architecture have
    #: the same answer, and naming the checkpoint reads as a fault in it.
    denoiser: str
    #: Every block `capture` would record from, by processor key.
    sites: tuple[str, ...] = ()
    #: Every attention block the walk saw, sites and otherwise. Carried
    #: because "none of them is a cross-attention block" is a claim that
    #: needs a denominator: zero out of zero and zero out of fifty-seven are
    #: different findings and the second is the one worth a sentence.
    blocks: int = 0


def capture_sites(denoiser) -> CaptureSites | None:
    """Where `capture` would find cross-attention, or `None` for "cannot say".

    THREE-VALUED, for the reason `image_steps._call_surface` is: the answer
    decides whether to REFUSE, so absence has to mean absence. A denoiser
    this cannot walk proves nothing about whether it has cross-attention, and
    a two-valued version returning `()` for both "no cross-attention blocks"
    and "could not look" would silently withhold the map from every wrapper
    around a working pipeline.

    `attn_processors` is a walk of `named_children` — no forward pass, no
    map allocated — so this is free at the moment the pipeline object exists,
    which is what makes a preflight possible at all.

    ## Why the NAME is not enough on its own

    SD 3.5 Large sets `dual_attention_layers`, and those blocks carry an
    `attn2` that `is_capture_site` matches. It is a plain SELF-attention over
    the image stream (`cross_attention_dim=None`) called with no
    `encoder_hidden_states`, so `_Capturing` records nothing from it — and a
    name-only test would keep the capability and hand the reader the same
    empty capture this exists to prevent. So each matching block is asked
    whether it is a cross-attention, through diffusers' own
    `is_cross_attention` flag. A block that cannot be asked still counts:
    the name matched, that is the only evidence there is, and withholding on
    a question nobody could put is a guess.
    """
    if denoiser is None:
        return None
    try:
        # `getattr(..., None)` is NOT enough on its own: `attn_processors` is
        # a PROPERTY, so a walk that raises anything other than AttributeError
        # comes straight back out through the default. And it is somebody
        # else's recursive walk over somebody else's modules — one raising is
        # a fact about this build's ability to look, not about the
        # architecture, so it lands on `None` with the rest of them.
        processors = getattr(denoiser, "attn_processors", None)
        if processors is None:
            return None
        names = list(processors)
    except Exception:
        return None

    # AN EMPTY MAP IS NOT AN ANSWER. A walk that succeeds and finds no
    # attention block at all has told us about this surface, not about the
    # architecture — and the sentence this would license says the denoiser
    # mixes text and image in joint attention blocks, which would be a
    # diagnosis invented out of nothing. `hasattr(set_attn_processor)` above
    # and the empty-capture refusal in `capture` both still apply to such a
    # denoiser, and both are honest about what they know.
    if not names:
        return None

    # One walk for the modules, matched back to the keys by path. Cheap for
    # the same reason the processor walk is, and skipped entirely when the
    # denoiser does not offer one.
    try:
        modules = dict(denoiser.named_modules())
    except Exception:
        modules = {}

    sites = []
    for name in names:
        if not is_capture_site(name):
            continue
        block = modules.get(name.removesuffix(".processor"))
        if getattr(block, "is_cross_attention", True) is False:
            continue
        sites.append(name)
    return CaptureSites(
        denoiser=type(denoiser).__name__, sites=tuple(sites), blocks=len(names)
    )


def no_capture_site_sentence(seen: CaptureSites) -> str:
    """Why a denoiser with no capture site has no word-to-pixel map.

    ONE authored sentence for two readers, and they need the same words: the
    loaded status publishes it as `unavailable["cross_attention"]` so the
    control is absent with a reason, and `capture` raises it when the route
    is called anyway. Two hand-written versions of an architectural fact
    drift, and the drift is invisible because each reader only ever sees one.

    Written for the case `seen.sites` is empty; the callers check first.
    `capture_sites` never returns a `CaptureSites` with no blocks in it at
    all, so the count below is always at least one and the sentence never
    claims something about a denoiser it saw nothing of.

    What it may not do is repeat the hedge in the post-hoc refusal at the end
    of `capture` ("may attend to its conditioning somewhere this does not
    reach"). That
    sentence is what a reader gets after paying for a generation, and its
    vagueness is the reason they go looking for a newer build. This one is a
    fact established before a pass was spent, and it says so.
    """
    blocks = f"{seen.blocks} attention block{'' if seen.blocks == 1 else 's'}"
    return (
        f"`{seen.denoiser}` has {blocks} and "
        f"none of them is a cross-attention block. diffusers names that block "
        f"`attn2` and it is the only place the prompt appears as attention "
        f"KEYS; a JOINT-attention denoiser has none by construction, because "
        f"the prompt and the image are concatenated into one sequence and "
        f"attended over together. So there is no word-to-pixel matrix here to "
        f"average, and the columns of one pulled out of the joint map would be "
        f"image patches as often as words. The model is not at fault and the "
        f"images it makes are unaffected — it attends to the prompt, and this "
        f"build cannot yet separate that out of a joint map. The knockout can: "
        f"it removes a word and regenerates at the same seed, which answers "
        f"the causal version of the same question on this checkpoint."
    )


def _wrap(processors: dict, store: _Collector) -> dict:
    """A capturing processor for every CROSS-attention block, and only those.

    EVERY key comes back, wrapped or not: `set_attn_processor` raises when the
    dict it is handed is shorter than the module's own processor count, so
    returning only the wrapped subset would break every pipeline.

    The membership test is `is_capture_site` rather than a condition written
    out here, so that the preflight in `capture` and the one in
    `image_runtime._measurable` can never come to answer a different question
    from the one this installs on.
    """
    wrapped = {}
    for name, processor in processors.items():
        if is_capture_site(name):
            wrapped[name] = _Capturing(processor, store)
        else:
            wrapped[name] = processor
    return wrapped


class _Capturing:
    """Computes the probabilities, records them, then delegates the real work.

    The attention weights are computed HERE rather than recovered afterwards,
    for the same reason `runtime.py` forces eager attention on the text side:
    SDPA never materialises the matrix, and anything reconstructed later is a
    different quantity from the one the model used.

    The cost is one extra QK product per cross-attention block. That is real
    and is why this is a capture mode rather than always-on.
    """

    def __init__(self, inner, store: _Collector) -> None:
        self.inner = inner
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        if encoder_hidden_states is not None:
            try:
                self._record(attn, hidden_states, encoder_hidden_states)
            except Exception as err:
                # A capture that fails must never break the generation it is
                # observing. The step simply contributes no map, and `blocks`
                # on that step says how many did — so a partial capture is
                # visible in the DATA rather than silent.
                #
                # Logged, not swallowed, because "why does this model produce
                # 4 blocks per step when that one produces 12" is a real
                # question and the answer lives here. The type only: this is
                # a third-party module's exception and its text is not ours.
                log.debug(
                    "cross-attention capture skipped a block (%s)",
                    type(err).__name__,
                )
        return self.inner(
            attn, hidden_states, encoder_hidden_states=encoder_hidden_states, **kwargs
        )

    def _record(self, attn, hidden_states, encoder_hidden_states) -> None:
        import torch

        query = attn.to_q(hidden_states)
        key = attn.to_k(
            attn.norm_encoder_hidden_states(encoder_hidden_states)
            if getattr(attn, "norm_cross", None)
            else encoder_hidden_states
        )
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        with torch.inference_mode():
            probs = attn.get_attention_scores(query, key, None)
        self.store.add(probs, tokens_axis=2)


# The most words one request may MARK. Not a bound on the work: `knockout`
# derives its arms from `prompt.split()` and runs one for every word, so the
# cost is the prompt's length and this list only says which rows the caller
# asked about.
#
# Named and published on the image status rather than only enforced at the
# route, because a panel that lets somebody pick twenty-five words and then
# hands them a validation error has charged them the picking before mentioning
# the limit.
MAX_KNOCKOUT_WORDS = 24


def knockout(
    pipe,
    prompt: str,
    *,
    tokens: list[str],
    seed: int,
    steps: int = 20,
    on_arm=None,
) -> dict:
    """Remove one prompt word at a time and measure what actually moved.

    The interventional counterpart to `capture`, and the reason this module
    does not stop at a heatmap. A word can be attended to and change nothing.

    The SEED does the work. Every arm runs at the identical seed, so the
    difference between two images is the word rather than the sampler — and
    without that the number is noise wearing a label. A seed is therefore
    required rather than optional here, which is the one place in this module
    where that is true.
    """
    import torch

    words = [w for w in prompt.split() if w.strip()]
    if len(words) < 2:
        raise BadRequest(
            "a one-word prompt has nothing to knock out — removing the only "
            "word measures the unconditional model, not the effect of a word."
        )

    def render(text: str):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.inference_mode():
            out = pipe(text, num_inference_steps=steps, generator=generator)
        return _latent_of(out)

    base = render(prompt)
    arms = []
    for i, word in enumerate(words):
        if on_arm is not None:
            on_arm(i, len(words))
        without = " ".join(words[:i] + words[i + 1 :])
        moved = _distance(base, render(without))
        arms.append(
            {
                "word": word,
                "index": i,
                "prompt_without": without,
                "distance": round(moved, 6),
            }
        )
    arms.sort(key=lambda a: a["distance"], reverse=True)

    return {
        "arms": arms,
        "seed": seed,
        "steps": steps,
        "tokens": list(tokens),
        "means": (
            f"Each row is how far the image moved when that ONE word was "
            f"removed and the image regenerated at seed {seed}. "
            # Through `fmt.measured`, because `:,.4f` floored a real distance
            # to 0.0000 — so the sentence naming the word that moved the image
            # FURTHEST reported it as having moved it by nothing, directly
            # under a row that read 3.0e-5. One quantity, two formatters.
            f"'{arms[0]['word']}' moved it furthest "
            f"({fmt.measured(arms[0]['distance'])}).\n\n"
            f"THE SEED IS DOING THE WORK. Every arm ran at the identical seed, "
            f"so the difference is the word rather than the sampler. At a "
            f"different seed per arm these numbers would be sampling noise "
            f"with a word's name on them.\n\n"
            f"REMOVING A WORD IS NOT SILENCING IT. The remaining prompt is a "
            f"different, shorter sentence, and the model conditions on all of "
            f"it — so a large number means 'this prompt without that word is a "
            f"different prompt', which is not quite the same as 'that word "
            f"caused this'."
        ),
    }


def _latent_of(output):
    """Pixels from whatever the pipeline returned, as a float array."""
    import numpy as np

    images = getattr(output, "images", None) or []
    if not images:
        raise NotSupported("the pipeline returned no image to compare.")
    first = images[0]
    return np.asarray(first, dtype="float32") / 255.0


def _distance(a, b) -> float:
    """RMS difference between two images.

    RMS rather than a perceptual metric on purpose: a perceptual score is a
    model's opinion, and this project does not put one model's judgement
    inside another model's measurement without saying so. RMS is arithmetic
    and everyone can check it.
    """
    import numpy as np

    if a.shape != b.shape:
        raise NotSupported(
            f"the two renders are {a.shape} and {b.shape}, so they cannot be "
            f"differenced."
        )
    return float(math.sqrt(float(np.mean((a - b) ** 2))))


def plan(steps: int, words: int) -> dict:
    """What a knockout will cost, before it costs it."""
    arms = words + 1
    return {
        "arms": arms,
        "steps_each": steps,
        "passes": arms * steps,
        "means": (
            f"{arms} renders ({words} words plus the unmodified prompt) at "
            f"{steps} steps each — {arms * steps} denoising passes. No seconds "
            f"are quoted because this machine has not been timed on this "
            f"pipeline."
        ),
    }
