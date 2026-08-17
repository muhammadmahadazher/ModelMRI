"""Every lerobot call in this package, in one file.

lerobot's API churns — this is not a complaint, it is the design constraint
the whole sidecar exists to absorb. Between 0.4 and 0.6 the policy factory
moved from `lerobot.common.policies` to `lerobot.policies`, and normalisation
stopped being modules hanging off the policy (`policy.normalize_targets`) and
became a processor PIPELINE built separately by `make_pre_post_processors`.
Either change would have silently broken a forward pass written against the
older shape.

So the rule here is the one `modelmri/patch.py` already follows for SDK
shapes: when the thing this expects is not there, REFUSE and name what moved.
A forward pass that quietly falls back to raw, unnormalised tensors does not
fail — it returns an action chunk in the wrong units, and an action chunk in
the wrong units is indistinguishable downstream from a policy that has learnt
something strange.

Confining lerobot to this file is what makes that checkable: `inputs.py` and
`server.py` are testable without a six-gigabyte environment, and everything
version-fragile is here where a single `shape_report()` can describe it.

Verified against **lerobot 0.6.1** with torch 2.11.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ShapeMoved(RuntimeError):
    """lerobot is not the shape this was written against, and it says which part."""


@dataclass
class Loaded:
    """A policy and everything needed to describe what it consumes and emits."""

    policy: object = None
    preprocessor: object = None
    postprocessor: object = None
    repo: str = ""
    revision: str = ""
    device: str = ""
    dtype: str = ""
    family: str = ""
    lerobot_version: str = ""
    torch_version: str = ""
    # The camera keys this policy was trained with, in the order its config
    # lists them. A request missing one is refused rather than blank-filled.
    cameras: list[str] = field(default_factory=list)
    # The frame shape each camera was trained at, when the checkpoint says.
    # Empty for a camera that does not publish one -- and empty means "it did
    # not say", never "any size is fine".
    camera_shapes: dict = field(default_factory=dict)
    # The width of the proprioceptive state vector, or None when the policy
    # consumes no state. None is NOT zero: "this policy takes no state" and
    # "the state is empty" are different facts.
    state_width: int | None = None
    action_width: int | None = None
    chunk_size: int | None = None
    # The statistics the ACTIONS are expressed against, read off the
    # postprocessor. Empty means the policy did not publish them, and empty
    # must be treated by every caller as "do not overlay" rather than as
    # identity scaling. See ROADMAP #50.
    normalisation: dict = field(default_factory=dict)
    # Whether the policy samples. A flow-matching or diffusion policy takes
    # noise and gives a different chunk per seed; ACT and VQ-BeT do not.
    # ROADMAP #50's instruction-swap test uses the policy's OWN sampling
    # variance as its reference, and that reference collapses to zero for a
    # deterministic policy — which is a refusal, not a result of 0.
    samples: bool = False

    def describe(self) -> dict:
        return {
            "policy_repo": self.repo,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "family": self.family,
            "lerobot_version": self.lerobot_version,
            "torch_version": self.torch_version,
            "cameras": list(self.cameras),
            "camera_shapes": dict(self.camera_shapes),
            "state_width": self.state_width,
            "action_width": self.action_width,
            "chunk_size": self.chunk_size,
            "normalisation": dict(self.normalisation),
            "samples": self.samples,
        }


# Families whose action head is stochastic: they take a `noise` tensor and
# return a different chunk for a different seed. Listed rather than probed,
# because probing means two forward passes on load and the answer is a fact
# about the architecture rather than about this checkpoint.
#
# A family NOT in this list is reported as deterministic, and the honest
# consequence is that #50's instruction-swap test refuses on it: its reference
# is the policy's own sampling spread, and a zero reference does not become a
# valid one by dividing by it.
SAMPLING_FAMILIES = frozenset(
    {
        "smolvla",
        "pi0",
        "pi05",
        "pi0_fast",
        "diffusion",
        "eo1",
        "evo1",
        "groot",
        "xvla",
        "pi_gemma",
        "fastwam",
        "multi_task_dit",
    }
)


def _need(module_path: str, name: str, why: str):
    """Import one lerobot symbol, or refuse naming it and what it was for."""
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as err:
        raise ShapeMoved(
            f"this sidecar's lerobot has no `{module_path}`, which is where "
            f"{why} ({type(err).__name__}). The package layout has moved; "
            f"rebuild the sidecar with `modelmri policy install --force`, and "
            f"if that does not fix it the adapter needs updating for this "
            f"lerobot."
        ) from None
    attr = getattr(module, name, None)
    if attr is None:
        raise ShapeMoved(
            f"`{module_path}` exists but has no `{name}`, which is what "
            f"{why}. Rather than guess at a replacement, this refuses: a "
            f"forward pass built on a guessed API returns numbers, and wrong "
            f"numbers here look exactly like right ones."
        )
    return attr


def versions() -> tuple[str, str]:
    """What this environment actually holds, for the receipt on every answer.

    `""` for a package that is not installed, and it must never raise. This
    used to `import lerobot` directly, which meant `shape_report()` -- the one
    function whose whole job is to say what is wrong with a sidecar -- could
    not run on the most broken sidecar there is, the one with no lerobot in
    it. A diagnostic that fails on the failure it diagnoses is not one.

    Empty is a real answer and reads as one: nothing displays "" as a version,
    and `PolicyStatus.accelerated()` already returns None rather than False
    when the torch build is unknown.
    """
    return _installed_version("lerobot"), _installed_version("torch")


def _installed_version(name: str) -> str:
    import importlib

    try:
        module = importlib.import_module(name)
    except ImportError:
        return ""
    return str(getattr(module, "__version__", "") or "")


def _features_of(cfg) -> tuple[list[str], dict, int | None, int | None]:
    """Cameras, state width and action width, read off the policy's own config.

    Read rather than assumed. The three hardcoded SmolVLA values that
    `vla.py`'s comment records — tensor prefix, vision config repo, module
    class — are exactly the kind of thing that makes a tool a one-policy
    viewer, and the config already says all of it.
    """
    types = _need(
        "lerobot.configs.types",
        "FeatureType",
        "the kinds of input a policy declares are enumerated",
    )
    inputs = getattr(cfg, "input_features", None) or {}
    outputs = getattr(cfg, "output_features", None) or {}

    # Name AND shape. The shape is published right beside the name and was
    # being dropped, which left the one thing a caller needs in order to send
    # a correctly sized frame unavailable — while the state and action widths
    # two lines below were read from the identical field. A resolution the
    # checkpoint states and this discards is a resolution the caller has to
    # guess, and a wrongly sized frame is resized silently by the preprocessor
    # rather than refused.
    cameras = []
    camera_shapes: dict[str, list[int]] = {}
    for key, feat in inputs.items():
        if getattr(feat, "type", None) is not types.VISUAL:
            continue
        cameras.append(key)
        shape = tuple(getattr(feat, "shape", ()) or ())
        if shape:
            camera_shapes[key] = [int(n) for n in shape]
    state_width = None
    for feat in inputs.values():
        if getattr(feat, "type", None) is types.STATE:
            shape = tuple(getattr(feat, "shape", ()) or ())
            state_width = int(shape[0]) if shape else None
            break
    action_width = None
    for feat in outputs.values():
        if getattr(feat, "type", None) is types.ACTION:
            shape = tuple(getattr(feat, "shape", ()) or ())
            action_width = int(shape[0]) if shape else None
            break
    return cameras, camera_shapes, state_width, action_width


def _normalisation_of(postprocessor) -> dict:
    """The statistics the actions are expressed in, or `{}` when unpublished.

    In lerobot 0.6 this lives on an `UnnormalizerProcessorStep` inside the
    postprocessor pipeline rather than on the policy. Walked rather than
    indexed: the pipeline's composition is not part of any contract, and a
    step at a fixed position is a guess that stops being true.

    `{}` is a real answer and the caller must not read it as identity. A
    policy whose action space is unnormalised and one that never said are the
    same bytes and completely different claims.
    """
    steps = getattr(postprocessor, "steps", None)
    if steps is None:
        return {}
    for step in steps:
        if type(step).__name__ != "UnnormalizerProcessorStep":
            continue
        stats = getattr(step, "stats", None)
        if not stats:
            continue
        out: dict = {}
        for feature, values in stats.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                try:
                    flat = [float(x) for x in value.flatten().tolist()]
                except AttributeError:
                    try:
                        flat = [float(x) for x in value]
                    except (TypeError, ValueError):
                        continue
                except (TypeError, ValueError):
                    continue
                out.setdefault(str(feature), {})[str(key)] = flat
        return out
    return {}


def load(repo: str, *, device: str = "") -> Loaded:
    """Bring a policy up, and read off it what it consumes.

    Everything reported here comes from the checkpoint: the family, the
    cameras, the state width, the action width, the chunk size and the
    normalisation. Nothing is defaulted to SmolVLA's values, because a default
    that happens to be right for one policy is a wrong answer for every other
    one and never says which it is being.
    """
    import torch

    PreTrainedConfig = _need(
        "lerobot.configs.policies",
        "PreTrainedConfig",
        "a checkpoint's own config is read before its weights",
    )
    get_policy_class = _need(
        "lerobot.policies.factory",
        "get_policy_class",
        "a family name becomes the class that can load it",
    )
    make_pre_post = _need(
        "lerobot.policies.factory",
        "make_pre_post_processors",
        "input normalisation and action un-normalisation are built",
    )

    cfg = PreTrainedConfig.from_pretrained(repo)
    family = str(getattr(cfg, "type", "") or "")
    if not family:
        raise ShapeMoved(
            f"the config at {repo} does not say which policy family it is, so "
            f"there is nothing here that could choose a class to load it with."
        )

    want = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy_cls = get_policy_class(family)
    policy = policy_cls.from_pretrained(repo)
    policy.eval()
    policy.to(want)

    if not hasattr(policy, "predict_action_chunk"):
        raise ShapeMoved(
            f"the {family} policy has no `predict_action_chunk`, which is how "
            f"a whole action chunk is asked for. Only a single-step "
            f"`select_action` would be available, and returning one step "
            f"labelled as a chunk would be a different measurement under the "
            f"same name."
        )

    # The device has to be forced into the PROCESSORS too, and this is not a
    # detail -- it is the bug that running this found.
    #
    # `PreTrainedConfig.from_pretrained` corrects its own `device` when the
    # requested one is unavailable ("Device 'cuda' is not available. Switching
    # to 'cpu'"). The processor pipelines do not go through that: they are
    # rebuilt from a SEPARATE json saved beside the weights, which still says
    # whatever device the checkpoint was trained on. On a CPU-only machine
    # that raises outright:
    #
    #     ValueError: Failed to instantiate processor step 'device_processor'
    #     with config: {'device': 'cuda', ...}
    #
    # The louder failure is the one that does not raise. On a machine that HAS
    # cuda, asking for `cpu` would leave the policy on the processor and the
    # preprocessor moving every tensor to the card -- one question, two
    # answers, and whichever way it resolved would not be the device the
    # caller asked for or the one the answer got labelled with.
    device_override = {"device_processor": {"device": str(want)}}
    pre, post = make_pre_post(
        cfg,
        pretrained_path=repo,
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    cameras, camera_shapes, state_width, action_width = _features_of(cfg)

    return Loaded(
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        repo=repo,
        revision=_revision_of(repo),
        device=str(want),
        dtype=str(next(policy.parameters()).dtype).removeprefix("torch."),
        family=family,
        lerobot_version=versions()[0],
        torch_version=versions()[1],
        cameras=cameras,
        camera_shapes=camera_shapes,
        state_width=state_width,
        action_width=action_width,
        chunk_size=int(getattr(cfg, "chunk_size", 0) or 0) or None,
        normalisation=_normalisation_of(post),
        samples=family in SAMPLING_FAMILIES,
    )


def _revision_of(repo: str) -> str:
    """The exact snapshot, or "" when unknowable.

    Empty rather than "unknown" or "main": somebody comparing two runs needs
    to tell "these are the same weights" from "nobody recorded which weights",
    and a placeholder string collapses those into one.
    """
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo).sha or "")
    except Exception:
        return ""


def act(
    loaded: Loaded,
    *,
    frames: dict,
    state=None,
    instruction: str = "",
    seed: int | None = None,
) -> dict:
    """One forward pass: what this policy would DO on this frame.

    The seed is applied by seeding torch inside a `fork_rng`, and the POLICY
    makes its own noise. An earlier version built the noise tensor here and
    passed it in — and its shape was a guess. It is not
    `(1, chunk_size, action_width)`: SmolVLA projects through
    `max_action_dim`, so a six-wide action space needs THIRTY-TWO-wide noise,
    and the guess died as

        RuntimeError: mat1 and mat2 shapes cannot be multiplied (50x6 and 32x720)

    Crashing is the good case. A family whose internal width happened to match
    its declared one would have accepted the tensor and sampled from a
    distribution nobody chose, and the chunk would have looked like every
    other chunk. Guessing the shape of a tensor the model multiplies is
    exactly the guess this project refuses, so it no longer guesses: the
    policy knows its own noise shape and makes it.

    The usual objection to seeding globally — that it is reproducible only if
    nothing else in the process draws from the same generator — is answered by
    the caller. `server._act` holds the policy lock across this call, so no
    second forward pass is drawing concurrently, and `fork_rng` puts the
    previous state back afterwards so nothing outside sees the seeding at all.

    A deterministic policy given a seed is told so in the response rather than
    having the seed quietly ignored — `seed_used` is None and `samples` is
    False, so a caller building #50's instruction-swap reference can see that
    its denominator would be zero before it divides by it.
    """
    import torch

    batch = _batch(loaded, frames=frames, state=state, instruction=instruction)

    seed_used = int(seed) if (seed is not None and loaded.samples) else None

    # Which generators `fork_rng` has to save and restore. CPU always; the
    # specific CUDA device too when the policy is on one, because
    # `torch.manual_seed` seeds every CUDA generator and leaving one reseeded
    # would make the NEXT unseeded request quietly reproducible — a result
    # that looks measured and is actually a leftover.
    devices = []
    if loaded.device.startswith("cuda"):
        devices = [torch.device(loaded.device).index or 0]

    with torch.inference_mode():
        prepared = loaded.preprocessor(batch)
        if seed_used is None:
            chunk = loaded.policy.predict_action_chunk(prepared)
        else:
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.manual_seed(seed_used)
                chunk = loaded.policy.predict_action_chunk(prepared)
        chunk = loaded.postprocessor(chunk)

    if not hasattr(chunk, "detach"):
        raise ShapeMoved(
            f"the {loaded.family} policy returned "
            f"{type(chunk).__name__} rather than a tensor, so there is no "
            f"action chunk here to read."
        )
    values = chunk.detach().to("cpu", dtype=torch.float32)
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]

    return {
        "action_chunk": values.tolist(),
        "shape": list(values.shape),
        "seed_used": seed_used,
        "samples": loaded.samples,
        **loaded.describe(),
    }


def _batch(loaded: Loaded, *, frames: dict, state, instruction: str) -> dict:
    """The observation dict lerobot's preprocessor expects.

    Images arrive HWC uint8 and go in as CHW float in [0, 1] with a batch
    dimension — the layout every lerobot vision tower is written against.
    Getting this wrong does not raise: a HWC tensor read as CHW is an image
    with three rows and a great many channels, which convolves to something,
    and the something looks like an answer.
    """
    import numpy as np
    import torch

    batch: dict = {}
    for name, array in frames.items():
        # `copy=True`, and torch says why: a PIL-derived array is read-only,
        # and `torch.from_numpy` on one produces a tensor torch believes it
        # may write to. It warns that writing is undefined behaviour. Nothing
        # here writes today -- the `div_` below lands on the fresh float
        # tensor `.to()` returns, not on this one -- but "undefined behaviour
        # that happens to be unreached" is one refactor away from a frame
        # being silently corrupted between decoding and the policy seeing it.
        tensor = torch.from_numpy(np.array(array, dtype=np.uint8, copy=True))
        tensor = tensor.permute(2, 0, 1).to(torch.float32).div_(255.0)
        batch[name] = tensor.unsqueeze(0).to(loaded.device)

    if state is not None:
        key = _state_key(loaded)
        batch[key] = (
            torch.from_numpy(np.ascontiguousarray(state))
            .to(torch.float32)
            .unsqueeze(0)
            .to(loaded.device)
        )

    # The task string travels as a list of one, matching the batch dimension.
    # An empty string is a CONDITION — #50 runs a "no instruction" arm on
    # purpose — so it is passed through rather than dropped.
    batch["task"] = [instruction]
    return batch


def _state_key(loaded: Loaded) -> str:
    """Whatever this policy calls its state input, from its own config."""
    types = _need(
        "lerobot.configs.types",
        "FeatureType",
        "the kinds of input a policy declares are enumerated",
    )
    cfg = getattr(loaded.policy, "config", None)
    for key, feat in (getattr(cfg, "input_features", None) or {}).items():
        if getattr(feat, "type", None) is types.STATE:
            return key
    raise ShapeMoved(
        "a state vector was supplied but this policy's config declares no "
        "state input, so there is no key to put it under. Feeding it in under "
        "a guessed name would be inventing an observation."
    )


def shape_report() -> dict:
    """What this adapter found, for `modelmri policy status` to print.

    Every symbol checked by name. When lerobot moves again this is the thing
    that says WHICH part moved, instead of a traceback from inside a forward
    pass three calls deep.
    """
    checks = [
        ("lerobot.configs.policies", "PreTrainedConfig"),
        ("lerobot.configs.types", "FeatureType"),
        ("lerobot.policies.factory", "get_policy_class"),
        ("lerobot.policies.factory", "make_pre_post_processors"),
        ("lerobot.processor.normalize_processor", "UnnormalizerProcessorStep"),
    ]
    found: dict = {}
    for module_path, name in checks:
        try:
            _need(module_path, name, "this adapter is written against it")
            found[f"{module_path}.{name}"] = True
        except ShapeMoved:
            found[f"{module_path}.{name}"] = False
    lerobot_version, torch_version = versions()
    return {
        "lerobot": lerobot_version,
        "torch": torch_version,
        "symbols": found,
        "intact": all(found.values()),
    }
