"""What reaches a policy has to be what the policy was trained on.

The refusals in `modelmri_policy.inputs` all guard the same failure, and it is
the quietest one available in this whole project: a VLA handed the wrong input
mostly does not crash. It broadcasts, pads, takes the first channel, or shifts
a joint vector by one — and returns an action chunk that is indistinguishable
from a measurement.

Downstream there is no way to tell. A chunk of near-zeros reads as a policy
deciding to hold still. A chunk computed from a black frame reads as a policy
with nothing to do. ROADMAP #50 then draws that curve against the dataset's
recorded actions, and somebody concludes something about a robot.

So every one of these is a refusal rather than a best effort, and every test
here is a specific way the wrong thing could have got through.

Nothing here needs lerobot, a GPU or a checkpoint — the rules are about PNG
bytes, array shapes and dict keys, which is exactly why they live in their own
module instead of inside the forward pass.
"""

from __future__ import annotations

import base64
import io

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from modelmri_policy import inputs  # noqa: E402


def _png(width=8, height=6, colour=(10, 20, 30), mode="RGB") -> str:
    """A real PNG, encoded the way a request carries one."""
    img = Image.new(mode, (width, height), colour if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------- frames


def test_a_real_png_decodes_to_hwc_uint8():
    got = inputs.decode_frame(_png(8, 6), camera="top")
    assert got.shape == (6, 8, 3)
    assert got.dtype == np.uint8
    # The colour actually survives, so this is a decode and not a shape check
    # over a blank array.
    assert tuple(got[0, 0]) == (10, 20, 30)


def test_a_greyscale_png_is_converted_rather_than_passed_through():
    """A single-channel frame reaching a policy expecting three does not fail
    — it gets broadcast or reinterpreted. Converting is the only outcome that
    is the same image."""
    got = inputs.decode_frame(_png(mode="L"), camera="wrist")
    assert got.shape[2] == 3


def test_base64_that_is_not_base64_names_the_camera():
    """A request carries several cameras, so "could not decode the image"
    sends somebody looking at the wrong one."""
    with pytest.raises(inputs.InputError, match="'wrist'"):
        inputs.decode_frame("not base64 at all !!", camera="wrist")


def test_valid_base64_that_is_not_an_image_is_refused():
    payload = base64.b64encode(b"this is text, not a PNG").decode()
    with pytest.raises(inputs.InputError, match="could not be decoded"):
        inputs.decode_frame(payload, camera="top")


def test_a_frame_that_is_not_a_string_is_refused_by_type():
    with pytest.raises(inputs.InputError, match="rather than a base64 image"):
        inputs.decode_frame({"pixels": []}, camera="top")


def test_an_empty_frame_is_refused():
    """Empty is caught before decoding, which is why `decode_frame` has no
    "decoded to zero bytes" branch — the only input that decodes to nothing is
    the empty string, and it never gets that far. A guard that cannot fire
    reads as coverage while testing nothing."""
    with pytest.raises(inputs.InputError, match="rather than a base64 image"):
        inputs.decode_frame("", camera="top")
    # And a single "=" is not valid base64 at all, which is a different
    # sentence again — three inputs, three distinct refusals.
    with pytest.raises(inputs.InputError, match="not valid base64"):
        inputs.decode_frame("=", camera="top")


# ---------------------------------------------------------- the camera SET


def test_no_frames_at_all_is_refused_in_words():
    with pytest.raises(inputs.InputError, match="cannot act without seeing"):
        inputs.read_frames({})


def test_a_missing_camera_is_refused_rather_than_filled_with_black():
    """The substantive one. A policy trained on a wrist view and a top-down
    view, given only the top-down, does not produce a degraded answer — it
    produces a confident answer to a different question."""
    body = {"frames": {"top": _png()}}
    with pytest.raises(inputs.InputError) as caught:
        inputs.read_frames(body, expected=["top", "wrist"])
    said = str(caught.value)
    assert "missing wrist" in said
    assert "different question" in said


def test_an_unexpected_camera_is_refused_rather_than_guessed_into_a_slot():
    body = {"frames": {"top": _png(), "elbow": _png()}}
    with pytest.raises(inputs.InputError, match="does not consume"):
        inputs.read_frames(body, expected=["top"])


def test_the_exact_camera_set_passes():
    body = {"frames": {"top": _png(), "wrist": _png()}}
    got = inputs.read_frames(body, expected=["wrist", "top"])
    assert sorted(got) == ["top", "wrist"]


def test_more_cameras_than_any_robot_has_is_refused_before_decoding():
    body = {"frames": {f"cam{i}": _png() for i in range(inputs.MAX_CAMERAS + 1)}}
    with pytest.raises(inputs.InputError, match="past the"):
        inputs.read_frames(body)


# -------------------------------------------------------------------- state


def test_a_state_vector_of_the_wrong_width_is_refused_with_the_reason():
    """Joint vectors are ORDERED. A six-joint reading fed to a seven-joint
    policy shifts every joint by one and returns a plausible chunk for a robot
    that does not exist."""
    with pytest.raises(inputs.InputError) as caught:
        inputs.read_state({"state": [0.0] * 6}, width=7)
    said = str(caught.value)
    assert "7-wide" in said and "carried 6" in said
    assert "ORDERED" in said


def test_a_state_the_policy_needs_and_the_request_omits_is_refused():
    with pytest.raises(inputs.InputError, match="carried none"):
        inputs.read_state({}, width=7)


def test_no_state_is_fine_when_the_policy_consumes_none():
    assert inputs.read_state({}, width=None) is None


def test_nan_in_the_state_is_refused_rather_than_propagated():
    """NaN through a forward pass comes out the other side as NaN in every
    dimension of the chunk, which reads as a broken policy rather than a bad
    input."""
    with pytest.raises(inputs.InputError, match="NaN or infinity"):
        inputs.read_state({"state": [0.1, float("nan"), 0.3]}, width=3)


def test_infinity_in_the_state_is_refused_too():
    with pytest.raises(inputs.InputError, match="NaN or infinity"):
        inputs.read_state({"state": [0.1, float("inf")]}, width=2)


def test_a_boolean_is_not_a_joint_reading():
    """`isinstance(True, int)` is True in Python, so a bare number check lets
    a boolean through and silently reads it as 1.0 radians."""
    with pytest.raises(inputs.InputError, match="bool"):
        inputs.read_state({"state": [0.1, True]}, width=2)


def test_a_good_state_arrives_as_float32():
    got = inputs.read_state({"state": [0.1, 0.2, 0.3]}, width=3)
    assert got.dtype == np.float32
    assert got.shape == (3,)


# ------------------------------------------------------- instruction and seed


def test_an_empty_instruction_is_a_condition_not_a_missing_field():
    """ROADMAP #50's instruction-swap test runs a "no instruction" arm on
    purpose, and the panel must label it "no instruction" rather than "the
    instruction did not matter". So empty travels as empty."""
    assert inputs.read_instruction({"instruction": ""}) == ""
    assert inputs.read_instruction({}) == ""


def test_an_instruction_that_is_not_text_is_refused():
    with pytest.raises(inputs.InputError, match="rather than text"):
        inputs.read_instruction({"instruction": ["pick", "up"]})


def test_seed_none_and_seed_zero_are_different_requests():
    """0 is a seed. None means "do not touch the RNG", which is the honest
    answer to return when nobody asked for reproducibility. Collapsing them
    would make every unseeded call silently reproducible and hide the fact
    that a policy samples at all."""
    assert inputs.read_seed({}) is None
    assert inputs.read_seed({"seed": None}) is None
    assert inputs.read_seed({"seed": 0}) == 0


def test_true_is_not_a_seed():
    with pytest.raises(inputs.InputError, match="bool"):
        inputs.read_seed({"seed": True})


def test_a_seed_outside_what_a_sampler_takes_is_refused():
    with pytest.raises(inputs.InputError, match="outside the range"):
        inputs.read_seed({"seed": -1})
    with pytest.raises(inputs.InputError, match="outside the range"):
        inputs.read_seed({"seed": 2**31})


# ------------------------------------- the adapter, where lerobot is confined


def test_a_missing_lerobot_is_a_named_refusal_not_a_traceback():
    """Runs in ModelMRI's OWN environment, which deliberately does not have
    lerobot — that is the whole point of the venv separation, and it makes
    this the honest place to test the refusal.

    A forward pass built on a guessed API returns numbers, and wrong numbers
    here look exactly like right ones. So a moved symbol is a refusal that
    names it."""
    from modelmri_policy import adapter

    with pytest.raises(adapter.ShapeMoved) as caught:
        adapter._need(
            "lerobot.configs.policies", "PreTrainedConfig", "a config is read"
        )
    said = str(caught.value)
    assert "lerobot.configs.policies" in said
    assert "install --force" in said


def test_a_symbol_that_moved_inside_a_module_that_stayed_is_named_too():
    """The two failures are different and lead to different fixes: a missing
    module means the package layout moved, a missing attribute means one
    function was renamed."""
    from modelmri_policy import adapter

    with pytest.raises(adapter.ShapeMoved, match="no `nonexistent_symbol_xyz`"):
        adapter._need("json", "nonexistent_symbol_xyz", "something is done")


def test_the_shape_report_is_all_false_without_lerobot_rather_than_raising():
    """`shape_report` exists to SAY which part moved. If it raised, the one
    tool for diagnosing a broken sidecar would be the one thing that could not
    run on a broken sidecar."""
    from modelmri_policy import adapter

    try:
        report = adapter.shape_report()
    except ImportError:
        # `versions()` imports lerobot and torch. In an environment with
        # neither, "cannot report" is itself the answer and is what the
        # sidecar's own /status turns into a sentence.
        pytest.skip("this environment has neither lerobot nor torch to report on")
    assert report["intact"] is False
    assert all(found is False for found in report["symbols"].values())


def test_a_deterministic_family_is_reported_as_deterministic():
    """ROADMAP #50's instruction-swap test measures instruction spread against
    the policy's OWN sampling spread. For ACT and VQ-BeT that reference is
    exactly zero, and a zero reference does not become valid by dividing by
    it — so the family has to travel with the answer."""
    from modelmri_policy import adapter

    assert "act" not in adapter.SAMPLING_FAMILIES
    assert "vqbet" not in adapter.SAMPLING_FAMILIES
    assert "smolvla" in adapter.SAMPLING_FAMILIES
    assert "diffusion" in adapter.SAMPLING_FAMILIES


def test_an_unloaded_description_never_invents_a_width():
    """`None` is the answer for "this policy consumes no state". Zero is a
    width, and a caller that reads zero as a width sends an empty vector to a
    policy that wanted six numbers."""
    from modelmri_policy import adapter

    said = adapter.Loaded().describe()
    assert said["state_width"] is None
    assert said["action_width"] is None
    assert said["chunk_size"] is None
    assert said["cameras"] == []
    assert said["normalisation"] == {}


def test_normalisation_off_a_pipeline_with_no_unnormaliser_is_empty():
    """Empty means "the policy did not publish its action statistics", and
    every caller must read that as "do not overlay" rather than as identity
    scaling."""
    from modelmri_policy import adapter

    class Pipeline:
        steps = [object(), object()]

    assert adapter._normalisation_of(Pipeline()) == {}
    assert adapter._normalisation_of(object()) == {}


def test_normalisation_is_found_wherever_it_sits_in_the_pipeline():
    """Walked rather than indexed. The pipeline's composition is not part of
    any contract, and a step at a fixed position is a guess that stops being
    true on the next lerobot."""
    from modelmri_policy import adapter

    class UnnormalizerProcessorStep:
        stats = {"action": {"mean": [0.0, 1.0], "std": [1.0, 2.0]}}

    class Pipeline:
        steps = [object(), object(), UnnormalizerProcessorStep()]

    got = adapter._normalisation_of(Pipeline())
    assert got == {"action": {"mean": [0.0, 1.0], "std": [1.0, 2.0]}}
