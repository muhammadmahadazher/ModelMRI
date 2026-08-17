"""The image-model routes, and what each of them refuses.

Two refusals matter more than the rest, and they are deliberately different
sentences because they lead to different actions:

  nothing is loaded          -> load a pipeline
  this family has no such
  thing to measure           -> loading another one will not help either

Collapsing those would send somebody to load a second model that also cannot
answer. `imaging.detect` decides which is which, through the handle's
`capabilities` list, so a panel asks rather than infers.

Nothing here loads a pipeline. The point of a cost route is that it answers
before anything is spent, and the point of a refusal is that it arrives first.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from modelmri.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


# --------------------------------------------------------- nothing held yet


def test_the_status_never_raises_and_says_what_is_missing(client):
    """A resting panel asks this on every load. It has to answer even when the
    answer is "nothing", and "loaded: false" alone is not an answer."""
    d = client.get("/api/image").json()
    assert d["loaded"] is False
    assert "Nothing has been loaded yet" in d["means"]
    assert d["capabilities"] == []


def test_every_measurement_refuses_with_the_same_sentence(client):
    """One reason, one wording. Two routes disagreeing about what "not loaded"
    means is the seam where somebody concludes the second one is broken."""
    said = set()
    for path, body in (
        ("/api/image/attention", {"prompt": "a cat", "steps": 4}),
        ("/api/image/knockout", {"prompt": "a cat", "words": ["cat"], "seed": 0}),
    ):
        r = client.post(path, json=body)
        assert r.status_code == 409, path
        said.add(r.json()["error"])
    assert len(said) == 1, "two routes gave two different not-loaded sentences"
    # Unpacked rather than popped. `said.pop()` empties the set it is
    # inspecting, so the check is not repeatable and anything added below it
    # reads an empty set — a set of one is destroyed by looking at it.
    (sentence,) = said
    assert "No image model is loaded" in sentence


# ------------------------------------------------------- cost before spending


def test_the_attention_cost_counts_arms_including_the_unmodified_prompt(client):
    """N words is N+1 renders. Forgetting the baseline under-quotes by exactly
    one render, which is the arm every other arm is compared against."""
    d = client.get("/api/image/attention/cost?steps=20&words=3").json()
    assert d["arms"] == 4
    assert d["passes"] == 80
    assert "plus the unmodified prompt" in d["means"]


def test_no_seconds_are_quoted_for_a_machine_that_has_not_been_timed(client):
    """A duration from somebody else's hardware is the number people plan
    around."""
    said = client.get("/api/image/attention/cost?steps=10&words=1").json()["means"]
    assert "has not been timed" in said
    assert "second" not in said.replace("No seconds", "")


def test_an_unpriceable_trace_reports_null_bytes_rather_than_zero(client):
    """With no pipeline held there is no latent shape to price, and a run
    whose memory could not be computed is not a run that costs nothing."""
    d = client.get("/api/image/steps/cost?steps=20").json()
    assert d["latent_bytes"] is None
    assert d["total_bytes"] is None
    assert d["fits"] is None
    assert d["denoiser_passes"] == 20


def test_the_cost_routes_answer_with_no_pipeline_at_all(client):
    """They exist to be asked FIRST. A cost route that needed the thing it is
    pricing would be useless."""
    assert client.get("/api/image/attention/cost?steps=5&words=2").status_code == 200
    assert client.get("/api/image/steps/cost?steps=5").status_code == 200


# ---------------------------------------------------------------- discovery


def test_the_available_list_reads_this_disk_and_downloads_nothing(client):
    """Whatever is cached on the machine running this. Every entry is either a
    family it named or an unknown that says why — never a bare row."""
    d = client.get("/api/image/available").json()
    assert "Nothing was downloaded" in d["means"]
    for m in d["models"]:
        assert m["path"]
        if m["known"]:
            assert m["capabilities"], f"{m['path']} named a family with no capabilities"
        else:
            assert m["reason"], f"{m['path']} is unknown with no reason"


def test_an_unknown_family_offers_no_capabilities_over_http(client):
    """The capability list is what a panel asks before drawing a control, so
    an unknown family must offer nothing rather than everything."""
    for m in client.get("/api/image/available").json()["models"]:
        if not m["known"]:
            assert m["capabilities"] == []


# ----------------------------------------------------------------- loading


def test_an_empty_repo_is_rejected_by_the_schema(client):
    """No default. The checkpoint decides which panels apply, so guessing one
    silently decides what the user is looking at."""
    assert client.post("/api/image/load", json={"repo": ""}).status_code == 422


def test_a_checkpoint_that_is_not_an_image_model_is_refused_by_name(
    client, tmp_path, monkeypatch
):
    """The machine guard fires FIRST for a local path — correctly — so it is
    patched out here to reach the layer beneath it. Both refusals are real and
    they are tested separately rather than one masking the other."""
    monkeypatch.setattr("modelmri.server._not_from_this_machine", lambda *a, **k: None)
    empty = tmp_path / "nothing"
    empty.mkdir()
    r = client.post("/api/image/load", json={"repo": str(empty)})
    assert r.status_code == 409
    assert "does not exist" in r.json()["error"]


def test_unloading_when_nothing_is_held_is_not_an_error(client):
    d = client.post("/api/image/unload").json()
    assert d["loaded"] is False


# ------------------------------------------------------- the schema's rules


def test_a_seed_outside_what_a_sampler_takes_is_rejected(client):
    for seed in (-1, 2**31):
        r = client.post("/api/image/attention", json={"prompt": "a cat", "seed": seed})
        assert r.status_code == 422, seed


def test_steps_are_bounded_in_both_directions(client):
    for steps in (0, 500):
        r = client.post(
            "/api/image/attention", json={"prompt": "a cat", "steps": steps}
        )
        assert r.status_code == 422, steps


def test_a_knockout_seed_is_required_rather_than_optional():
    """Every arm has to run at the identical seed or the difference between
    two images is the sampler rather than the word. Optional would let that
    through silently."""
    from modelmri.server import ImageKnockoutRequest

    assert ImageKnockoutRequest(prompt="a cat").seed == 0
    field = ImageKnockoutRequest.model_fields["seed"]
    assert field.annotation is int, "an optional seed makes the comparison noise"


def test_a_remote_caller_cannot_probe_for_directories_on_the_server(
    client, tmp_path, monkeypatch
):
    """The oracle CodeQL found, and it was in the guard rather than around it.

    The first version of this guard asked `Path(repo).is_dir()` to decide
    whether the guard applied — and asking that question about caller text
    ANSWERS it. Measured before the fix, from a client the server treats as
    remote, with a directory in the working directory:

        POST {"repo": "there_is_a_dir_here"}   -> 403   (guard fired)
        POST {"repo": "there_is_no_dir_here"}  -> 500   (fell through)

    One bit per request, unauthenticated, on any server started with
    `--host 0.0.0.0`. The two must now be indistinguishable.
    """
    import os

    monkeypatch.chdir(tmp_path)
    os.mkdir("there_is_a_dir_here")

    seen = []
    for name in ("there_is_a_dir_here", "there_is_no_dir_here"):
        r = client.post("/api/image/load", json={"repo": name})
        seen.append((r.status_code, "not a repository on the Hub" in r.text))
    assert seen[0] == seen[1], f"the two answers differ, which is the oracle: {seen}"


def test_the_hub_branch_is_what_a_remote_caller_gets(client, tmp_path, monkeypatch):
    """The shape gate alone is NOT sufficient, which is the half that is easy
    to miss. `is_hub_id("models")` is True — a bare name is a valid repo id —
    and `_snapshot` used an existing directory as-is, so a remote caller
    naming `models` had the server's own `./models` opened for them. Worse
    than the oracle it replaced: it does not reveal that the directory
    exists, it reads it."""
    import os

    monkeypatch.chdir(tmp_path)
    os.mkdir("models")
    (tmp_path / "models" / "config.json").write_text(
        '{"model_type": "vit"}', encoding="utf-8"
    )
    r = client.post("/api/image/load", json={"repo": "models"})
    # It must NOT have read that directory and found a ViT in it.
    assert "vision transformer" not in r.text
    assert "not a repository on the Hub" in r.text


def test_a_path_shaped_name_is_still_refused_by_the_machine_guard(client, tmp_path):
    """The ordinary case the guard exists for. Anything not hub-SHAPED is a
    path, decided without touching the disk."""
    r = client.post("/api/image/load", json={"repo": str(tmp_path)})
    assert r.status_code == 403
    assert "only possible from this machine" in r.json()["error"]


def test_a_hub_id_is_not_treated_as_a_path(client):
    """`owner/name` is a public name, and refusing it from a remote caller
    would be refusing the ordinary case."""
    from modelmri.behavdiff import is_hub_id

    assert is_hub_id("stabilityai/stable-diffusion-x4-upscaler") is True
    assert is_hub_id("runwayml/stable-diffusion-v1-5") is True


def test_the_shape_test_never_touches_the_filesystem(monkeypatch):
    """The property the whole fix rests on. If `is_hub_id` ever grows a
    filesystem check the oracle comes straight back, so this asserts it does
    not — by making any stat explode."""
    from pathlib import Path as _P

    from modelmri.behavdiff import is_hub_id

    def _explode(*_a, **_k):
        raise AssertionError("the shape test touched the filesystem")

    for name in ("is_dir", "exists", "stat", "expanduser", "resolve"):
        monkeypatch.setattr(_P, name, _explode, raising=False)

    for value in ("owner/name", "models", "/etc/passwd", "../x", "C:/m/x", "~/m", ""):
        is_hub_id(value)


def test_a_capacity_refusal_arrives_as_a_refusal_and_not_a_crash(
    client, tmp_path, monkeypatch
):
    """`TooBig` subclasses plain `ValueError`, NOT `BadRequest`, so the
    `(Refusal, BadRequest)` arm does not catch it and it fell through to
    `except Exception` and a 500.

    That turned the one refusal this route exists to deliver into "Something
    inside ModelMRI failed rather than refusing" — the capacity guard's whole
    job is to say "this will not fit, here is what to do" BEFORE a
    twenty-minute download, and a reader was shown a crash instead. Four other
    capacity-gated routes have carried the arm since they were written.

    Writing this test is also what caught the `NameError` in the arm itself:
    an `except` clause naming a module the handler had not imported, which
    fails only when the arm is actually reached.
    """
    from modelmri import capacity

    monkeypatch.setattr("modelmri.server._not_from_this_machine", lambda *a, **k: None)

    checkpoint = tmp_path / "vit"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "vit"}', encoding="utf-8")

    def too_big(*_a, **_k):
        raise capacity.TooBig(
            "that will not fit: 20.0 GB against 8.0 GB free", overridable=False
        )

    monkeypatch.setattr("modelmri.image_runtime._guard", too_big)

    r = client.post("/api/image/load", json={"repo": str(checkpoint)})
    assert r.status_code == 422, f"a capacity refusal came back as {r.status_code}"
    assert "will not fit" in r.json()["error"]
    assert "failed rather than refusing" not in r.text
