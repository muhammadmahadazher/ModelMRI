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
    assert "No image model is loaded" in said.pop()


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


def test_a_checkpoint_that_is_not_an_image_model_is_refused_by_name(client, tmp_path):
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
