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


# ------------------------------------------------- finding one to load at all
#
# The catalogue routes. `image_catalog` has its own unit tests; what these
# assert is the wiring and the status codes, because a refusal that arrives as
# a 500 reads as "ModelMRI is broken" rather than "the Hub is down", and those
# send a reader to two different places.


def test_the_task_list_is_not_empty_and_its_default_is_one_of_them(client):
    """A picker asks this instead of hardcoding a list, so an empty answer is a
    picker with nothing in it — and a default outside the list is a picker that
    opens on a selection it cannot make."""
    d = client.get("/api/image/tasks").json()
    assert d["tasks"], "the picker would render an empty dropdown"
    tags = [t["task"] for t in d["tasks"]]
    assert d["default"] in tags
    for t in d["tasks"]:
        assert t["label"] and t["means"], t["task"]


def test_a_task_nobody_can_open_is_refused_by_name_rather_than_searched(client):
    """422, not 500: the request is wrong, and the answer has to name the tags
    that would work — a caller who typed one that does not exist has no other
    way to learn which ones do. Nothing is downloaded to say so."""
    r = client.get("/api/image/search", params={"task": "nonsense"})
    assert r.status_code == 422
    said = r.json()["error"]
    for tag in client.get("/api/image/tasks").json()["tasks"]:
        assert tag["task"] in said, f"the refusal did not offer `{tag['task']}`"


def test_the_local_list_never_claims_bytes_for_an_unfinished_download(client):
    """Whatever is cached on the machine running this. A cache entry holding
    configs and no weights is an interrupted download, and reporting it at 0
    bytes would say it is ready and costs nothing — so an incomplete row must
    carry `null` and must not be counted into the total."""
    d = client.get("/api/image/local").json()
    assert d["means"]
    for m in d["models"]:
        if not m["complete"]:
            assert m["size_bytes"] is None, f"{m['path']} claimed bytes it has none of"
    assert d["bytes_on_disk"] == sum(
        m["size_bytes"] or 0 for m in d["models"] if m["complete"]
    )


def test_pricing_with_no_model_named_is_a_422_rather_than_a_guess(client):
    """The cost route exists to be asked before anything is spent, and a
    default repo would price something the caller did not ask about."""
    r = client.get("/api/image/size")
    assert r.status_code == 422
    assert "nothing to price" in r.json()["error"]


def test_an_unreachable_hub_answers_503_rather_than_500(client, monkeypatch):
    """503 says "the thing this depends on is down, try again"; 500 says
    "ModelMRI broke". The Hub being unreachable is neither ModelMRI's fault nor
    anything the reader can fix by reloading the page, and the sentence already
    tells them their downloaded models still work — which a 500 would bury
    under a generic failure."""
    from modelmri.errors import Refusal

    def unreachable(*_a, **_k):
        raise Refusal(
            "Could not reach the HuggingFace Hub. Check your connection — the "
            "full error is in the terminal running `modelmri serve`."
        )

    monkeypatch.setattr("modelmri.image_catalog.search", unreachable)
    r = client.get("/api/image/search")
    assert r.status_code == 503, f"an unreachable Hub came back as {r.status_code}"
    assert "Could not reach the HuggingFace Hub" in r.json()["error"]


# ------------------------------------------- covering the picture up


def test_the_attribution_cost_answers_with_no_model_at_all(client):
    """Asked FIRST, because the number it produces is what decides whether to
    run: the same image at stride 1 rather than 16 is not a slower run, it is
    a different afternoon."""
    d = client.get("/api/image/attribution/cost?height=224&width=224&patch=16").json()
    assert d["map_rows"] == 14
    assert d["map_cols"] == 14
    # N windows plus the unoccluded reference, which every window is measured
    # against. Forgetting it under-quotes by exactly the pass the map needs.
    assert d["passes"] == 196 + 1
    assert d["within_ceiling"] is True


def test_a_stride_that_would_take_an_afternoon_still_gets_priced(client):
    """`estimate` NEVER refuses on the ceiling. A caller who is about to be
    refused needs the number that got them refused, or they are guessing at
    the stride."""
    d = client.get(
        "/api/image/attribution/cost?height=224&width=224&patch=8&stride=1"
    ).json()
    assert d["passes"] > d["ceiling"]
    assert d["within_ceiling"] is False
    assert d["map_rows"] == 217


def test_no_seconds_are_forecast_for_a_machine_nobody_timed(client):
    """A duration from somebody else's hardware is the number people plan
    around, so there is none rather than an invented one."""
    d = client.get("/api/image/attribution/cost").json()
    assert d["seconds"] is None
    assert "no per-pass time was measured" in d["means"].lower()


def test_attribution_refuses_before_it_reads_the_image(client):
    """Nothing is loaded, so decoding a picture would be work done for an
    answer that cannot come. The refusal names the missing model rather than
    complaining about the image."""
    r = client.post("/api/image/attribution", json={"image": "not-a-data-url"})
    assert r.status_code == 409
    assert "No image model is loaded" in r.json()["error"]


def test_an_image_is_required_rather_than_defaulted(client):
    """This measures what a model looked at in ONE picture. There is no
    default image worth substituting."""
    assert client.post("/api/image/attribution", json={}).status_code == 422
    assert client.post("/api/image/attribution", json={"image": ""}).status_code == 422


def test_class_names_are_ordered_by_index_and_not_by_text(monkeypatch):
    """The ordering trap, on the real function.

    `id2label` survives a JSON round-trip with STRING keys, so sorting them as
    text puts "10" immediately after "1" and every name lands against the
    wrong class — while looking entirely reasonable. Sorting on the integer is
    what fixes it.

    Measured on the real head: `google/vit-base-patch16-224` reports class 610
    as "jersey, T-shirt, tee shirt", which is the correct ImageNet name for
    that index.
    """
    from modelmri.server import _label_names

    class _Model:
        class config:
            id2label = {"2": "third", "0": "first", "10": "eleventh", "1": "second"}

    names = _label_names(_Model())
    assert names[:3] == ["first", "second", "third"]
    assert names[-1] == "eleventh", "sorted as text, '10' would land at index 1"


def test_a_head_that_publishes_no_names_offers_none_rather_than_inventing_them():
    """`vision_attr` DROPS names that do not match the head's width rather
    than applying them to the wrong classes. A list of "class 0", "class 1"
    would be exactly the right length and would defeat that check."""
    from modelmri.server import _label_names

    class _Bare:
        class config:
            pass

    class _Empty:
        class config:
            id2label = {}

    assert _label_names(_Bare()) is None
    assert _label_names(_Empty()) is None
    assert _label_names(object()) is None


def test_a_label_table_that_is_not_keyed_by_index_is_unknown_not_guessed():
    """Keys that are not indices at all. Reported as unknown rather than
    ordered by whatever `sorted` does to mixed types."""
    from modelmri.server import _label_names

    class _Weird:
        class config:
            id2label = {"cat": "a cat", "dog": "a dog"}

    assert _label_names(_Weird()) is None


def test_the_folder_walk_reports_the_budget_its_truncation_refers_to(client):
    """`truncated` alone is a caveat nobody can size.

    "The walk stopped at its budget" reads identically whether 12 of a
    possible 20 were reached or 120 of a possible 120, and those are not the
    same situation. The number travels with the flag so the panel can state
    it rather than describe it.

    It is the DIRECTORY walk's own limit. `SCAN_CACHE_LIMIT` is a different
    number for a different walk, and quoting one for the other would print a
    wrong figure with full confidence.
    """
    from modelmri import imaging

    body = client.get("/api/image/discovered").json()
    assert body["scan_limit"] == imaging.SCAN_DIRS_LIMIT
    assert isinstance(body["truncated"], bool)
    # Whatever this machine happens to hold, the list cannot exceed the budget
    # that produced it.
    assert len(body["models"]) <= body["scan_limit"]


def test_the_cache_walk_keeps_its_own_budget_and_they_are_not_shared(client):
    """The two walks answer different questions and are capped separately."""
    from modelmri import imaging

    assert client.get("/api/image/available").json()["scan_limit"] == (
        imaging.SCAN_CACHE_LIMIT
    )
    assert imaging.SCAN_CACHE_LIMIT != imaging.SCAN_DIRS_LIMIT


def test_the_cache_listing_reports_the_cap_the_sibling_route_reports(client):
    """One walk, two routes, and only the unread one disclosed its cap.

    `/api/image/available` and `/api/image/local` both read
    `imaging.scan_cache`, which stops at `SCAN_CACHE_LIMIT`. `available` has
    said so for months; `local` did not — and `local` is the one the picker
    renders, because `getImageAvailable` has no consumers. A list that
    silently stops at 200 reads as "everything on this disk".
    """
    from modelmri import imaging

    body = client.get("/api/image/local").json()
    assert body["scan_limit"] == imaging.SCAN_CACHE_LIMIT
    assert isinstance(body["truncated"], bool)
    assert len(body["models"]) <= body["scan_limit"]
    # The two routes agree about the limit they share.
    assert body["scan_limit"] == client.get("/api/image/available").json()["scan_limit"]


def test_a_small_cache_is_not_reported_as_zero_gigabytes(client, monkeypatch):
    """`{held / 1e9:,.1f} GB` called a real 4 MB cache "0.0 GB in total" in the
    same sentence that says the weights are present."""
    from modelmri import image_catalog

    def one_tiny_model():
        return [
            {
                "path": "hf-internal-testing/tiny-stable-diffusion-torch",
                "family": "stable-diffusion",
                "label": "tiny",
                "known": True,
                "architecture": "StableDiffusionPipeline",
                "capabilities": [],
                "reason": "",
                "size_bytes": 4_000_000,
                "complete": True,
            }
        ]

    monkeypatch.setattr(image_catalog, "local", one_tiny_model)
    body = client.get("/api/image/local").json()
    assert body["bytes_on_disk"] == 4_000_000
    assert "4 MB in total" in body["means"], body["means"]
    assert "0.0 GB" not in body["means"]


def test_the_readout_refuses_a_field_it_never_read(client):
    """`/api/image/cv/readout` took `CVPredictRequest`, which declares `top_k`
    and `mask_threshold` — and `layer_readout` reads neither.

    It returns per-layer maps, not a class ranking, so there is no list to cut
    and no mask to threshold. Both were accepted and silently discarded, and
    the app's own client was sending `top_k`, so a reader tuning it watched
    nothing change with no way to learn why.
    """
    r = client.post(
        "/api/image/cv/readout",
        json={"image": "data:image/png;base64,AAA", "top_k": 5},
    )
    assert r.status_code == 422
    assert "top_k" in r.text

    # And the shape it does take gets past validation — the refusal below is
    # about no model being loaded, which is a different and correct one.
    ok = client.post(
        "/api/image/cv/readout", json={"image": "data:image/png;base64,AAA"}
    )
    assert ok.status_code != 422, ok.text


def test_the_knockout_marking_bound_is_published_not_only_enforced(client):
    """A panel that lets somebody pick twenty-five words and then hands them a
    validation error has charged them the picking before mentioning the limit.

    It bounds the MARKING, not the work: `image_attention.knockout` derives its
    arms from `prompt.split()` and runs one for every word, so this list only
    says which rows the caller asked about.
    """
    from modelmri import image_attention

    status = client.get("/api/image").json()
    assert status["max_knockout_words"] == image_attention.MAX_KNOCKOUT_WORDS

    # The route enforces the same number it publishes.
    over = client.post(
        "/api/image/knockout",
        json={
            "prompt": "a b c",
            "seed": 1,
            "words": [f"w{i}" for i in range(image_attention.MAX_KNOCKOUT_WORDS + 1)],
        },
    )
    assert over.status_code == 422
    assert "words" in over.text
