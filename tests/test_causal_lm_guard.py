"""The guard that decides whether the playground may load a repo at all.

Written after `_require_causal_lm` refused every multimodal Gemma while the
line below it -- `AutoModelForCausalLM.from_pretrained` -- built them without
complaint. Two code paths were answering one question differently, and the
stricter one was the one that had no idea what the loader does.

Nothing here reaches the network. `AutoConfig.from_pretrained` is replaced with
a function returning a config object built in-process, but the config CLASSES
and the causal-LM mapping are transformers' own, so these assert against the
real registry rather than a mock of it. Which config classes exist depends on
the installed transformers, so the multimodal case is discovered at run time
instead of naming a version-specific model.
"""

from __future__ import annotations

import pytest
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

from modelmri.errors import BadRequest
from modelmri.runtime import _require_causal_lm


@pytest.fixture
def stub_config(monkeypatch):
    """Make `AutoConfig.from_pretrained` answer with a config we chose.

    Patched on the `transformers` module rather than on `modelmri.runtime`,
    because the guard imports the name inside its own body -- patching a
    module attribute the function never reads would test nothing and pass.
    """
    import transformers

    def install(config_cls, architectures, **kwargs):
        cfg = config_cls(**kwargs)
        cfg.architectures = architectures
        monkeypatch.setattr(
            transformers.AutoConfig, "from_pretrained", lambda *a, **k: cfg
        )
        return cfg

    return install


def _conditional_generation_causal_lm():
    """A (config class, mapped class) pair the OLD suffix test gets wrong.

    Found by asking the mapping rather than hardcoding `gemma4`: the point is
    that some causal LM is registered under a class name that does not end in
    ForCausalLM or LMHeadModel, not that a particular model exists in whatever
    transformers is installed.
    """
    for config_cls, model_cls in MODEL_FOR_CAUSAL_LM_MAPPING.items():
        if not model_cls.__name__.endswith(("ForCausalLM", "LMHeadModel")):
            return config_cls, model_cls
    return None, None


# ----------------------------------------------------- what must be allowed


def test_a_plain_causal_lm_passes(stub_config):
    from transformers import GPT2Config

    stub_config(GPT2Config, ["GPT2LMHeadModel"])
    _require_causal_lm("owner/whatever")  # no raise


def test_a_repo_publishing_no_architectures_is_not_blocked_on_a_guess(stub_config):
    """Unknown shape is not a refusal. google/gemma-4-E4B-it-qat-mobile-
    transformers publishes no `architectures` key, and this arm is the only
    reason it ever loaded -- which is exactly why the arm below had to exist
    too, rather than that accident being mistaken for the rule working."""
    from transformers import GPT2Config

    stub_config(GPT2Config, [])
    _require_causal_lm("owner/whatever")


def test_the_class_automodelforcausallm_would_build_is_allowed(stub_config):
    """The regression. A checkpoint whose declared architecture IS the class
    the loader constructs must not be refused for how that name is spelled."""
    config_cls, model_cls = _conditional_generation_causal_lm()
    if config_cls is None:
        pytest.skip(
            "this transformers registers no causal LM under a name that fails "
            "the suffix test, so there is nothing here to get wrong"
        )
    # The old rule, restated, so the test fails loudly if the premise dies.
    assert not model_cls.__name__.endswith(("ForCausalLM", "LMHeadModel"))

    stub_config(config_cls, [model_cls.__name__])
    _require_causal_lm("owner/whatever")


def test_the_gemma_that_motivated_this_is_allowed(stub_config):
    """The case from the field, pinned by name rather than discovered.

    The test above asserts the general property and passes on a transformers
    that happens to register some OTHER conditional-generation causal LM first
    -- on the box this was written on it found BertGenerationDecoder, which is
    correct and is not the model anybody was trying to load. Gemma is what
    sent the maintainer here, so Gemma gets its own assertion when the
    installed transformers knows about it.
    """
    try:
        from transformers import Gemma4Config as config_cls
    except ImportError:
        try:
            from transformers import Gemma3Config as config_cls
        except ImportError:
            pytest.skip("this transformers has neither Gemma 4 nor Gemma 3")

    mapped = MODEL_FOR_CAUSAL_LM_MAPPING.get(config_cls, None)
    assert mapped is not None, (
        f"{config_cls.__name__} is no longer in the causal-LM mapping, so the "
        "guard is not the thing keeping this model out any more"
    )
    stub_config(config_cls, [mapped.__name__])
    _require_causal_lm("google/gemma-4-E4B-it")


# ----------------------------------------------------- what must stay refused


def test_an_image_classifier_is_still_refused(stub_config):
    from transformers import ViTConfig

    stub_config(ViTConfig, ["ViTForImageClassification"])
    with pytest.raises(BadRequest) as err:
        _require_causal_lm("google/vit-base-patch16-224")
    assert "ViTForImageClassification" in str(err.value)


def test_a_masked_encoder_is_still_refused(stub_config):
    """The loose test this was NOT written as. bert-base-uncased declares
    `BertForMaskedLM`; the causal mapping holds `BertLMHeadModel` for the same
    config, so a bare "is this model_type in the mapping" check would wave it
    through and `AutoModelForCausalLM` would build an encoder with an untrained
    LM head that generates fluent nonsense. The declared class has to match."""
    from transformers import BertConfig

    stub_config(BertConfig, ["BertForMaskedLM"])
    with pytest.raises(BadRequest) as err:
        _require_causal_lm("bert-base-uncased")
    assert "BertForMaskedLM" in str(err.value)


def test_a_config_that_cannot_be_read_defers_to_the_real_loader(monkeypatch):
    """`AutoConfig` raises on diffusion and policy repos. The guard's job is to
    replace an unreadable traceback, not to add one, so it stands aside and
    lets the loader produce the error it knows how to explain."""
    import transformers

    def boom(*_a, **_k):
        raise ValueError("unrecognized model type")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", boom)
    _require_causal_lm("stabilityai/stable-diffusion-x4-upscaler")


def test_a_mapping_lookup_that_raises_does_not_escape(stub_config, monkeypatch):
    """`_LazyAutoMapping.get` imports a modeling module, so it can fail on a
    transformers built without that model. The refusal below still has to be a
    refusal, not an ImportError from inside the guard."""
    import transformers
    from transformers import ViTConfig

    stub_config(ViTConfig, ["ViTForImageClassification"])

    class Exploding:
        def get(self, *_a, **_k):
            raise ImportError("no modeling module here")

    monkeypatch.setattr(transformers, "MODEL_FOR_CAUSAL_LM_MAPPING", Exploding())
    with pytest.raises(BadRequest):
        _require_causal_lm("google/vit-base-patch16-224")
