# Third-party notices

What ModelMRI ships that it did not write, what it loads that it does not
ship, and the rule that keeps the two apart. The project's own licensing is in
[LICENSING.md](LICENSING.md); this page is about everyone else's.

## Dependencies

<!-- generated-section: dependencies
     This section is written by scripts/third_party_notices.py from uv.lock and
     frontend/package-lock.json, and a test refuses a direct dependency that has
     no entry here. Until that script lands, the two lockfiles are the record. -->

The Python and npm packages ModelMRI depends on are listed, with their
versions and licenses, by the lockfiles `uv.lock` and
`frontend/package-lock.json`. A generated table — name, version, license,
homepage, and whether the package is bundled into the wheel or the built
application — is added to this section by `scripts/third_party_notices.py`,
together with the result of the dependency license audit that produces it.

<!-- /generated-section -->

## Bundled assets

Shipped inside the built application (and therefore inside the `modelmri`
wheel, which carries the built application):

- **Geist** — `frontend/src/fonts/Geist-Variable.woff2`. Copyright 2024 The
  Geist Project Authors (<https://github.com/vercel/geist-font>). SIL Open
  Font License, Version 1.1; the license text is beside the font,
  `frontend/src/fonts/Geist-OFL.txt`.
- **JetBrains Mono** — `frontend/src/fonts/JetBrainsMono-Variable.woff2`.
  Copyright 2020 The JetBrains Mono Project Authors
  (<https://github.com/JetBrains/JetBrainsMono>). SIL Open Font License,
  Version 1.1; the license text is beside the font,
  `frontend/src/fonts/JetBrainsMono-OFL.txt`.

In the repository but in no build, recorded so that the state is written
down rather than discovered:

- **Switzer** — `frontend/src/fonts/Switzer-Variable.woff2` is committed but
  referenced by no stylesheet rule (only by comments), so it is not part of
  any build. It is an Indian Type Foundry face distributed through Fontshare
  under Fontshare's free-font licence, which is not the OFL, and its licence
  text is not in the repository. The file will be removed, or its licence
  restored beside it; until one of those has happened, this entry is the
  notice.
- **Archivo Black** — `@fontsource/archivo-black` (SIL OFL 1.1) is declared
  in `frontend/package.json` and imported by nothing, so it is in the
  dependency tree and in no build.

## Loaded at runtime, not distributed

Model weights, sparse autoencoders, datasets and robot policies are fetched
when you ask for them, from the publisher, under the publisher's license.
ModelMRI does not redistribute any of them, and where a publisher gates a
download behind accepting terms, ModelMRI links you to that page with your
own account rather than accepting anything for you. In particular:

- the GPT-2 sparse autoencoders from `jbloom/GPT2-Small-SAEs-Reformatted`
  (SAELens format), and the other SAE releases the registry in
  `modelmri/saes.py` knows how to load, each under its own publisher's terms;
- `lerobot/smolvla_base` and `lerobot/pusht` (HuggingFace LeRobot);
- any model you load from the HuggingFace Hub or from a local Ollama daemon.

## Test fixtures

None with an external origin are committed. `tests/` contains Python only;
a test that needs a model, an SAE or a dataset either builds a synthetic one
or fetches the real one from the Hub at run time, under that publisher's
terms.

## Provenance rule

Every method in ModelMRI is implemented independently from the paper that
describes it, and the paper is recorded — in the module's docstring and in
the changelog entry that introduced the method. Sources are for checking
claims, not for lifting designs. No code is copied from a repository that
does not ship a usable license: `modelmri/model_diff.py` declines to port two
crosscoder implementations for exactly that reason and says so in its
docstring, and a clean-room implementation from the papers is the only route
that would be taken. A contribution that brings third-party code with it must
say so, name the source and its license, and pass the same test — see
[CONTRIBUTING.md](CONTRIBUTING.md).
