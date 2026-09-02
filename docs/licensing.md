# Licensing

ModelMRI is open core. The application — the server, the runtime, the CLI, the
app, and every instrument and analysis method in it — is free software under
**AGPL-3.0-only**. The pieces meant to live inside *your* software —
`modelmri-record`, `modelmri-policy`, the `npx modelmri` shim and the `.mri`
codec — are **Apache-2.0**, so they carry no copyleft into your product. These
docs are **CC-BY-4.0**. ModelMRI Cloud / Enterprise, the hosted and managed
product, is proprietary and is not in the repository.

The authoritative version of this page, with the path-by-path table, is
[`LICENSING.md`](https://github.com/muhammadmahadazher/ModelMRI/blob/main/LICENSING.md)
in the repository, and the license texts are in
[`LICENSES/`](https://github.com/muhammadmahadazher/ModelMRI/tree/main/LICENSES).
This page explains; it is not legal advice, and where it and a license text
disagree, the license text governs.

## What is under which license

| | License |
|---|---|
| The application: `modelmri/**` — server, runtime, CLI, every instrument and analysis method — and `frontend/**` — the app, the live demo, the `.mri` viewer — with `tests/**` and `scripts/**` | AGPL-3.0-only |
| `modelmri-record`, `modelmri-policy`, the `npx modelmri` shim, the `.mri` codec (`session.py`, `receipts.py`, `errors.py`, `fmt.py`, `paths.py`), and `examples/**` | Apache-2.0 |
| The documentation and its media; code samples in the docs may also be used under Apache-2.0 | CC-BY-4.0 |
| ModelMRI Cloud / Enterprise — managed compute, hosted execution, collaboration, enterprise administration | proprietary, not in the repository |

## The commitment

> ModelMRI does not charge for access to diagnostic or research methods. Every instrument and
> analysis method — including advanced and future ones such as SAE training, AutoSteer, autonomous
> investigation, causal and circuit analysis, model diffing, evals, agent tracing, vision/VLM,
> robotics/VLA and security scanning — ships in ModelMRI Community, and anyone with their own
> hardware can run it. What ModelMRI sells is operating, scaling, and organizing those methods:
> managed compute, hosted execution, collaboration, and enterprise administration. No capability
> that has shipped in Community will be moved behind a paywall.

## What this means for you

**I run ModelMRI on my machine, or on my cluster.** No obligations. Use it
for anything, commercial work included, at any scale. The AGPL's conditions
attach to distributing the software and to letting other people use a
*modified* version over a network; running the software is neither.

**I modified ModelMRI and let other people use my modified version over a
network.** Section 13 of the AGPL applies: offer those users the source of
your modified version, under the AGPL — or take a commercial license instead
(two cases down). Running an *unmodified* ModelMRI for other people adds no
such obligation; the source they could ask for is the repository.

**I import `modelmri-record`, `modelmri-policy` or the `.mri` codec into my
product.** Apache-2.0: use it, change it, ship it under your own license, no
copyleft. Writing `.mri` files from your own tool, or reading them, is the
same case — the format and its codec are meant to be used everywhere.

**I want to embed or redistribute the application under my own terms.** A
commercial license exists for that. Commercial licensing and support are
available — open a GitHub issue titled *commercial licensing* and the
maintainer will follow up privately.

## Versions before this change

ModelMRI was MIT-licensed from its first commit up to the commit tagged
`mit-final`. Everything distributed before that keeps its MIT terms for anyone
who has it: `modelmri` up to and including 0.12.0, `modelmri-record` up to and
including 0.1.4, the `modelmri` npm shim 0.0.1, and every commit up to the
tag. History is not rewritten. AGPL-3.0-only applies to `modelmri` from
0.13.0, and Apache-2.0 to `modelmri-record` from 0.2.0.

## Contributors

You keep your copyright. The contributor terms are in `CLA.md` in the
repository — a Contributor License Agreement that is a license, not an
assignment, and that promises your accepted work stays available under the
Community license. It is a draft until counsel has reviewed it and is not in
force; until it is, the rule in
[CONTRIBUTING.md](https://github.com/muhammadmahadazher/ModelMRI/blob/main/CONTRIBUTING.md)
applies: a contribution is licensed under the license of the files it changes.

## Why AGPL, and for how long

AGPL is a current strategy, not an ideology. It keeps every method public while
making a hosted, modified ModelMRI something its operator has to share back —
the one arrangement that would otherwise let someone sell this project's own
work against it. We will reconsider the Community license with evidence:
adoption, contributor experience, and what the license costs the people who
use it. Whatever the outbound license becomes, accepted contributions remain
open under the license they were contributed under; the CLA says so in
writing.
