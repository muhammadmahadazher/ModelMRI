# Licensing

ModelMRI is open core, and this page is the map. The application — the
server, the runtime, the CLI, the app, and every instrument and analysis method
in it — is free software under the GNU Affero General Public License, version 3
only (**AGPL-3.0-only**). The pieces meant to live inside *your* software — the
SDK packages and the `.mri` codec — are under the Apache License 2.0
(**Apache-2.0**), so they carry no copyleft into your product. The
documentation is **CC-BY-4.0**. ModelMRI Cloud / Enterprise, the hosted and
managed product, is proprietary and is not in this repository.

This page explains; it is not legal advice, and where it and a license text
disagree, the license text governs. The full texts are in
[`LICENSES/`](LICENSES/): [AGPL-3.0-only](LICENSES/AGPL-3.0-only.txt),
[Apache-2.0](LICENSES/Apache-2.0.txt), [CC-BY-4.0](LICENSES/CC-BY-4.0.txt).

## Which license covers which files

| Path | License | What it is |
|---|---|---|
| `modelmri/**`, except the five codec files below | AGPL-3.0-only | the server, runtime and CLI; every instrument and analysis method; the built app the server serves |
| `frontend/**` | AGPL-3.0-only | the app, the live demo, and the `.mri` viewer application |
| `tests/**`, `scripts/**`, `.github/**`, and the configuration and lockfiles at the root | AGPL-3.0-only | what builds, tests and ships the application |
| `modelmri/session.py`, `modelmri/receipts.py`, `modelmri/errors.py`, `modelmri/fmt.py`, `modelmri/paths.py` | Apache-2.0 | the `.mri` codec — reading, writing, validating and hashing the format. They are Apache-2.0 today; the handful of places where they still reach into the application are listed in tests/test_licensing.py as debts that can only shrink, and they move into their own package, modelmri-mri, with the codec extraction. They live inside the application today and will move into their own package, `modelmri-mri` |
| `packages/modelmri-record/**` | Apache-2.0 | `modelmri-record`, the dependency-free recorder you import into your agent |
| `packages/modelmri-policy/**` | Apache-2.0 | `modelmri-policy`, the robot-policy sidecar |
| `npm-stub/**` | Apache-2.0 | the `npx modelmri` launcher shim; it contains no application code |
| `examples/**` | Apache-2.0 | code samples written to be copied into your own code |
| `docs/**`, including `docs/media/**` | CC-BY-4.0 | the documentation and its images; code samples in the docs may also be used under Apache-2.0 |
| `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and the project's other documents at the root | CC-BY-4.0 | prose about the project. Texts adopted from elsewhere — the Contributor Covenant, the Harmony CLA — say so at their top and keep their own terms |
| `spec/mri/**`, once it exists | Apache-2.0 for schemas, codecs and validators; CC-BY-4.0 for the specification text | the `.mri` format specification |
| ModelMRI Cloud / Enterprise | proprietary | managed compute, hosted execution, collaboration, organisations, RBAC, SSO/SAML/SCIM, audit, registries, fleet, scheduling, policy and compliance, support, VPC and on-prem — none of it is in this repository |

A file's own `SPDX-License-Identifier` header is the authoritative statement
for that file; the files that cannot carry a header are covered by
`REUSE.toml`; this table is the map both of them follow.

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
such obligation; the source they could ask for is this repository.

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

You keep your copyright. The contributor terms are in `CLA.md` — a
Contributor License Agreement that is a license, not an assignment: it lets
ModelMRI include your work in Community and in commercially licensed
distributions, and it promises that your accepted work stays available under
the Community license. It is a draft until counsel has reviewed it, and it is
not in force. Until it is, the rule in [CONTRIBUTING.md](CONTRIBUTING.md)
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

## Checking

Every first-party source file carries an `SPDX-License-Identifier` header,
the files that cannot carry one are covered by `REUSE.toml`, and CI runs
`reuse lint` on every pull request beside `tests/test_licensing.py`, which
checks by path and by import that the Apache-2.0 packages and the `.mri`
codec import nothing new from the AGPL application — the crossings that
predate the check are listed in that file with where each one goes, and the
list only shrinks. To check a checkout yourself:

```bash
uvx --from "reuse[charset-normalizer]" reuse lint
uv run pytest tests/test_licensing.py
```

A red result is a bug.
