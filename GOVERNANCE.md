# Governance

This document says who decides what in ModelMRI, today and as the project
grows. It describes the project as it is. There is no foundation, board,
steering committee or council, and none is claimed here.

## Today

ModelMRI is built and maintained by its founder, **Muhammad Mahad Azher**, who
is the **Project Lead**. The Project Lead:

- accepts or declines contributions;
- appoints and removes Maintainers and Reviewers;
- approves releases, and signs and publishes them;
- decides the high-governance changes listed below;
- is the final word when a decision is contested.

Every other role below is defined so that it means something the day it is
filled. Today none of them is.

## Roles

**Contributor.** Anyone whose pull request, issue, review comment, benchmark,
model adapter or documentation has been accepted. Contributors keep their
copyright (see [CLA.md](CLA.md), a draft) and are credited in the history of
the repository, which is the record.

**Reviewer.** A Contributor the Project Lead has asked to review pull requests
in an area they know. A Reviewer's approval is a signal, not a merge: it says
"I read this, I ran it, and it is right", and the pull request still needs a
Maintainer to merge it. Reviewers are named in `.github/CODEOWNERS` for the
paths they cover.

**Maintainer.** A Reviewer with merge rights to `main`, appointed by the
Project Lead after a sustained record of reviewed, accepted work. A Maintainer
may merge pull requests that pass the required checks, triage issues, label
and close them, cut pre-release builds for testing, and moderate community
spaces under the [Code of Conduct](CODE_OF_CONDUCT.md). A Maintainer may not
merge a high-governance change without the Project Lead's approval, may not
publish a release, and may not change repository settings.

**Core Maintainer.** A Maintainer who has held the role long enough, across
enough of the codebase, that the Project Lead trusts their judgment on
questions of measurement semantics — what a number on screen means and when
the tool should refuse to show one. Core Maintainers may approve
high-governance changes in the Project Lead's absence, and their recorded
approval is what a release's changelog cites for such changes.

**Project Lead.** Holds the roles above and the responsibilities listed under
*Today*. Names a successor if stepping down.

Roles are gained by appointment, on the basis of work that is already in the
repository — not by asking, and not by volume. They are lost by stepping down,
by a year of inactivity (announced first, never silently), or by the Project
Lead's decision after a Code of Conduct process.

## High-governance changes

Some changes are not ordinary pull requests, whoever opens them. Each needs
the Project Lead's approval, **recorded in the pull request** (a review, or a
comment that says so), before it merges:

- **the `.mri` format** — any change to what a `.mri` file contains, how a
  section is validated, or how a receipt is hashed;
- **measurement semantics** — a change to what any number the tool shows
  means, how it is measured, or when the tool refuses to measure it;
- **security boundaries** — path roots, the custom-adapter loader, anything
  that decides what code runs or what a request may read;
- **licensing** — `LICENSE*`, `LICENSING.md`, `LICENSES/`, `REUSE.toml`,
  `CLA.md`, and the license metadata of any package;
- **release signing** — how releases are built, signed and published;
- **dependency additions to the Apache-2.0 packages** — `modelmri-record`
  and `modelmri-policy` ship into other people's software and stay
  dependency-free; adding one is a decision, not a convenience.

`.github/CODEOWNERS` names the paths this covers, so that the rule is visible
where the change happens.

## How decisions are made

In the open, in the repository. A change is proposed as an issue or a pull
request and decided there; the reasoning stays attached to the decision. A
release is a tag on `main` and a changelog entry. Larger questions —
direction, licensing, what the project will not build — are announced in
GitHub Discussions and are open to comment before they are settled.

Disagreement about a method is welcome and is settled by measurement: a
reproducible result beats an opinion, including the Project Lead's. The
[Code of Conduct](CODE_OF_CONDUCT.md) is about how that argument is conducted,
not whether it may be.

## What this document does not do

It does not create an entity, a foundation or a legal relationship of any
kind. It does not promise a timeline for filling any role. It changes by pull
request, like everything else here, and a change to it is itself a
high-governance change.
