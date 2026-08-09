# Security policy

ModelMRI runs on your machine, loads model weights, and records agent traces
that may contain credentials. Each of those is a real surface, and this
document says plainly what is defended and what is not.

## Supported versions

Security fixes land on the latest `main` and are released to PyPI. Until 1.0,
no older release is guaranteed a backport.

| package | current |
|---|---|
| `modelmri` | 0.6.x |
| `modelmri-record` | 0.1.x |

## Report a vulnerability

Do not open a public issue. Use GitHub's private reporting flow:

<https://github.com/muhammadmahadazher/ModelMRI/security/advisories/new>

Include the affected version or commit, the impact, reproduction steps, and a
minimal proof of concept. Please don't test against anyone else's machine or
account.

## The trust model, stated plainly

**ModelMRI is a local single-user tool.** The server binds `127.0.0.1` by
default and has no authentication, because it assumes one trusted user on one
machine. It is *not* hardened for multi-tenant or public exposure.

If you bind it to another interface, you have given everyone who can reach that
port the ability to load models, execute the code described below, and read
your recorded traces. Don't do that without putting your own authentication in
front of it.

## Loading a model executes code

This is inherent to the ecosystem, not specific to ModelMRI, and it is worth
saying out loud:

- **`.bin` / pickle checkpoints execute arbitrary code on load.** Prefer
  `safetensors`, which cannot.
- **`trust_remote_code` is never enabled by ModelMRI.** Models that require it
  will fail to load rather than silently running code from the Hub.
- **Custom model adapters are Python files that ModelMRI imports and runs**, by
  design — that's how you point it at a network you trained yourself. It will
  only import a file you explicitly named, it refuses paths outside the
  directories you configured, and it never fetches an adapter from the network.
  Treat an adapter you didn't write exactly like any other script someone sent
  you.

## Credentials

**Your HuggingFace token** is stored in ModelMRI's config directory, which
follows platform convention rather than a fixed path — `%APPDATA%\ModelMRI` on
Windows, `~/Library/Application Support/ModelMRI` on macOS,
`$XDG_CONFIG_HOME/modelmri` on Linux. Run `modelmri where` to print the
resolved location on your machine; that command reads the same code the token
writer does, so it cannot drift from this document.

The file is created owner-only — opened at mode `0600` rather than narrowed to
it after the fact, so there is no window in which it exists world-readable —
and moved into place atomically, so an interrupted write cannot leave a
half-written credential. **That mode is enforced on POSIX only.** On Windows,
`chmod` sets the read-only attribute and grants nothing, so the file inherits
your user profile's ACL; that is the same protection your other profile data
has, and it is not equivalent to `0600` on a machine where other accounts have
administrative access.

The token is never written into the repository or a trace, and is sent to
nowhere except `huggingface.co`. ModelMRI never asks for your password and has
no account of its own. Revoke a token at
<https://huggingface.co/settings/tokens>; deleting the file is enough to sign
out locally.

**Traces are redacted before they leave your process.** `modelmri-record`
scrubs the common credential shapes — API keys, bearer tokens, private keys,
connection strings — including from payloads the recorder itself truncated.
That last part was a real bug: the recorder truncates a long value at capture
time, which severed a PEM key's `-----END` line and defeated a pattern that
required it. Redaction now matches the headless remainder too.

Redaction is a safety net, not a guarantee. It matches shapes it knows. If your
secrets have a house format, add it:

```python
from modelmri_record.redact import make_redactor
red = make_redactor([r"ACME-[0-9]{6}"])
```

A credential shape that slips through the defaults is a legitimate security
report.

## What ModelMRI sends over the network

Only what you ask for:

- Model and dataset downloads from HuggingFace, when you load one.
- Requests to a local Ollama daemon, when you select an Ollama model.
- Nothing else. No telemetry, no analytics, no crash reporting, no phone-home.
  The static demo on GitHub Pages runs entirely in your browser against baked
  fixtures and talks to no server at all.

## Known limitations

- **No authentication or authorization.** See the trust model above.
- **No sandboxing of model code.** A model runs with your user's privileges.
- **Traces are stored unencrypted** as JSON, on your disk, under your
  permissions.
- **The recorder never raises.** If delivery, redaction, or serialization
  fails, it degrades quietly — a tracing library that can take down the host
  application is one nobody leaves switched on. The trade-off is that a silent
  failure is possible; when the endpoint is unreachable a trace is written to
  ModelMRI's data directory (`modelmri where` prints it, `MODELMRI_TRACE_DIR`
  overrides it) so the data isn't simply lost. Deliberately *not* the working
  directory: the recorder is imported by your agent, so that would drop full
  prompts and tool output into whatever repo you launched from.
