# modelmri-policy

Holds a robot policy — vision tower **and** action expert — in its own
process, so [ModelMRI](https://github.com/muhammadmahadazher/ModelMRI) can ask
it what it would *do* rather than only where it looked.

It is a separate package, and a separate virtual environment, because lerobot
pins torch and numpy hard enough that installing it beside ModelMRI breaks
both. That process boundary is the permanent answer to a conflict that left
the one shipped competitor stale on a modified fork.

```
modelmri policy install     # builds the venv, installs the pinned lerobot
modelmri policy start       # brings the sidecar up on loopback
```

Installing this package alone pulls nothing: the heavy half is the `policy`
extra, and the installer puts it in its own environment.

## The contract is versioned, and the version is checked

Every exchange carries a contract number, and both halves declare it
independently — see `contract.py`. A mismatch is refused rather than
best-effort, because an action chunk is a claim about what a robot would do,
and one served across a version boundary is a different policy's answer
wearing this one's name.

MIT.
