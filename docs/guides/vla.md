# Robot policies

The same question — *what is it looking at?* — asked of a vision-language-action
policy instead of a language model.

## What it shows

ModelMRI reads a [LeRobot](https://github.com/huggingface/lerobot) dataset
frame by frame and runs the vision tower of `lerobot/smolvla_base` over a
chosen frame, then paints the policy's attention over the camera image.

- Scrub the timeline to any frame in any episode.
- Press **Run policy on this frame** to get the attention grid.
- Move through layers to watch attention concentrate.

Each frame also carries its real robot state and action vectors, so you can see
what the arm was actually doing at that instant.

## Setup

```bash
pip install "modelmri[vla-lite]"
```

That adds `av`, `pyarrow` and `pillow` — enough to read LeRobot v3.0 datasets
and decode their video. The default dataset is `lerobot/pusht`: 206 episodes of
a robot pushing a T-shaped block onto a target.

Both the dataset and the policy must already be in your HuggingFace cache;
ModelMRI will tell you which is missing rather than silently downloading
gigabytes.

!!! note "No LeRobot dependency"
    ModelMRI reads the dataset format directly with `pyarrow` and `pyav`
    instead of importing `lerobot`, whose torch and numpy pins conflict with a
    current install. The format is stable and the reader is about 200 lines.

    One detail that cost real time: PushT's cache ref is `v3.0`, not `main`.
    Anything that assumes `main` finds nothing and reports an empty dataset.

## Reading the heat

The vision tower produces a 32×32 patch grid. Attention is reduced per head and
painted over the upscaled frame — cool where the policy isn't looking, hot
where it is. Low attention stays nearly transparent so the image reads through
rather than being buried under a wash of colour.

As with language attention, the useful signal is how it **changes with depth**:
early layers spread across the frame, later ones settle on the block and the
target.

## Honest limits

- **This is the vision tower, not the whole policy.** ModelMRI shows what the
  encoder attends to, not how the action head turns that into motion. The
  action vectors are displayed alongside, but the mapping between them is not
  visualised.
- **It is not running a robot.** These are recorded episodes. Nothing here
  actuates anything, and no hardware is required — which is the point, since
  the alternative is owning an arm.
- **One policy so far.** SmolVLA's vision tower is what has been tested end to
  end: 12 layers × 12 heads, 197 tensors loaded from the checkpoint, and a
  refusal to start if any are missing rather than running with a partly
  initialised tower.
