# smalldet — configuration-driven small-object detection

A PyTorch/torchvision object detection project built around one question:
**how do you actually move `AP_small` and `AP_medium` at IoU=[0.50:0.95]?**

Everything the project does — finetuning, prediction, visualization, and the
Gradio UI — is described by one YAML document validated into a typed config
tree. Python code never reads YAML directly and never hard-codes a
hyper-parameter; it asks the config object.

```
python -m smalldet.cli synth    --root Dataset/synthetic --num-images 48
python -m smalldet.cli stats    configs/finetune_synthetic.yaml --plot outputs/areas.png
python -m smalldet.cli train    configs/finetune_synthetic.yaml
python -m smalldet.cli evaluate configs/finetune_synthetic.yaml --checkpoint outputs/synthetic/checkpoints/best.pt
python -m smalldet.cli predict  configs/predict.yaml --images Dataset/synthetic/images
python -m smalldet.cli app      configs/app.yaml
```

## Why small objects need their own project

`AP_small` is not a harder version of `AP`. It fails for specific, fixable
reasons, and most of them are settings rather than model capacity. In the order
they pay off:

| # | Lever | Why it matters |
|---|---|---|
| 1 | **Anchor base sizes** | torchvision's default RPN pyramid starts at 32px. An anchor is a positive training sample only when its IoU with a ground-truth box clears ~0.7, and a 12px object **cannot reach that against a 32px anchor at any position**. It is never sampled, produces no proposal, and is invisible to the second stage. No amount of training fixes it. |
| 2 | **Input resolution** (`min_size`) | The detector resizes every input internally. At `min_size=800`, a 4000px frame shrinks 5× and a 20px object becomes 4px. |
| 3 | **`box_detections_per_img`** | Defaults to 100. A tray of 300 parts silently loses 200 of them, which reads as a recall ceiling that looks like a model problem. |
| 4 | **Augmentation** | `RandomZoomOut` shrinks every object in the frame — actively harmful here, so it is registered but flagged. `ScaleJitter` is biased upward (`[1.0, 1.8]`), never below 1.0. `SanitizeBoundingBoxes(min_size=…)` deletes small boxes; the default stays at 1.0. |
| 5 | **Area buckets** | COCO's 32²/96² cut-offs are calibrated to COCO's ~640px images. On a 4000px frame *every* object lands in "small": `AP_small` becomes a copy of `AP` and `AP_medium` returns the `-1` sentinel because its bucket is empty. |

That last point is why `smalldet.cli stats` exists — run it on any new dataset
**before** trusting a metric:

```
python -m smalldet.cli stats configs/finetune_toy_voc.yaml --plot outputs/areas.png
```

Set `eval.auto_area_ranges: true` to cut the buckets at percentiles of the
dataset's own size distribution, which keeps all three populated by construction.

> **`-1` means "no ground truth in this bucket", not a score of −1.** Checkpoint
> monitors never let it win, plots never draw it, averages never include it.

## Install

```
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest, playwright
playwright install chromium              # for the browser tests only
```

Only `torch`, `torchvision`, `numpy`, `pillow`, `pyyaml` and `matplotlib` are
required. `pycocotools`, `torchmetrics`, `tensorboard` and `psutil` are optional
and all have working fallbacks.

## Layout

```
smalldet/
  config/       typed schema, strict binding, `_base_` composition, validation
  data/         COCO dataset, transforms v2 pipelines, collate, size stats, splitting
  models/       detector factory per family, small-object anchor pyramids
  engine/       finetuning strategies, optimizers/schedules, callbacks, Trainer
  evaluation/   COCOeval reimplemented in NumPy, with AP_small/AP_medium first-class
  inference/    postprocessing, tiled inference, the Predictor facade
  visualization/ box & mask rendering, PR curves, size histograms
  app/          DetectionService + FinetuneService (no Gradio) and the UI wiring
  pipeline.py   config -> running system, in one place
  cli.py        train / evaluate / predict / app / stats / synth
configs/        base.yaml plus four configs that inherit from it
tests/          unit suite, plus tests/e2e for the Gradio app
```

### Design patterns that earned their place

- **Registry** for transforms, detector families, optimizers, schedulers,
  strategies, and callbacks. A YAML string becomes a key lookup, so a typo
  fails immediately with the list of valid names rather than an `AttributeError`
  three layers down.
- **Strict config binding** — unknown keys are errors. A silently-ignored
  `bacth_size` typo means a run that quietly uses the default and wastes a
  GPU-day. Errors name their path: `model.anchors.base_sizes[2]`.
- **`_base_` composition** — `app.yaml` → `predict.yaml` →
  `finetune_synthetic.yaml` → `base.yaml`. The predict and app configs inherit
  the model definition from the training config they must stay consistent with,
  instead of duplicating it and drifting away from the checkpoint.
- **Strategy** for finetuning, so the trainer never branches on it.
- **Observer/callbacks** for logging, checkpointing, and early stopping, so the
  loop reads as the algorithm it is.
- **Service/UI split** — `app/service.py` and `app/finetune_service.py` import
  no Gradio, so the whole UI is unit-testable without a browser.

## Configuration

One document drives everything. See [`configs/base.yaml`](configs/base.yaml) for
the annotated defaults.

```yaml
data:       splits, augmentation pipelines, dataloaders
model:      architecture, weights, anchors, min_size/max_size
finetune:   strategy, trainable layers, backbone LR multiplier
optimizer:  name, lr, weight decay
scheduler:  schedule + per-iteration warmup
train:      loop mechanics, checkpointing, early stopping
eval:       IoU thresholds, max_dets, area_ranges, auto_area_ranges
predict:    score threshold, extra NMS, tiled inference
visualize:  colours, labels, mask alpha, small-object highlight
app:        server, examples, which controls the user may move
```

Override anything without editing a file:

```
python -m smalldet.cli train configs/finetune_synthetic.yaml \
    --set train.epochs=3 --set model.anchors.base_sizes=[4,8,16,32,64]
```

`train.checkpoint.monitor` defaults to **`AP_small`**, not `AP` — a checkpoint
that wins on overall AP by improving large objects is the wrong checkpoint here,
and defaults are what people actually run.

### Finetuning strategies

| Strategy | What trains | When |
|---|---|---|
| `head_only` | Only the new prediction head | Tens to low hundreds of images |
| `partial` | Last *N* backbone stages + FPN + head | The reliable default |
| `gradual` | Unfreezes deeper stages on a schedule | When "enough data" is unclear |
| `full` | Everything, backbone at a lower LR | Highest ceiling, most data |

The FPN stays trainable in every mode except `head_only`: it is where the
multi-scale features small objects depend on are actually formed.

## Evaluation

`smalldet/evaluation/coco_eval.py` is a faithful NumPy reimplementation of
`pycocotools.cocoeval.COCOeval` for boxes — same greedy score-ordered matching,
same crowd handling (IoA rather than IoU against crowd regions), same 101-point
interpolated precision, same `-1` sentinel.

It exists because `pycocotools` is a compiled dependency that is awkward on
Windows and evaluation is the one thing this project cannot do without — and
because owning it makes the area cut-offs a config value rather than a constant
buried in a C extension. When `pycocotools` *is* installed, the test suite
asserts the two agree to 1e-6, so it is checked rather than assumed.

## Tiled inference

For high-resolution frames, `predict.tiling` runs the detector over overlapping
crops at native resolution and merges with class-aware NMS. Tiles are strided by
`(1 - overlap) × tile_size` so any object smaller than the overlap band lies
wholly inside at least one tile; the last tile in each direction is pulled flush
with the edge rather than padded; the full frame is optionally also run to catch
objects larger than one tile.

## The Gradio app

```
python -m smalldet.cli app configs/app.yaml
```

**Detect** — upload an image, adjust confidence and max detections, toggle tiled
inference. The summary breaks results into the same small/medium/large buckets
`AP_small` and `AP_medium` are computed over, so the demo and the metrics tell
the same story. Objects under 32² px are outlined in a distinct colour.

**Finetune** — configure and run a finetuning job from the browser:

- **Dataset and splits** — point at train/val/test COCO files, or split a single
  annotation file in place. Splitting is per *image*, never per annotation:
  putting two objects from one image on opposite sides leaks the exact pixels
  being tested on into training.
- **Model** — architecture, pretrained weights, input resolution, anchor pyramid.
- **Strategy** — all four modes, each with its trade-off spelled out.
- **Hyper-parameters** — optimizer, LR, weight decay, schedule, warmup, epochs,
  batch size, gradient accumulation, augmentation toggles.
- **Evaluation** — which metric selects the best checkpoint, auto area buckets,
  early stopping.

**Validate settings** checks the configuration *before* training and warns about
combinations that are legal but usually mistakes — selecting on `AP_small` while
leaving anchors at the 32–512 default, a learning rate that will diverge, warmup
switched off, LR milestones that fall outside the run. **Export config** writes
the YAML the page built, so any UI run is reproducible with
`python -m smalldet.cli train <exported.yaml>`.

## Tests

```
pytest tests/                  # everything
pytest tests/e2e               # the Gradio app: service, HTTP API, browser
python tests/run_tests.py      # dependency-free fallback when pytest is absent
```

The Gradio app is tested in three layers, cheapest first, so a failure localises
itself: the services with no Gradio import at all, then the app as a real HTTP
server driven through `gradio_client`, then a real browser via Playwright. Every
component carries an explicit `elem_id` (see `ELEM_IDS` in
`smalldet/app/gradio_app.py`) — Gradio's generated ids change between releases
and layout edits, so those constants are the testing contract.

Tests use `weights: null` throughout: pretrained weights are a ~170 MB download,
and randomly initialised weights exercise every code path identically.

## Datasets

- `Dataset/synthetic` — generated by `smalldet.cli synth`. Mostly 6–16px objects
  with a deliberate 34–70px minority, so both `AP_small` and `AP_medium` have
  ground truth to score and neither reports the `-1` sentinel.
- `configs/finetune_toy_voc.yaml` — a 60-image VOC-in-COCO set. Not
  small-object-dominated (roughly 13 small / 46 medium / 39 large), which makes
  it the useful counterpart: all three buckets are populated, so all three
  metrics are real numbers.

## Further reading

`.claude/skills/small-object-detection/SKILL.md` records the design decisions
behind this project — the five levers, the family-specific model surgery, the
COCOeval reimplementation checklist, and the gotchas that cost real time.
