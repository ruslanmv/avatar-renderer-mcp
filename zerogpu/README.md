---
title: Avatar Renderer
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: space_app.py
pinned: true
license: apache-2.0
short_description: GPU talking-avatar video generator (ZeroGPU)
---

# Avatar Renderer — ZeroGPU

Generate a talking-avatar video from a **portrait image** + **audio clip**, on
Hugging Face **ZeroGPU**.

- **UI:** the Gradio app on this Space.
- **API:** call `/predict` with `@gradio/client` (used by the Vercel frontend).
- **Inference:** `generate()` is `@spaces.GPU`-decorated; it reuses the repo
  pipeline (`app.render.render`), which runs the full GPU lip-sync pipeline when
  models are present and otherwise falls back to a fast ffmpeg renderer so a
  video is always returned.

## Enabling real lip-sync
ZeroGPU attaches a GPU only inside `generate()`. To produce true lip-sync,
load an **in-process** model + weights there (subprocess-based pipelines are not
GPU-accelerated under ZeroGPU). The fallback keeps the product working until
those weights are wired in.
