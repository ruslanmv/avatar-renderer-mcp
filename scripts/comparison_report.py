#!/usr/bin/env python3
"""
comparison_report.py — metrics + GIFs + dataset upload + README rewrite for the
"Generation Methods — Quality Comparison" section.

Invoked by scripts/regen_comparison.sh after the per-engine renders exist. Reads a
TSV of successfully-rendered variants and the directory holding `<label>.mp4`, then:
  • measures mouth sharpness / lip motion / face flicker / background motion,
  • renders a looping `<label>.gif`,
  • (optional) uploads mp4+gif+metrics.json to the HF comparison dataset,
  • (optional) rewrites README.md between <!-- COMPARISON:START/END -->.

variants TSV columns:  label<TAB>engine<TAB>quality<TAB>star(0|1)<TAB>caption
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

DATASET_BASE = "https://huggingface.co/datasets/{ds}/resolve/main/{name}"

INTRO = (
    "All variants generated from the **same portrait + same text** with the "
    "production `orchestrate()` pipeline. Samples hosted on the "
    "[comparison dataset](https://huggingface.co/datasets/{ds}).\n\n"
    "The Wav2Lip engine is the **faithful dev-v0.1.25 full-face pipeline**: the whole\n"
    "predicted face crop is pasted back each frame and restored with GFPGAN per-frame\n"
    "(no mouth-only blend) — the most natural mouth. The `wav2lip_band` column is the\n"
    "alternative anti-flicker compositing (mouth band on a static base)."
)


def _good_box(frames):
    """Find a real face box across a few frames; fall back to a centered portrait
    box if detection fails (otherwise the ROI = whole frame, which makes the
    background dominate sharpness and zeroes out 'mouth motion')."""
    h, w = frames[0].shape[:2]
    idxs = sorted(set([0, len(frames) // 3, len(frames) // 2]))
    for i in idxs:
        try:
            from app.lipsync import _detect_face_box
            x, y, x2, y2 = _detect_face_box(frames[i])
        except Exception:
            x, y, x2, y2 = 0, 0, w, h
        if (x2 - x) * (y2 - y) < 0.85 * w * h and (x2 - x) > 0.10 * w:
            return x, y, x2, y2
    return int(0.30 * w), int(0.12 * h), int(0.70 * w), int(0.66 * h)


def measure(mp4: str, work: Path) -> dict:
    import cv2
    import numpy as np

    d = work / "frames"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, str(d / "%04d.png")], check=True)
    files = sorted(glob.glob(str(d / "*.png")))
    if not files:
        return {"frames": 0}
    col = [cv2.imread(f) for f in files]
    gray = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype("int16") for c in col]
    x, y, x2, y2 = _good_box(col)
    fw, fh = x2 - x, y2 - y
    n = len(gray)

    def region_motion(a, b, ry1, ry2, rx1, rx2):
        return float(np.abs(gray[a][ry1:ry2, rx1:rx2] - gray[b][ry1:ry2, rx1:rx2]).mean())

    mouth = lambda i, j: region_motion(i, j, y + int(fh * .62), y + int(fh * .92), x + int(fw * .20), x + int(fw * .80))
    face = lambda i, j: region_motion(i, j, y + int(fh * .10), y + int(fh * .45), x + int(fw * .20), x + int(fw * .80))
    bg = lambda i, j: region_motion(i, j, 5, min(110, gray[0].shape[0]), 5, min(110, gray[0].shape[1]))

    mm = float(np.mean([mouth(i, i + 1) for i in range(n - 1)])) if n > 1 else 0.0
    ff = float(np.mean([face(i, i + 1) for i in range(n - 1)])) if n > 1 else 0.0
    bb = float(np.mean([bg(i, i + 1) for i in range(n - 1)])) if n > 1 else 0.0
    mid = col[n // 2]
    roi = mid[y + int(fh * .60):y + int(fh * .90), x + int(fw * .20):x + int(fw * .80)]
    sharp = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if roi.size else 0.0
    return {"frames": n, "mouth_sharpness": round(sharp, 1), "mouth_motion": round(mm, 3),
            "face_flicker": round(ff, 3), "bg_motion": round(bb, 3)}


def make_gif(mp4: str, gif: str, work: Path) -> None:
    pal = str(work / "pal.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=10,scale=300:-1:flags=lanczos,palettegen", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", pal, "-lavfi",
                    "fps=10,scale=300:-1:flags=lanczos[x];[x][1:v]paletteuse", gif], check=True)


def build_block(rows: list[dict], dataset: str) -> str:
    def url(name):
        return DATASET_BASE.format(ds=dataset, name=name)

    head = "| " + " | ".join(f"`{r['label']}`" + (" ⭐" if r["star"] else "") for r in rows) + " |"
    sep = "|" + ":---:|" * len(rows)
    gifs = "| " + " | ".join(f"![{r['label']}]({url(r['label']+'.gif')})" for r in rows) + " |"
    caps = "| " + " | ".join(r["caption"] for r in rows) + " |"
    mp4s = "| " + " | ".join(f"[mp4]({url(r['label']+'.mp4')})" for r in rows) + " |"

    m_head = "| Method | Mouth sharpness ↑ | Lip motion ↑ | Face flicker ↓ | Background motion ↓ |"
    m_sep = "|---|---:|---:|---:|---:|"
    m_rows = []
    for r in rows:
        m = r["metrics"]
        name = f"**`{r['label']}`** ⭐" if r["star"] else f"`{r['label']}`"
        m_rows.append(f"| {name} | {m.get('mouth_sharpness','-')} | {m.get('mouth_motion','-')} | "
                      f"{m.get('face_flicker','-')} | {m.get('bg_motion','-')} |")

    return "\n".join([
        INTRO.format(ds=dataset), "",
        head, sep, gifs, caps, mp4s, "",
        "**Objective metrics** (measured on the rendered frames):", "",
        m_head, m_sep, *m_rows,
    ])


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    a, b = "<!-- COMPARISON:START -->", "<!-- COMPARISON:END -->"
    if a not in text or b not in text:
        raise SystemExit(f"markers {a}/{b} not found in {readme}")
    pre, rest = text.split(a, 1)
    _, post = rest.split(b, 1)
    readme.write_text(f"{pre}{a}\n{block}\n{b}{post}")
    print(f"README updated ({readme})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variants", required=True, help="TSV: label engine quality star caption")
    ap.add_argument("--dataset", default="ruslanmv/avatar-renderer-samples")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    out = Path(args.dir)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for line in Path(args.variants).read_text().splitlines():
        if not line.strip():
            continue
        label, engine, quality, star, caption = (line.split("\t") + ["", "", "", "0", ""])[:5]
        mp4 = out / f"{label}.mp4"
        if not mp4.exists():
            print(f"skip {label}: no mp4")
            continue
        gif = out / f"{label}.gif"
        print(f"measuring {label} ...")
        metrics = measure(str(mp4), work)
        make_gif(str(mp4), str(gif), work)
        rows.append({"label": label, "engine": engine, "quality": quality,
                     "star": star == "1", "caption": caption, "metrics": metrics})

    if not rows:
        raise SystemExit("no rendered variants found")

    (out / "metrics.json").write_text(json.dumps(
        {r["label"]: {"engine": r["engine"], **r["metrics"]} for r in rows}, indent=2))

    if args.upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--upload set but HF_TOKEN is not in the environment")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        for r in rows:
            for ext in ("mp4", "gif"):
                p = out / f"{r['label']}.{ext}"
                api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                                repo_id=args.dataset, repo_type="dataset",
                                commit_message=f"comparison: {p.name}")
                print("uploaded", p.name)
        api.upload_file(path_or_fileobj=str(out / "metrics.json"), path_in_repo="metrics.json",
                        repo_id=args.dataset, repo_type="dataset", commit_message="comparison: metrics.json")
        print("uploaded metrics.json")

    block = build_block(rows, args.dataset)
    (out / "comparison_block.md").write_text(block)
    print("\n--- comparison block ---\n" + block + "\n")
    if args.update_readme:
        update_readme(Path(args.readme), block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
