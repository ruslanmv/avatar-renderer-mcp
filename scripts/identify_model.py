#!/usr/bin/env python3
"""
scripts/identify_model.py

Diff2Lip checkpoint “fingerprint” utility.

Goals:
- Load a Diff2Lip checkpoint (.pth/.pt) WITHOUT importing Diff2Lip code.
- Infer likely UNet hyperparameters from tensor shapes:
  - model_channels (base channels / num_channels)
  - in_channels / out_channels
  - learn_sigma (out_channels==6 is a strong hint)
  - whether an audio encoder exists (keys under audio_encoder.*)
  - whether attention exists + a few attention channel sizes
  - rough channel_mult guess from conv widths
- Detect checkpoint dtype (fp16 vs fp32) and basic param counts.

NEW in this updated version:
- If no argument is passed, it will auto-scan common Diff2Lip locations:
    models/diff2lip/e7.24.1.3_model260000_paper.pt
    models/diff2lip/Diff2Lip.pth
  and print reports for BOTH if present.
- Supports passing a directory: it will fingerprint all *.pt/*.pth inside.
- Prints top-level checkpoint keys (useful to distinguish “paper” style vs legacy).
- Prints a “suggested MODEL_FLAGS” block based on what we can infer.

Examples:
  python scripts/identify_model.py
  python scripts/identify_model.py models/diff2lip/e7.24.1.3_model260000_paper.pt
  python scripts/identify_model.py models/diff2lip --verbose
  python scripts/identify_model.py --json /tmp/d2l_fingerprint.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

BANNER = "=" * 80


# ---------------------------
# Loading helpers
# ---------------------------

def _torch_load_cpu(path: Path) -> Any:
    # weights_only exists in newer torch; keep backward compatible
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _is_state_dict_like(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(isinstance(v, torch.Tensor) for v in obj.values())


def _extract_state_dict(obj: Any) -> Tuple[Dict[str, torch.Tensor], List[str], str]:
    """
    Returns: (state_dict, top_level_keys, format_note)
    """
    if _is_state_dict_like(obj):
        return obj, sorted(list(obj.keys()))[:20], "raw_state_dict"

    if isinstance(obj, dict):
        top_keys = sorted(list(obj.keys()))
        # common patterns
        for k in ("state_dict", "model", "net", "ema", "model_state", "module"):
            v = obj.get(k)
            if _is_state_dict_like(v):
                return v, top_keys[:40], f"dict['{k}']"

        # sometimes model is nested deeper
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            v = obj["state_dict"]
            if _is_state_dict_like(v):
                return v, top_keys[:40], "dict['state_dict']"

        raise ValueError(
            "Unrecognized checkpoint dict format. "
            f"Top-level keys: {top_keys[:50]}"
        )

    raise ValueError(f"Unrecognized checkpoint object type: {type(obj)}")


def _load_checkpoint_state_dict(path: Path) -> Tuple[Dict[str, torch.Tensor], List[str], str]:
    obj = _torch_load_cpu(path)
    return _extract_state_dict(obj)


# ---------------------------
# Fingerprinting helpers
# ---------------------------

def _get(sd: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    v = sd.get(key)
    return v if isinstance(v, torch.Tensor) else None


def _max_block_index(sd: Dict[str, torch.Tensor], prefix: str) -> Optional[int]:
    m = -1
    for k in sd.keys():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        head = rest.split(".", 1)[0]
        if head.isdigit():
            m = max(m, int(head))
    return m if m >= 0 else None


def _guess_channel_mult(model_channels: int, sd: Dict[str, torch.Tensor]) -> Optional[List[int]]:
    # scan conv out-channels and take max ratio against base
    conv_outs: List[int] = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 4 and k.endswith(".weight"):
            conv_outs.append(int(v.shape[0]))
    if not conv_outs or model_channels <= 0:
        return None
    max_c = max(conv_outs)
    ratio = max_c / float(model_channels)

    # snap to common guided-diffusion patterns
    if ratio >= 7.5:
        return [1, 2, 4, 8]
    if ratio >= 3.5:
        return [1, 2, 3, 4]
    if ratio >= 1.9:
        return [1, 2, 2, 2]
    return None


def _has_attention(sd: Dict[str, torch.Tensor]) -> bool:
    for k in sd.keys():
        if ".qkv." in k or ".attn." in k or "attention" in k.lower():
            return True
    return False


def _attention_channel_samples(sd: Dict[str, torch.Tensor], limit: int = 6) -> List[int]:
    """
    Try to infer a few attention channel sizes from qkv weights.
    In guided-diffusion, qkv weight often has shape:
      - [3*C, C, 1, 1] (conv1x1) or
      - [3*C, C] (linear)
    We return some inferred C values.
    """
    chans: List[int] = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "qkv" not in k or not k.endswith("weight"):
            continue
        if v.ndim == 4 and v.shape[0] % 3 == 0:
            c = int(v.shape[0] // 3)
            chans.append(c)
        elif v.ndim == 2 and v.shape[0] % 3 == 0:
            c = int(v.shape[0] // 3)
            chans.append(c)
        if len(chans) >= limit:
            break
    # unique + sorted
    return sorted(list(dict.fromkeys(chans)))


def _dtype_summary(sd: Dict[str, torch.Tensor]) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for v in sd.values():
        if not isinstance(v, torch.Tensor):
            continue
        name = str(v.dtype).replace("torch.", "")
        counts[name] = counts.get(name, 0) + v.numel()
    if not counts:
        return ("unknown", {})
    # dominant dtype by numel
    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    return dominant, counts


def _param_count(sd: Dict[str, torch.Tensor]) -> int:
    total = 0
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            total += int(v.numel())
    return total


def _estimate_in_channels(sd: Dict[str, torch.Tensor]) -> Tuple[Optional[int], Optional[int]]:
    """
    input_blocks.0.0.weight is usually the first conv:
      [out_ch, in_ch, kh, kw]
    """
    w = _get(sd, "input_blocks.0.0.weight")
    if w is not None and w.ndim == 4:
        return int(w.shape[1]), int(w.shape[0])
    return None, None


def _estimate_out_channels(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    out.2.weight is commonly final conv:
      [out_ch, in_ch, kh, kw]
    """
    w = _get(sd, "out.2.weight")
    if w is not None and w.ndim == 4:
        return int(w.shape[0])
    return None


def _time_embed_dim(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    w = _get(sd, "time_embed.0.weight")
    if w is not None and w.ndim == 2:
        return int(w.shape[0])
    return None


def _has_audio_encoder(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("audio_encoder.") for k in sd.keys())


def _audio_encoder_widths(sd: Dict[str, torch.Tensor], max_items: int = 10) -> Optional[List[int]]:
    if not _has_audio_encoder(sd):
        return None
    widths = set()
    for k, v in sd.items():
        if not k.startswith("audio_encoder."):
            continue
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim in (2, 4):
            widths.add(int(v.shape[0]))
        if len(widths) >= max_items:
            break
    return sorted(list(widths)) if widths else None


def _guess_ref_conditioning(in_channels: Optional[int]) -> Optional[str]:
    """
    Heuristic: many ref-conditioned guided-diffusion setups concatenate:
      - input frame (3ch) + reference (3ch) + masked/extra (>=0)
    Diff2Lip commonly ends up >= 6 in-ch when ref is used.
    """
    if in_channels is None:
        return None
    if in_channels >= 6:
        return "likely (in_channels>=6)"
    return "unlikely (in_channels<6)"


@dataclass
class Fingerprint:
    path: str
    file_size_mb: float
    ckpt_format: str
    top_level_keys_sample: List[str]

    num_tensors: int
    total_params: int

    dominant_dtype: str
    dtype_numel_breakdown: Dict[str, int]

    # Shapes / core inferred params
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    learn_sigma_likely: Optional[bool] = None
    model_channels: Optional[int] = None
    time_embed_dim: Optional[int] = None

    # Structure
    input_blocks_max_index: Optional[int] = None
    output_blocks_max_index: Optional[int] = None
    channel_mult_guess: Optional[List[int]] = None

    # Conditioning
    has_audio_encoder: bool = False
    audio_encoder_widths_sample: Optional[List[int]] = None
    ref_conditioning_guess: Optional[str] = None

    # Attention
    attention_present: bool = False
    attention_channel_samples: Optional[List[int]] = None

    # Suggested flags (best-effort)
    suggested_model_flags: Optional[List[str]] = None


def fingerprint_checkpoint(path: Path, verbose: bool = False) -> Fingerprint:
    sd, top_keys, fmt = _load_checkpoint_state_dict(path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    in_ch, base_ch = _estimate_in_channels(sd)
    out_ch = _estimate_out_channels(sd)
    tdim = _time_embed_dim(sd)

    dominant_dtype, dtype_breakdown = _dtype_summary(sd)

    fp = Fingerprint(
        path=str(path.resolve()),
        file_size_mb=float(f"{file_size_mb:.2f}"),
        ckpt_format=fmt,
        top_level_keys_sample=top_keys[:40],

        num_tensors=len(sd),
        total_params=_param_count(sd),

        dominant_dtype=dominant_dtype,
        dtype_numel_breakdown=dtype_breakdown,

        in_channels=in_ch,
        out_channels=out_ch,
        learn_sigma_likely=(out_ch == 6) if out_ch is not None else None,
        model_channels=base_ch,
        time_embed_dim=tdim,

        input_blocks_max_index=_max_block_index(sd, "input_blocks."),
        output_blocks_max_index=_max_block_index(sd, "output_blocks."),
        channel_mult_guess=_guess_channel_mult(base_ch, sd) if base_ch else None,

        has_audio_encoder=_has_audio_encoder(sd),
        audio_encoder_widths_sample=_audio_encoder_widths(sd),
        ref_conditioning_guess=_guess_ref_conditioning(in_ch),

        attention_present=_has_attention(sd),
        attention_channel_samples=_attention_channel_samples(sd) if _has_attention(sd) else [],
        suggested_model_flags=None,
    )

    # Suggested MODEL_FLAGS (best-effort; Diffusion/Sampling flags are not inferable from weights alone)
    flags: List[str] = []
    if fp.model_channels is not None:
        flags += ["--num_channels", str(fp.model_channels)]
    if fp.learn_sigma_likely is not None:
        flags += ["--learn_sigma", "True" if fp.learn_sigma_likely else "False"]
    # fp16 hint
    flags += ["--use_fp16", "True" if fp.dominant_dtype in ("float16", "bfloat16") else "False"]
    # attention hint
    if fp.attention_present:
        flags += ["--attention_resolutions", "32,16,8"]  # cannot infer exactly; this is the common Diff2Lip paper setup
    # resblocks guess: Diff2Lip paper uses 2; we can’t infer reliably, but 2 is common
    flags += ["--num_res_blocks", "2", "--resblock_updown", "True", "--class_cond", "False"]

    fp.suggested_model_flags = flags

    if verbose:
        # Add a few key probes if available
        probes = [
            "input_blocks.0.0.weight",
            "time_embed.0.weight",
            "out.2.weight",
        ]
        print("\n[verbose] Probe tensor shapes:")
        for k in probes:
            v = _get(sd, k)
            if v is None:
                continue
            print(f"  - {k}: shape={tuple(v.shape)} dtype={str(v.dtype).replace('torch.', '')}")

        # Show a few sample keys (useful for debugging naming differences)
        sample_keys = sorted(sd.keys())[:25]
        print("\n[verbose] First 25 state_dict keys:")
        for k in sample_keys:
            print(f"  - {k}")

    return fp


def print_report(fp: Fingerprint) -> None:
    print(BANNER)
    print("Diff2Lip checkpoint fingerprint")
    print(BANNER)
    print(f"Path:           {fp.path}")
    print(f"File size:      {fp.file_size_mb} MB")
    print(f"Format:         {fp.ckpt_format}")
    print(f"Top-level keys: {fp.top_level_keys_sample}")
    print()

    print("Core inferred parameters (from tensor shapes)")
    print("-" * 80)
    print(f"in_channels:           {fp.in_channels}")
    print(f"model_channels/base:   {fp.model_channels}   (this is your --num_channels candidate)")
    print(f"out_channels:          {fp.out_channels}")
    print(f"learn_sigma likely:    {fp.learn_sigma_likely}")
    print(f"time_embed_dim:        {fp.time_embed_dim}")
    print()

    print("Numerics / size")
    print("-" * 80)
    print(f"num_tensors:           {fp.num_tensors}")
    print(f"total_params:          {fp.total_params:,}")
    print(f"dominant dtype:        {fp.dominant_dtype}")
    # show only a couple dtypes to keep it readable
    if fp.dtype_numel_breakdown:
        top = sorted(fp.dtype_numel_breakdown.items(), key=lambda kv: kv[1], reverse=True)[:4]
        print(f"dtype breakdown (top): {top}")
    print()

    print("Structure heuristics")
    print("-" * 80)
    print(f"input_blocks max idx:  {fp.input_blocks_max_index}")
    print(f"output_blocks max idx: {fp.output_blocks_max_index}")
    print(f"channel_mult guess:    {fp.channel_mult_guess}")
    print()

    print("Conditioning")
    print("-" * 80)
    print(f"has_audio_encoder:         {fp.has_audio_encoder}")
    print(f"audio_encoder widths samp: {fp.audio_encoder_widths_sample}")
    print(f"ref conditioning guess:    {fp.ref_conditioning_guess}")
    print()

    print("Attention")
    print("-" * 80)
    print(f"attention present:         {fp.attention_present}")
    print(f"attention channel samples: {fp.attention_channel_samples}")
    print()

    print("Suggested MODEL_FLAGS (best-effort)")
    print("-" * 80)
    print(" ".join(fp.suggested_model_flags or []))
    print(BANNER)


# ---------------------------
# CLI / multi-scan
# ---------------------------

def _default_candidates() -> List[Path]:
    cands = [
        Path("models/diff2lip/e7.24.1.3_model260000_paper.pt"),
        Path("models/diff2lip/Diff2Lip.pth"),
        Path("./models/diff2lip/e7.24.1.3_model260000_paper.pt"),
        Path("./models/diff2lip/Diff2Lip.pth"),
    ]
    seen = set()
    out: List[Path] = []
    for p in cands:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.exists():
            out.append(p)
    return out


def _expand_targets(arg: Optional[str]) -> List[Path]:
    if not arg:
        return _default_candidates()

    p = Path(arg).expanduser()
    if p.is_dir():
        files = []
        for ext in ("*.pt", "*.pth", "*.ckpt"):
            files.extend(sorted(p.glob(ext)))
        return files

    return [p]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fingerprint Diff2Lip checkpoint(s).")
    ap.add_argument(
        "target",
        nargs="?",
        default="",
        help="Checkpoint path OR a directory containing checkpoints. If omitted, scans common Diff2Lip paths.",
    )
    ap.add_argument("--json", default="", help="Write JSON report. If multiple checkpoints, writes a list.")
    ap.add_argument("--verbose", action="store_true", help="Print extra probe shapes and sample keys.")
    args = ap.parse_args()

    targets = _expand_targets(args.target if args.target else None)
    if not targets:
        print(BANNER)
        print("❌ No checkpoints found.")
        print(BANNER)
        print("Looked for:")
        for p in _default_candidates() or [
            Path("models/diff2lip/e7.24.1.3_model260000_paper.pt"),
            Path("models/diff2lip/Diff2Lip.pth"),
        ]:
            print(f"  - {p}")
        print("\nFix:")
        print("  - Put a checkpoint in models/diff2lip/, or")
        print("  - Run: python scripts/identify_model.py /path/to/checkpoint.pt")
        return 2

    reports: List[Fingerprint] = []
    for ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"\n❌ Missing: {ckpt_path}")
            continue
        try:
            fp = fingerprint_checkpoint(ckpt_path, verbose=args.verbose)
            print_report(fp)
            reports.append(fp)
        except Exception as e:
            print(BANNER)
            print(f"❌ Failed to fingerprint: {ckpt_path}")
            print(BANNER)
            print(f"Error: {e}")
            print()

    if args.json:
        out_json = Path(args.json).expanduser()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload: Union[dict, list]
        if len(reports) == 1:
            payload = asdict(reports[0])
        else:
            payload = [asdict(r) for r in reports]
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n✅ Wrote JSON report: {out_json.resolve()}")

    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
