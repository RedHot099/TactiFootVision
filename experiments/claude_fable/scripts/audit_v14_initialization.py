"""Measure initialization equivalence without changing the frozen v14 implementation."""

import argparse
import json
from pathlib import Path

import torch

from xgpoc.v13.protocol import digest
from xgpoc.v14.models import MultiViewViTXG
from xgpoc.v14.views import LOCAL_SIZE


def check(model, image, views):
    model.eval()
    captured = {}

    def capture(module, args, kwargs, output):
        captured.update(input=args[0], output=output, pad=kwargs["src_key_padding_mask"])

    handle = model.fusion.register_forward_hook(capture, with_kwargs=True)
    rows = []
    try:
        with torch.inference_mode():
            base = model.base(image)
            full_count = model.base.backbone.forward_features(image).shape[1]
            for name, flags in (("missing", [0., 0.]), ("shooter", [1., 0.]), ("both", [1., 1.])):
                valid = image.new_tensor([flags]).expand(len(image), -1)
                output = model(image, views, valid)
                pad = captured["pad"].clone()
                pad[:, full_count:] = True
                restored = model.base.head(captured["output"], key_padding_mask=pad)
                rows.append({"views": name,
                             "fusion_identity_error": float((captured["input"] - captured["output"]).abs().max()),
                             "base_logit_error": float((base - output).abs().max()),
                             "masked_local_logit_error": float((base - restored).abs().max())})
    finally:
        handle.remove()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="dinov2_l")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("research/v15_diagnostics/initialization.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new output path; this audit does not overwrite evidence")
    torch.manual_seed(1300)
    model = MultiViewViTXG(backbone=args.backbone, pretrained=False, grad_checkpointing=False).to(args.device)
    image = torch.randn(1, 3, 252, 448, device=args.device)
    views = torch.randn(1, 2, 3, LOCAL_SIZE[1], LOCAL_SIZE[0], device=args.device)
    rows = check(model, image, views)
    if any(r["fusion_identity_error"] > 1e-6 or r["masked_local_logit_error"] > 1e-5 for r in rows):
        raise AssertionError("Unexpected failure of identity fusion or masked baseline reconstruction")
    result = {"backbone": args.backbone, "seed": 1300, "pretrained": False, "input": "synthetic, full image dimensions",
              "rows": rows, "source_sha256": {str(p): digest(p) for p in
                  (Path(__file__), Path("xgpoc/v14/models.py"), Path("xgpoc/models/d_foundation.py"))},
              "interpretation": "Identity fusion does not imply G equivalence: local tokens enter pooling. "
                                "This mechanism check does not attribute the observed training degradation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
