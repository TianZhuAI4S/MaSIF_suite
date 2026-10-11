"""Convert and numerically validate all successfully exported checkpoint families."""

import argparse
import json
from pathlib import Path
import torch
from masif_torch.checkpoints import convert
from masif_torch.validation import validate, validate_adam, write_report


def main():
    """Record strict and explicit gradient-roundoff results separately, without hiding failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--historical-gradient-roundoff", action="store_true",
                        help="Historical search only: grad atol=.01 rtol=.001; retain strict errors")
    args = parser.parse_args()
    devices = ["cpu", "cuda"] if args.cuda else ["cpu"]
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA validation, but CUDA is unavailable")
    reports = []
    for path in sorted(args.exports.rglob("config.json")):
        export_dir = path.parent
        relative = export_dir.relative_to(args.exports)
        output = args.output / relative / "converted.pt"
        config = json.loads(path.read_text())
        manifest = json.loads((export_dir / "manifest.json").read_text())
        convert(export_dir, output, include_optimizer=manifest.get("optimizer") is not None)
        for device in devices:
            report = validate(export_dir, output, device)
            # Identify pinned source bytes, independently of the user's export directory name.
            historical = manifest.get("files", {}).get("model.data-00000-of-00001") == (
                "4541e79570cdefe0e741359a0601e9b8391a136544c7c26070cde1ec59fa7133")
            if historical and args.historical_gradient_roundoff:
                accepted = validate(export_dir, output, device, grad_atol=1e-2, grad_rtol=1e-3)
                accepted["strict_result"] = report
                accepted["gradient_policy"] = "Historical high-scale search: FP32 gradient cancellation; forward tolerance unchanged"
                report = accepted
            reports.append(report)
            print(f"{relative}: {device} passed={report['passed']}", flush=True)
    report = dict(models=reports, adam=validate_adam(args.exports / "adam_reference.npz"),
                  rejected=[json.loads(p.read_text()) for p in args.exports.rglob("rejected.json")])
    write_report(report, args.report)
    if not reports or not all(r["passed"] for r in reports) or not report["adam"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
