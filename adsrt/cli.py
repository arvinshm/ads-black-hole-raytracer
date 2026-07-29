from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .camera import PinholeCamera
from .config import load_json, load_metric
from .renderer import Renderer
from .sources import BoundaryPointSource
from .tracer import RayTracer, TraceConfig
from .validation import run_validation_suite


def render_from_config(config_path: str, output_override: str | None = None) -> None:
    config = load_json(config_path)
    metric = load_metric(config["metric"]["name"], config["metric"].get("parameters", {}))
    camera_cfg = config["camera"]
    source_cfg = config["source"]
    trace_cfg = config["trace"]

    camera = PinholeCamera(
        metric=metric,
        position=np.asarray(camera_cfg["position"], dtype=float),
        width=int(camera_cfg.get("width", 128)),
        height=int(camera_cfg.get("height", 128)),
        horizontal_fov_degrees=float(camera_cfg.get("horizontal_fov_degrees", 100.0)),
        angular_velocity=camera_cfg.get("angular_velocity"),
    )
    tracer = RayTracer(metric, TraceConfig(**trace_cfg))
    source = BoundaryPointSource(**source_cfg)
    result = Renderer(camera, tracer, source).render(progress=True, workers=int(config.get("workers", 1)))
    output = output_override or config.get("output", "outputs/render")
    result.save(output, metadata=config)
    unique, counts = np.unique(result.status, return_counts=True)
    print("status counts:", dict(zip(unique.tolist(), counts.tolist(), strict=True)))
    print("saved:", Path(output).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(prog="adsrt", description="AdS black-hole ray tracer")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render from a JSON configuration")
    render.add_argument("config")
    render.add_argument("--output")

    validate = sub.add_parser("validate", help="run analytic/numerical validation checks")
    validate.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args()
    if args.command == "render":
        render_from_config(args.config, args.output)
    elif args.command == "validate":
        report = run_validation_suite()
        if args.as_json:
            print(json.dumps(report, indent=2, default=float))
        else:
            for name, details in report.items():
                print(name)
                for key, value in details.items():
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
