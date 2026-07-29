from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from adsrt.camera import PinholeCamera
from adsrt.metrics import KerrAdS4, PureAdS4, SchwarzschildAdS4
from adsrt.renderer import Renderer
from adsrt.sources import BoundaryPointSource
from adsrt.tracer import RayTracer, TraceConfig

ROOT = Path(__file__).resolve().parents[1]
OUTROOT = ROOT / "outputs" / "continuity_experiment"


@dataclass
class Case:
    label: str
    metric_name: str
    metric_parameters: dict
    output_name: str
    derived_physics: dict


def solve_a_for_target_omega_h_l(target: float, L: float = 1.0, mu: float = 0.5) -> float:
    """Return the Kerr parameter ``a`` for a requested ``Omega_H L``.

    The function first locates the extremal endpoint of the fixed-``(L, mu)``
    family and then bisects on the regular-horizon branch. Targets above the
    maximum attainable value for that family raise a clear error.
    """
    if target < 0.0:
        raise ValueError("target Omega_H L must be nonnegative")
    if abs(target) < 1.0e-15:
        return 0.0

    def make_metric(a: float) -> KerrAdS4:
        return KerrAdS4(L=L, mu=mu, a=a)

    valid_a = 0.0
    invalid_a = np.nextafter(L, 0.0)
    for _ in range(120):
        midpoint = 0.5 * (valid_a + invalid_a)
        try:
            make_metric(midpoint)
            valid_a = midpoint
        except ValueError:
            invalid_a = midpoint

    maximum_a = np.nextafter(valid_a, 0.0)
    maximum_ratio = make_metric(maximum_a).hawking_reall_ratio()
    if target > maximum_ratio + 1.0e-10:
        raise ValueError(
            f"Requested Omega_H L = {target:.8g}, but the maximum for "
            f"L={L} and mu={mu} is approximately {maximum_ratio:.8g}."
        )

    lower_a = 0.0
    upper_a = maximum_a
    for _ in range(120):
        midpoint = 0.5 * (lower_a + upper_a)
        ratio = make_metric(midpoint).hawking_reall_ratio()
        if ratio < target:
            lower_a = midpoint
        else:
            upper_a = midpoint
    return 0.5 * (lower_a + upper_a)


def make_cases(L: float = 1.0, mu: float = 0.5, targets: list[float] | None = None) -> list[Case]:
    cases: list[Case] = []
    cases.append(
        Case(
            label="Schwarzschild–AdS (mu=0.5)",
            metric_name="schwarzschild_ads",
            metric_parameters={"L": L, "mu": mu},
            output_name="schwarzschild_ads_mu0p50",
            derived_physics={"mu": mu, "L": L},
        )
    )
    if targets is None:
        targets = [0.0, 0.01, 0.03, 0.10, 0.30]
    for target in targets:
        a = solve_a_for_target_omega_h_l(target, L=L, mu=mu)
        metric = KerrAdS4(L=L, mu=mu, a=a)
        r_plus = metric.horizon_radius()
        tag = f"{target:.2f}".replace(".", "p")
        cases.append(
            Case(
                label=f"Kerr–AdS: Omega_H L = {target:.2f}",
                metric_name="kerr_ads",
                metric_parameters={"L": L, "mu": mu, "a": a},
                output_name=f"kerr_ads_omegaHL_{tag}",
                derived_physics={
                    "L": L,
                    "mu": mu,
                    "a": a,
                    "r_plus": r_plus,
                    "Omega_H": metric.horizon_angular_velocity(),
                    "Omega_H_times_L": metric.hawking_reall_ratio(),
                    "hawking_reall_bound": 1.0,
                },
            )
        )
    return cases


def save_status_png(status: np.ndarray, destination: Path) -> None:
    import matplotlib.pyplot as plt

    status_order = ["boundary", "origin_bridged", "horizon", "max_affine", "failed"]
    status_codes = np.full(status.shape, np.nan)
    for index, name in enumerate(status_order):
        status_codes[status == name] = index
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    image = ax.imshow(status_codes, origin="upper", vmin=-0.5, vmax=len(status_order) - 0.5)
    colorbar = fig.colorbar(image, ax=ax, ticks=range(len(status_order)))
    colorbar.ax.set_yticklabels(status_order)
    ax.set_title("Ray termination status")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def build_common_config_dict(
    case: Case,
    width: int,
    height: int,
    workers: int,
    preset: str,
    source_phi: float,
    source_theta: float,
    source_sigma: float,
    stationary_source: bool,
    output_path: Path,
) -> dict:
    source_velocity = [0.0, 0.0, 0.0] if stationary_source else [0.0, 0.0, 0.25]
    return {
        "metric": {"name": case.metric_name, "parameters": case.metric_parameters},
        "camera": {
            "position": [0.0, 20.0, float(np.pi / 2.0), float(np.pi)],
            "width": width,
            "height": height,
            "horizontal_fov_degrees": 168.0,
        },
        "source": {
            "theta": source_theta,
            "phi": source_phi,
            "wavelength_nm": 550.0,
            "peak_intensity": 1.0,
            "angular_sigma": source_sigma,
            "emitter_velocity_local": source_velocity,
        },
        "trace": {
            "outer_radius": 20.0,
            "max_affine": 60.0,
            "rtol": 2.0e-7,
            "atol": 2.0e-9,
            "max_step": 0.25,
            "horizon_buffer": 2.0e-4,
            "store_path": False,
            "max_rhs_evaluations": 2500,
            "ray_wall_time_limit_seconds": 2.0,
        },
        "output": str(output_path),
        "workers": workers,
        "continuity_experiment": {
            "preset": preset,
            "source_phi": source_phi,
            "source_theta": source_theta,
            "stationary_source": stationary_source,
        },
        "derived_physics": case.derived_physics,
        "render_notes": "Controlled continuity experiment with matched camera, source size, numerical tolerances, and common display scaling.",
    }


def render_case(config: dict) -> Path:
    metric_name = config["metric"]["name"]
    params = config["metric"]["parameters"]
    if metric_name == "schwarzschild_ads":
        metric = SchwarzschildAdS4(**params)
    elif metric_name == "pure_ads":
        metric = PureAdS4(**params)
    elif metric_name == "kerr_ads":
        metric = KerrAdS4(**params)
    else:
        raise ValueError(f"unknown metric {metric_name}")

    camera_cfg = config["camera"]
    source_cfg = config["source"]
    trace_cfg = config["trace"]

    camera = PinholeCamera(
        metric=metric,
        position=np.asarray(camera_cfg["position"], dtype=float),
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        horizontal_fov_degrees=float(camera_cfg["horizontal_fov_degrees"]),
        angular_velocity=camera_cfg.get("angular_velocity"),
    )
    tracer = RayTracer(metric, TraceConfig(**trace_cfg))
    source = BoundaryPointSource(**source_cfg)
    result = Renderer(camera, tracer, source).render(progress=True, workers=int(config["workers"]))
    output = Path(config["output"])
    result.save(output, metadata=config)
    save_status_png(result.status, output / "status.png")
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2, default=float))
    return output


def apply_common_exposure(case_dirs: Iterable[Path], destination: Path, title_lines: list[str]) -> None:
    arrays = []
    positive_values = []
    for folder in case_dirs:
        data = np.load(folder / "render_data.npz")
        arrays.append((folder, data))
        positive = data["intensity"][data["intensity"] > 0.0]
        if positive.size:
            positive_values.append(positive)
    if positive_values:
        all_positive = np.concatenate(positive_values)
        exposure = float(np.quantile(all_positive, 0.995))
    else:
        exposure = 1.0
    panels = []
    scale = 4
    label_height = 68
    for folder, data in arrays:
        rgb = np.clip(data["rgb"] / max(exposure, 1.0e-30), 0.0, 1.0)
        image = Image.fromarray((255.0 * rgb).astype(np.uint8), mode="RGB")
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (image.width, image.height + label_height), "white")
        panel.paste(image, (0, label_height))
        draw = ImageDraw.Draw(panel)
        line1 = folder.name.replace("_", " ")
        meta = json.loads((folder / "metadata.json").read_text())
        line2 = meta["metric"]["name"]
        line3 = f"{meta['camera']['width']} x {meta['camera']['height']} rays"
        for y, text in [(6, line1), (26, line2), (46, line3)]:
            bbox = draw.textbbox((0, 0), text)
            tw = bbox[2] - bbox[0]
            draw.text(((panel.width - tw) // 2, y), text, fill="black")
        panels.append(panel)

    total_width = sum(p.width for p in panels)
    body_height = max(p.height for p in panels)
    header_height = 60
    sheet = Image.new("RGB", (total_width, body_height + header_height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, text in enumerate(title_lines):
        bbox = draw.textbbox((0, 0), text)
        tw = bbox[2] - bbox[0]
        draw.text(((total_width - tw) // 2, 8 + 20 * i), text, fill="black")
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, header_height))
        x += panel.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled Kerr-AdS continuity experiment")
    parser.add_argument("--preset", choices=["exact_antipodal", "misaligned"], default="exact_antipodal")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--source-sigma", type=float, default=0.05)
    parser.add_argument(
        "--targets",
        type=str,
        default="0,0.01,0.03,0.10,0.30",
        help="Comma-separated target values of Omega_H L.",
    )
    parser.add_argument("--moving-source", action="store_true", help="Use the older azimuthally moving source instead of a stationary one")
    parser.add_argument("--output-root", type=str, default=str(OUTROOT))
    args = parser.parse_args()

    if args.preset == "exact_antipodal":
        source_phi = 0.0
    else:
        source_phi = 0.10
    source_theta = float(np.pi / 2.0)

    targets = [float(value.strip()) for value in args.targets.split(",") if value.strip()]
    cases = make_cases(L=args.L, mu=args.mu, targets=targets)
    outroot = Path(args.output_root) / args.preset
    outroot.mkdir(parents=True, exist_ok=True)

    case_dirs: list[Path] = []
    summary = {}
    for case in cases:
        folder = outroot / f"{case.output_name}_{args.width}x{args.height}"
        config = build_common_config_dict(
            case=case,
            width=args.width,
            height=args.height,
            workers=args.workers,
            preset=args.preset,
            source_phi=source_phi,
            source_theta=source_theta,
            source_sigma=args.source_sigma,
            stationary_source=not args.moving_source,
            output_path=folder,
        )
        print(f"\n=== Rendering {case.label} ===")
        print(json.dumps(config["metric"], indent=2))
        render_case(config)
        case_dirs.append(folder)
        data = np.load(folder / "render_data.npz")
        status = data["status"]
        unique, counts = np.unique(status, return_counts=True)
        summary[folder.name] = {str(k): int(v) for k, v in zip(unique, counts, strict=True)}

    (outroot / "render_summary.json").write_text(json.dumps(summary, indent=2))
    title = [
        f"Controlled Kerr-AdS continuity experiment: {args.preset}",
        f"L={args.L}, mu={args.mu}, source_sigma={args.source_sigma}, stationary_source={not args.moving_source}",
    ]
    apply_common_exposure(case_dirs, outroot / "continuity_comparison_common_exposure.png", title)
    print("\nSaved summary to", outroot / "render_summary.json")
    print("Saved contact sheet to", outroot / "continuity_comparison_common_exposure.png")


if __name__ == "__main__":
    main()
