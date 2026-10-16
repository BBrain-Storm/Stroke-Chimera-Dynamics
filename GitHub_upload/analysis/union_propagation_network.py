"""Union 324 stimulus-rooted propagation subnetworks for one subject."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DYNAMICS = (ROOT / "results" / "stimulation" / "dynamics"
                    / "Controls" / "subject_000.npz")
DEFAULT_DATASET = ROOT / "data" / "Dataset_Controls.npz"
DEFAULT_OUTPUT = (ROOT / "results" / "stimulation" / "union_network"
                  / "Controls" / "subject_000")
BASE_SEED = 42
SYSTEM_NAMES = ("VIS", "RSPT", "SMD", "SMV", "AUD", "CON", "VAN",
                "SAI", "DAN", "FPN", "DMN", "CP", "NON")
COMMUNITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "community_brain",
    [(0.10, 0.20, 0.60), (0.00, 0.60, 0.80), (0.20, 0.70, 0.50),
     (0.90, 0.90, 0.60), (0.95, 0.60, 0.20), (0.80, 0.10, 0.10)],
    N=13).reversed()
COMMUNITY_COLORS = COMMUNITY_CMAP(np.linspace(0, 1, 13))


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def load_and_validate(dynamics_path: Path, dataset_path: Path):
    with np.load(dynamics_path) as data:
        required = ("completed_stimulus_mask", "propagation_parent",
                    "propagation_stimulus_reachable_mask", "subject_index",
                    "group")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"dynamics file lacks {missing}")
        completed = np.asarray(data["completed_stimulus_mask"], dtype=bool)
        parent = np.asarray(data["propagation_parent"], dtype=np.int16)
        reachable = np.asarray(
            data["propagation_stimulus_reachable_mask"], dtype=bool)
        subject = int(data["subject_index"])
        group = str(data["group"])
    if completed.shape != (324,) or not np.all(completed):
        raise ValueError("all 324 stimulus experiments must be complete")
    if parent.shape != (324, 324) or reachable.shape != parent.shape:
        raise ValueError("propagation arrays must have shape (324, 324)")
    dataset = np.load(dataset_path, allow_pickle=False)
    template = np.asarray(dataset["template"], dtype=int).squeeze()
    xyz = np.asarray(dataset["xyz"], dtype=float)
    if template.shape != (324,) or xyz.shape != (324, 3):
        raise ValueError("dataset template/xyz shape mismatch")
    if np.any((template < 1) | (template > 13)):
        raise ValueError("functional-system labels must be in [1, 13]")
    return parent, reachable, template, xyz, group, subject


def union_edge_counts(parent: np.ndarray, reachable: np.ndarray) -> Counter:
    """Count each directed edge across all rooted propagation subnetworks."""
    counts: Counter = Counter()
    for stimulus in range(parent.shape[0]):
        targets = np.flatnonzero(reachable[stimulus] & (parent[stimulus] >= 0))
        for target in targets:
            counts[(int(parent[stimulus, target]), int(target))] += 1
    return counts


def save_tables(output_dir: Path, counts: Counter, denominator: int,
                template: np.ndarray, xyz: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "union_propagation_edges.csv"
    with edge_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["source", "target", "occurrence_count",
                  "occurrence_proportion", "source_system_id",
                  "source_system", "target_system_id", "target_system"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (source, target), count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({
                "source": source, "target": target,
                "occurrence_count": count,
                "occurrence_proportion": f"{count / denominator:.9f}",
                "source_system_id": int(template[source]),
                "source_system": SYSTEM_NAMES[template[source] - 1],
                "target_system_id": int(template[target]),
                "target_system": SYSTEM_NAMES[template[target] - 1],
            })
    incoming = np.zeros(324, dtype=int)
    outgoing = np.zeros(324, dtype=int)
    weighted_in = np.zeros(324, dtype=float)
    weighted_out = np.zeros(324, dtype=float)
    for (source, target), count in counts.items():
        outgoing[source] += 1
        incoming[target] += 1
        weighted_out[source] += count / denominator
        weighted_in[target] += count / denominator
    node_path = output_dir / "union_propagation_nodes.csv"
    with node_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["node", "system_id", "system", "x", "y", "z",
                  "union_in_degree", "union_out_degree",
                  "weighted_in_strength", "weighted_out_strength"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in range(324):
            writer.writerow({
                "node": node, "system_id": int(template[node]),
                "system": SYSTEM_NAMES[template[node] - 1],
                "x": xyz[node, 0], "y": xyz[node, 1], "z": xyz[node, 2],
                "union_in_degree": incoming[node],
                "union_out_degree": outgoing[node],
                "weighted_in_strength": f"{weighted_in[node]:.9f}",
                "weighted_out_strength": f"{weighted_out[node]:.9f}",
            })
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    sources = np.asarray([edge[0][0] for edge in ordered], dtype=np.int16)
    targets = np.asarray([edge[0][1] for edge in ordered], dtype=np.int16)
    occurrence_count = np.asarray([edge[1] for edge in ordered], dtype=np.int16)
    np.savez_compressed(
        output_dir / "union_propagation_network.npz",
        source=sources,
        target=targets,
        occurrence_count=occurrence_count,
        occurrence_proportion=occurrence_count.astype(np.float64) / denominator,
        stimulus_experiments=np.array(denominator, dtype=np.int16),
        system_ids=np.asarray(template, dtype=np.int16),
        xyz=np.asarray(xyz, dtype=np.float32),
        edge_semantics=np.array(
            "stimulus-reachable dominant dynamical propagation direction"),
    )


def _shortened_vectors(xyz: np.ndarray, sources: np.ndarray,
                       targets: np.ndarray, fraction: float = 0.10):
    start = xyz[sources, :2]
    delta = xyz[targets, :2] - start
    return (start[:, 0] + fraction * delta[:, 0],
            start[:, 1] + fraction * delta[:, 1],
            delta[:, 0] * (1.0 - 2.0 * fraction),
            delta[:, 1] * (1.0 - 2.0 * fraction))


def _draw_nodes(ax: plt.Axes, xyz: np.ndarray, template: np.ndarray) -> None:
    ax.scatter(xyz[:, 0], xyz[:, 1], s=8.5,
               c=COMMUNITY_COLORS[template - 1], edgecolors="0.20",
               linewidths=0.18, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(xyz[:, 0].min() - 7, xyz[:, 0].max() + 7)
    ax.set_ylim(xyz[:, 1].min() - 7, xyz[:, 1].max() + 7)
    ax.axis("off")


def _draw_union(ax: plt.Axes, xyz: np.ndarray, template: np.ndarray,
                sources: np.ndarray, targets: np.ndarray,
                occurrences: np.ndarray) -> None:
    for count, color, alpha, width in (
            (1, "#aeb5bf", 0.075, 0.00115),):
        mask = occurrences == count
        if np.any(mask):
            x, y, u, v = _shortened_vectors(xyz, sources[mask], targets[mask])
            ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=1,
                      color=color, alpha=alpha, width=width,
                      headwidth=3.2, headlength=4.0, headaxislength=3.5,
                      minlength=0, zorder=1)
    mask = occurrences >= 2
    if np.any(mask):
        x, y, u, v = _shortened_vectors(xyz, sources[mask], targets[mask])
        colors = mpl.colormaps["Blues"](
            0.42 + 0.53 * (occurrences[mask] - 3)
            / max(int(occurrences.max()) - 3, 1))
        ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=1,
                  color=colors, alpha=0.60, width=0.0019,
                  headwidth=3.4, headlength=4.2, headaxislength=3.7,
                  minlength=0, zorder=2)
    _draw_nodes(ax, xyz, template)
    ax.set_title("All-edge union", pad=2)


def _draw_recurrent(ax: plt.Axes, xyz: np.ndarray, template: np.ndarray,
                    sources: np.ndarray, targets: np.ndarray,
                    occurrences: np.ndarray, denominator: int):
    mask = occurrences >= 2
    shown = occurrences[mask] / denominator
    norm = Normalize(vmin=2 / denominator,
                     vmax=max(float(occurrences.max() / denominator),
                              2 / denominator + np.finfo(float).eps))
    x, y, u, v = _shortened_vectors(xyz, sources[mask], targets[mask])
    quiver = ax.quiver(
        x, y, u, v, shown, cmap="Blues", norm=norm,
        angles="xy", scale_units="xy", scale=1, alpha=0.72,
        width=0.0022, headwidth=3.8, headlength=4.6,
        headaxislength=4.0, minlength=0, zorder=2)
    _draw_nodes(ax, xyz, template)
    ax.set_title(r"Repeated backbone ($\geq2$ stimuli)", pad=2)
    return quiver, int(mask.sum())


def draw_figure(output_dir: Path, counts: Counter, denominator: int,
                template: np.ndarray, xyz: np.ndarray,
                group: str, subject: int) -> None:
    edges = sorted(counts)
    sources = np.asarray([edge[0] for edge in edges], dtype=int)
    targets = np.asarray([edge[1] for edge in edges], dtype=int)
    occurrences = np.asarray([counts[edge] for edge in edges], dtype=int)
    unique, frequencies = np.unique(occurrences, return_counts=True)

    fig = plt.figure(figsize=(7.2, 7.25), facecolor="white")
    grid = GridSpec(2, 2, figure=fig, height_ratios=(1.0, 0.54),
                    hspace=0.18, wspace=0.08)
    ax_union = fig.add_subplot(grid[0, 0])
    ax_recurrent = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[1, :])
    _draw_union(ax_union, xyz, template, sources, targets, occurrences)
    quiver, recurrent_count = _draw_recurrent(
        ax_recurrent, xyz, template, sources, targets, occurrences, denominator)
    colorbar_axis = ax_recurrent.inset_axes([0.50, -0.035, 0.44, 0.022])
    colorbar = fig.colorbar(quiver, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Edge prevalence across 324 stimuli", labelpad=2,
                       fontsize=6)
    colorbar.ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{100 * value:.1f}%"))
    colorbar.ax.tick_params(labelsize=5.8, length=2)

    bars = ax_hist.bar(unique, frequencies, width=0.72,
                       color=mpl.colormaps["Blues"](
                           np.linspace(0.38, 0.88, len(unique))),
                       edgecolor="0.25", linewidth=0.35)
    ax_hist.set_yscale("log")
    ax_hist.set_xticks(unique)
    ax_hist.set_xlabel("Number of stimulus-rooted subnetworks containing edge")
    ax_hist.set_ylabel("Number of union edges")
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.grid(axis="y", color="0.90", lw=0.5, zorder=0)
    for bar, value in zip(bars, frequencies):
        ax_hist.text(bar.get_x() + bar.get_width() / 2, value * 1.10,
                     f"{value:,}", ha="center", va="bottom", fontsize=6)
    single_fraction = frequencies[unique.tolist().index(1)] / len(edges)
    ax_hist.text(
        0.985, 0.95,
        f"Union edges = {len(edges):,}\n"
        rf"Repeated $\geq2$ = {recurrent_count:,}" "\n"
        f"Single-stimulus edges = {single_fraction:.1%}\n"
        f"Maximum prevalence = {occurrences.max() / denominator:.2%}",
        transform=ax_hist.transAxes, ha="right", va="top", fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "0.75", "linewidth": 0.5})

    legend = [Line2D([], [], marker="o", linestyle="", markersize=4.2,
                     markerfacecolor=COMMUNITY_COLORS[index],
                     markeredgecolor="0.25", markeredgewidth=0.25,
                     label=SYSTEM_NAMES[index]) for index in range(13)]
    fig.legend(handles=legend, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, 0.003), columnspacing=0.85,
               handletextpad=0.25, fontsize=6.2)
    fig.suptitle(
        f"Union of 324 stimulus-rooted dominant propagation subnetworks\n"
        f"{group}, subject {subject:03d}", y=0.988, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.925, bottom=0.095)
    prefix = output_dir / "Figure_union_propagation_network"
    fig.savefig(f"{prefix}.jpg", dpi=300, facecolor="white")
    fig.savefig(f"{prefix}.png", dpi=300, facecolor="white")
    fig.savefig(f"{prefix}.pdf", facecolor="white")
    fig.savefig(f"{prefix}.svg", facecolor="white")
    fig.savefig(f"{prefix}.tiff", dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def process_subject(dynamics_path: Path, dataset_path: Path,
                    output_dir: Path, make_figure: bool = True) -> dict:
    parent, reachable, template, xyz, group, subject = load_and_validate(
        dynamics_path, dataset_path)
    counts = union_edge_counts(parent, reachable)
    denominator = parent.shape[0]
    save_tables(output_dir, counts, denominator, template, xyz)
    values = np.asarray(list(counts.values()), dtype=int)
    summary = {
        "group": group,
        "subject_index": subject,
        "stimulus_experiments": denominator,
        "subnetwork_edge_instances": int(values.sum()),
        "union_directed_edges": int(len(values)),
        "edges_occurring_once": int(np.sum(values == 1)),
        "edges_occurring_at_least_twice": int(np.sum(values >= 2)),
        "edges_occurring_at_least_three_times": int(np.sum(values >= 3)),
        "maximum_occurrence_count": int(values.max()),
        "maximum_occurrence_proportion": float(values.max() / denominator),
        "median_occurrence_count": float(np.median(values)),
        "weight_definition": "occurrence_count / 324 completed stimuli",
        "subnetwork_definition": (
            "stimulus-reachable edges only; root excluded from reachable mask"),
        "node_color_definition": "Figure 3 functional-system palette",
        "random_seed": BASE_SEED,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "union_propagation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if make_figure:
        draw_figure(output_dir, counts, denominator, template, xyz,
                    group, subject)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    summary = process_subject(
        args.dynamics, args.dataset, args.output_dir,
        make_figure=not args.no_figure)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
