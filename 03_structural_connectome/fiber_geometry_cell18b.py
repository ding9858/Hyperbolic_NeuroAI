# Cell 18b: redraw the saved ROI comparison with XYZ axes and numeric ticks.
# Read existing coordinates and displayed pairs; no fitting or shortest paths.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROJECT = "/mnt/g/Research/TUM_connectome/HyperbolicTest"
ROI_ID, GRAPH = 1, "supported"


def bounds18b(points):
    low, high = points.min(0), points.max(0)
    center, half = (low+high)/2, max(float(np.max(high-low))*.58, 1e-6)
    return center-half, center+half


def arcs18b(P, a, b):
    norm2 = np.sum(P*P, axis=1)
    if np.any(norm2 >= 1):
        raise ValueError("Saved ball coordinates must lie inside the unit sphere.")
    L = np.column_stack((1+norm2, 2*P))/(1-norm2[:, None])
    t = np.linspace(0, 1, 48)[None, :, None]
    Q = (1-t)*L[a, None, :] + t*L[b, None, :]
    norm = np.sqrt(Q[..., 0]**2-np.sum(Q[..., 1:]**2, axis=-1))
    return Q[..., 1:]/(Q[..., 0]+norm)[..., None]


def redraw_cell18b(project, roi_id=1, graph="supported"):
    src = Path(project) / f"results/cell18_roi_{roi_id:03d}_{graph}"
    out = src / "with_axes"
    protocol = json.loads((src / "protocol.json").read_text())
    seeds = tuple(protocol["split_seeds"])
    coordinates = pd.read_csv(src / "embedding_coordinates.csv")
    landmarks = pd.read_csv(src / "landmarks.csv")
    membership = pd.read_csv(src / "roi_node_membership.csv")
    pairs = pd.read_csv(src / "display_pairs.csv")
    scores = pd.read_csv(src / "fit_results.csv")
    coverage = pd.read_csv(src / "roi_coverage.csv").iloc[0]
    columns = (("fdt_x_mm", "fdt_y_mm", "fdt_z_mm"),
               ("E_x_mm", "E_y_mm", "E_z_mm"),
               ("H_ball_x", "H_ball_y", "H_ball_z"))
    anatomical = membership[list(columns[0])].to_numpy()
    limits = (bounds18b(anatomical), bounds18b(coordinates[list(columns[1])].to_numpy()),
              (-np.ones(3), np.ones(3)))
    color_norm = Normalize(landmarks.fdt_z_mm.min(), landmarks.fdt_z_mm.max())
    cmap = plt.get_cmap("viridis")
    context_xyz = anatomical[::max(1, int(np.ceil(len(anatomical)/3000)))]
    u, v = np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
              np.outer(np.ones_like(u), np.cos(v)))
    fig = plt.figure(figsize=(14, 4.6*len(seeds)+.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=.12, h_pad=.15, hspace=.10, wspace=.06)
    axes = []
    for row, seed in enumerate(seeds):
        part = coordinates[coordinates.split_seed == seed].set_index("base_node_id").loc[landmarks.base_node_id]
        selected = pairs[pairs.split_seed == seed]
        a, b = selected.landmark_a.to_numpy(int), selected.landmark_b.to_numpy(int)
        xyz, E, P = [part[list(c)].to_numpy() for c in columns]
        for col, points in enumerate((xyz, E, P)):
            ax = fig.add_subplot(len(seeds), 3, row*3+col+1, projection="3d")
            axes.append(ax)
            curves = arcs18b(P, a, b) if col == 2 else np.stack((points[a], points[b]), axis=1)
            ax.add_collection3d(Line3DCollection(curves, colors="#64748b", linewidths=.55, alpha=.23))
            ax.scatter(*points.T, c=xyz[:, 2], cmap=cmap, norm=color_norm, s=20,
                       edgecolors="white", linewidths=.3, depthshade=False)
            if col == 0:
                ax.scatter(*context_xyz.T, c="0.65", s=2, alpha=.13, depthshade=False)
                title, labels = "ROI anatomical XYZ", ("X (mm)", "Y (mm)", "Z (mm)")
            else:
                model = ("euclidean", "hyperbolic")[col-1]
                score = scores[(scores.split_seed == seed) & (scores.model == model)].iloc[0]
                title = ("E3" if col == 1 else f"H3 (k={score.curvature_scaled:g})")
                title += f" | test nRMSE={score.test_nRMSE:.4f}"
                if str(score.fit_ok).lower() not in ("true", "1"):
                    title += " | FIT FLAGGED"
                labels = (("Latent x (mm)", "Latent y (mm)", "Latent z (mm)") if col == 1
                          else ("Ball x", "Ball y", "Ball z"))
            if col == 2:
                ax.plot_wireframe(*sphere, color="0.65", linewidth=.35, alpha=.18)
            low, high = limits[col]
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
            ax.set_title(f"{title}\nPair split {seed}", fontsize=11, pad=12)
            for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), labels):
                axis.set_label_text(label, fontsize=10)
                axis.labelpad = 6
                axis.pane.fill = False
                axis.set_major_locator(MaxNLocator(nbins=3))
            if col == 2:
                ax.set_xticks((-1, 0, 1))
                ax.set_yticks((-1, 0, 1))
                ax.set_zticks((-1, 0, 1))
            ax.tick_params(labelsize=8, pad=1)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-55)
            ax.set_proj_type("ortho")
            ax.grid(False)
    fig.colorbar(ScalarMappable(norm=color_norm, cmap=cmap), ax=axes, shrink=.6, pad=.015,
                 label="Anatomical z in fdt_paths world space (mm)")
    fig.suptitle(f"ROI {roi_id} | {graph} | {len(landmarks)} landmarks\n"
        f"Within-ROI component: {int(coverage.largest_nodes)}/{int(coverage.roi_nodes)} nodes. "
        "Displayed lines are pair geometry, not fiber trajectories.", fontsize=12)
    out.mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(out / "roi_xyz_E3_H3_axes.png", dpi=240, bbox_inches="tight", pad_inches=.15)
        fig.savefig(out / "roi_xyz_E3_H3_axes.pdf", bbox_inches="tight", pad_inches=.15)
    coordinates.to_csv(out / "point_coordinates.csv", index=False)
    (out / "coordinate_units.txt").write_text(
        "fdt_x_mm / fdt_y_mm / fdt_z_mm: anatomical coordinates in the saved fdt_paths world space, mm.\n"
        "E_x_mm / E_y_mm / E_z_mm: fitted Euclidean latent coordinates in graph-distance mm; not anatomical axes.\n"
        "H_ball_x / H_ball_y / H_ball_z: dimensionless coordinates inside the unit display ball.\n"
        "Latent orientation is arbitrary. Euclidean chord length inside the ball is not hyperbolic distance.\n"
        "Each column uses equal spatial scaling along its three axes and fixed limits across pair splits.\n"
        "Saved coordinates, node pairs, and model scores are reused without refitting.\n", encoding="utf-8")
    plt.show()
    print("Redrawn with labeled XYZ axes and numerical ticks. No fitting was run.")
    print(f"Coordinates: {out / 'point_coordinates.csv'}")
    print(f"Figures: {out / 'roi_xyz_E3_H3_axes.png'}")
    print(f"         {out / 'roi_xyz_E3_H3_axes.pdf'}")
    return fig


if __name__ == "__main__":
    figure18b = redraw_cell18b(PROJECT, ROI_ID, GRAPH)
