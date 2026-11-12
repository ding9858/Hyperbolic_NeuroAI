# Cell 18c: one aligned row from saved ROI coordinates; no refitting.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROJECT = "/mnt/g/Research/TUM_connectome/HyperbolicTest"
ROI_ID, GRAPH, SPLIT = 1, "supported", 1807
WIDTH, HEIGHT = 1904, 626
# Horizontal brain centers measured from the supplied T1_ROI1.png.
CENTERS = (360.0, 1052.5, 1611.5)


def make_row18c(project=PROJECT, roi_id=ROI_ID, graph=GRAPH, split=SPLIT):
    src = Path(project) / f"results/cell18_roi_{roi_id:03d}_{graph}"
    lm = pd.read_csv(src / "landmarks.csv").sort_values("landmark_id")
    coords = pd.read_csv(src / "embedding_coordinates.csv")
    part = coords[coords.split_seed == split].set_index("base_node_id")
    if part.empty or not part.index.is_unique or not lm.base_node_id.is_unique:
        raise ValueError("Missing split or duplicate landmark IDs.")
    part = part.loc[lm.base_node_id]
    columns = [("fdt_x_mm", "fdt_y_mm", "fdt_z_mm"),
               ("E_x_mm", "E_y_mm", "E_z_mm"),
               ("H_ball_x", "H_ball_y", "H_ball_z")]
    xyz, E, P = [part[list(c)].to_numpy(float) for c in columns]
    if not all(np.isfinite(q).all() for q in (xyz, E, P)):
        raise ValueError("Coordinates contain nonfinite values.")
    if not np.array_equal(lm.landmark_id, np.arange(len(lm))):
        raise ValueError("Expected contiguous, zero-based landmark IDs.")
    if not np.allclose(xyz, lm[list(columns[0])].to_numpy()):
        raise ValueError("Saved coordinate tables disagree.")
    scores = pd.read_csv(src / "fit_results.csv")
    scores = scores[(scores.split_seed == split) & scores.model.isin(["euclidean", "hyperbolic"])]
    if (len(scores) != 2 or set(scores.model) != {"euclidean", "hyperbolic"}
            or not scores.fit_ok.astype(str).str.lower().isin(["true", "1"]).all()):
        raise ValueError("The selected E/H fits are missing or flagged.")
    pairs = pd.read_csv(src / "display_pairs.csv")
    pairs = pairs[pairs.split_seed == split]
    a, b = pairs.landmark_a.to_numpy(int), pairs.landmark_b.to_numpy(int)
    if not len(a) or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= len(lm):
        raise ValueError("Invalid saved display pairs.")
    context = pd.read_csv(src / "roi_node_membership.csv")[list(columns[0])].to_numpy()
    cloud = context[::max(1, int(np.ceil(len(context) / 3000)))]

    # Unit-ball geodesics, using the already-saved hyperbolic fit.
    norm2 = np.sum(P * P, axis=1)
    if np.any(norm2 >= 1):
        raise ValueError("A saved point lies outside the unit ball.")
    L = np.column_stack((1 + norm2, 2 * P)) / (1 - norm2[:, None])
    t = np.linspace(0, 1, 64)[None, :, None]
    Q = (1 - t) * L[a, None, :] + t * L[b, None, :]
    norm = np.sqrt(Q[..., 0]**2 - np.sum(Q[..., 1:]**2, axis=-1))
    arcs = Q[..., 1:] / (Q[..., 0] + norm)[..., None]
    theta, phi = np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (np.outer(np.cos(theta), np.sin(phi)),
              np.outer(np.sin(theta), np.sin(phi)),
              np.outer(np.ones_like(theta), np.cos(phi)))
    colors = plt.get_cmap("viridis")(Normalize(xyz[:, 2].min(), xyz[:, 2].max())(xyz[:, 2]))

    style = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
             "figure.autolayout": False, "figure.constrained_layout.use": False,
             "savefig.bbox": None, "axes.labelcolor": "black",
             "xtick.color": "black", "ytick.color": "black", "text.color": "black"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(WIDTH/200, HEIGHT/200), dpi=200, facecolor="white")
        side = .80 * HEIGHT
        axes = []
        for col, points in enumerate((xyz, E, P)):
            ax = fig.add_axes([(CENTERS[col]-side/2)/WIDTH, .13,
                               side/WIDTH, side/HEIGHT], projection="3d", facecolor="white")
            axes.append(ax)
            segments = arcs if col == 2 else np.stack((points[a], points[b]), axis=1)
            ax.add_collection3d(Line3DCollection(segments, colors="#64748b", linewidths=.45, alpha=.27))
            if col == 0:
                ax.scatter(*cloud.T, color="#aab0b8", s=1.7, alpha=.18, depthshade=False, linewidths=0)
            if col == 2:
                ax.plot_wireframe(*sphere, color="#b7c1cc", linewidth=.3, alpha=.36)
            ax.scatter(*points.T, c=colors, s=16, edgecolors="white", linewidths=.25, depthshade=False)
            extent = context if col == 0 else points
            center = (extent.min(0) + extent.max(0)) / 2
            radius = max(float(np.ptp(extent, axis=0).max()) * .56, 1e-6)
            low, high = ((-np.ones(3), np.ones(3)) if col == 2 else (center-radius, center+radius))
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
            ax.set_box_aspect((1, 1, 1))
            ax.set_proj_type("ortho")
            ax.view_init(elev=20, azim=-55)
            ax.grid(False)
            if col == 0:
                ax.set_axis_off()
            else:
                labels = ("X", "Y", "Z") if col == 1 else ("Ball x", "Ball y", "Ball z")
                for dim, (axis, label) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), labels)):
                    axis.set_label_text(label, fontsize=8)
                    axis.labelpad = -8 if dim == 2 else 2
                    axis.pane.set_facecolor((1, 1, 1, 0))
                    axis.pane.set_edgecolor((1, 1, 1, 0))
                    axis.line.set_color("#777777")
                    axis.line.set_linewidth(.5)
                    ticks = (np.array([-1, 0, 1]) if col == 2 else
                             MaxNLocator(nbins=3).tick_values(low[dim], high[dim]))
                    axis.set_ticks(ticks[(ticks >= low[dim]) & (ticks <= high[dim])])
                ax.tick_params(labelsize=6.5, pad=0)

        out = src / "figure_row"
        out.mkdir(exist_ok=True)
        stem = out / "roi_xyz_E3_H3_row"
        # Never use tight_layout or bbox_inches='tight': preserve the row geometry.
        fig.savefig(str(stem)+".png", dpi=200, bbox_inches=None, facecolor="white")
        fig.savefig(str(stem)+"_600dpi.png", dpi=600, bbox_inches=None, facecolor="white")
        fig.savefig(str(stem)+".pdf", bbox_inches=None, facecolor="white")
        info = dict(roi_id=int(roi_id), graph=graph, split_seed=int(split),
                    canvas_pixels=[WIDTH, HEIGHT], column_centers_px=list(CENTERS),
                    selection="Fixed split 1807 by default; no score-based selection",
                    colors="Anatomical z in fdt world coordinates; same colors in all panels",
                    euclidean_units="Graph-distance mm; latent axes are not anatomical axes",
                    ball_units="Dimensionless Cartesian coordinates inside the Poincare unit ball",
                    lines="Saved landmark-pair chords in XYZ/E; model geodesics in H; not fiber trajectories",
                    alignment="Panel layout only; no new registration or embedding alignment")
        (out / "layout_and_caption_notes.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    plt.show()
    print(f"Saved: {out}")
    return fig


if __name__ == "__main__":
    figure18c = make_row18c()
