# Cell 19: Single-color ROI views with axes and saved held-out fit errors.
# No fitting, shortest paths, curvature search, registration, or FSL calls.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROJECT = "/mnt/g/Research/TUM_connectome/HyperbolicTest"
ROI_ID, GRAPH, SPLIT = 1, "supported", 1807
NODE_COLOR = "#2475A9"


def save_visible19(fig, stem):
    # Include 3D axis labels explicitly; fixed figure bounds can clip them.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [fig.bbox]
    for ax in fig.axes:
        boxes.append(ax.get_tightbbox(renderer))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            # Axes3D tight bounds include projected ticks. Reading tick labels
            # again here can reset their positions to unprojected data values.
            for artist in (axis.label, axis.offsetText):
                if artist.get_visible() and artist.get_text():
                    boxes.append(artist.get_window_extent(renderer))
    boxes += [text.get_window_extent(renderer) for text in fig.texts]
    bounds = Bbox.union([box for box in boxes if box is not None])
    bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(.16)
    for suffix, dpi in ((".png", 200), ("_600dpi.png", 600), (".pdf", 200)):
        fig.savefig(str(stem)+suffix, dpi=dpi, bbox_inches=bounds, facecolor="white")


def make_cell19(project=PROJECT, roi_id=ROI_ID, graph=GRAPH, split=SPLIT):
    src = Path(project) / f"results/cell18_roi_{roi_id:03d}_{graph}"
    out = Path(project) / f"results/cell19_roi_{roi_id:03d}_{graph}"
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
    score_table = scores.set_index("model")
    errors = score_table.loc[["euclidean", "hyperbolic"], "test_nRMSE"].to_numpy(float)
    scales = score_table.loc[["euclidean", "hyperbolic"], "train_scale_mm"].to_numpy(float)
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Missing or invalid held-out nRMSE values.")
    if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.allclose(scales, scales[0]):
        raise ValueError("E/H errors must use the same positive normalization scale.")
    captions = ("Anatomical XYZ",
                rf"Euclidean $E^3$ fit | nRMSE = {errors[0]:.4f}",
                rf"Hyperbolic $H^3$ fit | nRMSE = {errors[1]:.4f}")
    pairs = pd.read_csv(src / "display_pairs.csv")
    pairs = pairs[pairs.split_seed == split]
    a, b = pairs.landmark_a.to_numpy(int), pairs.landmark_b.to_numpy(int)
    if not len(a) or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= len(lm):
        raise ValueError("Invalid saved display pairs.")
    membership = pd.read_csv(src / "roi_node_membership.csv")
    context = membership[list(columns[0])].to_numpy(float)
    if not len(context) or not np.isfinite(context).all():
        raise ValueError("Missing or invalid anatomical context.")
    cloud = context[::max(1, int(np.ceil(len(context) / 3000)))]
    coverage = pd.read_csv(src / "roi_coverage.csv").iloc[0]
    sampling = pd.DataFrame([
        dict(reference=name, landmarks=len(lm), denominator=int(coverage[key]),
             sampled_percent=100*len(lm)/int(coverage[key]))
        for name, key in (("ROI largest-component voxels", "largest_voxels"),
                          ("ROI largest-component nodes", "largest_nodes"),
                          ("All ROI graph nodes", "roi_nodes"))
    ])

    # Reconstruct display geodesics from saved unit-ball coordinates.
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

    style = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
             "svg.fonttype": "none", "figure.autolayout": False,
             "figure.constrained_layout.use": False, "savefig.bbox": None,
             "axes.labelcolor": "black", "xtick.color": "black",
             "ytick.color": "black", "text.color": "black"}
    out.mkdir(parents=True, exist_ok=True)
    figures = {}
    with plt.rc_context(style):
        for layout in ("row", "column"):
            width, height = (1904, 626) if layout == "row" else (626, 1878)
            centers = tuple((i+.5)*width/3 for i in range(3)) if layout == "row" else (313.0,) * 3
            fig = plt.figure(figsize=(width/200, height/200), dpi=200, facecolor="white")
            side = .70 * 626
            for col, points in enumerate((xyz, E, P)):
                bottom = 0 if layout == "row" else (2-col) * 626
                # Reserve extra space on the right for Z labels and ticks.
                ax = fig.add_axes([(centers[col]-.04*626-side/2)/width, (bottom+.24*626)/height,
                                   side/width, side/height], projection="3d", facecolor="white")
                fig.text(centers[col]/width, (bottom+.045*626)/height, captions[col],
                         ha="center", va="center", fontsize=8.5)
                segments = arcs if col == 2 else np.stack((points[a], points[b]), axis=1)
                ax.add_collection3d(Line3DCollection(segments, colors="#64748b",
                                                     linewidths=.45, alpha=.27))
                if col == 0:
                    ax.scatter(*cloud.T, color=NODE_COLOR, s=1.7, alpha=.12,
                               depthshade=False, linewidths=0)
                if col == 2:
                    ax.plot_wireframe(*sphere, color="#b7c1cc", linewidth=.3, alpha=.36)
                ax.scatter(*points.T, color=NODE_COLOR, s=16, edgecolors="white",
                           linewidths=.25, depthshade=False)
                extent = context if col == 0 else points
                center = (extent.min(0) + extent.max(0)) / 2
                radius = max(float(np.ptp(extent, axis=0).max()) * .56, 1e-6)
                low, high = ((-np.ones(3), np.ones(3)) if col == 2 else
                             (center-radius, center+radius))
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
                ax.set_box_aspect((1, 1, 1))
                ax.set_proj_type("ortho")
                ax.view_init(elev=20, azim=-55)
                ax.grid(False)
                labels = ("X (mm)", "Y (mm)", "Z (mm)") if col < 2 else (r"$p_x$", r"$p_y$", r"$p_z$")
                for dim, (axis, label) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), labels)):
                    axis.set_label_text(label, fontsize=8)
                    axis.labelpad = 5
                    axis.pane.fill = False
                    axis.pane.set_edgecolor("#C3C8CF")
                    axis.pane.set_linewidth(.6)
                    axis.pane.set_alpha(.8)
                    axis.line.set_color("#777777")
                    axis.line.set_linewidth(.5)
                    ticks = (np.array([-1, 0, 1]) if col == 2 else
                             MaxNLocator(nbins=3).tick_values(low[dim], high[dim]))
                    axis.set_ticks(ticks[(ticks >= low[dim]) & (ticks <= high[dim])])
                ax.tick_params(labelsize=6.5, pad=0)

            stem = out / f"roi_xyz_E3_H3_{layout}_labeled"
            save_visible19(fig, stem)
            figures[layout] = fig
    info = dict(roi_id=int(roi_id), graph=graph, split_seed=int(split),
                source=str(src), captions=list(captions), node_color=NODE_COLOR,
                colors="Single blue for all landmarks; no scalar color mapping or colorbar",
                anatomical_context="Strided display of all ROI nodes, including other components and possible landmark overlap",
                context_points_drawn=len(cloud), context_nodes_available=len(context),
                sampling=sampling.to_dict(orient="records"),
                sampling_rule="Uniform distinct voxels in the largest ROI component; one eligible orientation node per selected voxel",
                frame="Unfilled panes with light-gray edges, as in Cell 18b",
                export="Explicit label-inclusive bounds plus padding",
                anatomical_units="fdt_paths reference world coordinates in mm",
                euclidean_units="Graph-distance mm; embedding axes are not anatomical axes",
                ball_units="Dimensionless Cartesian coordinates inside the Poincare unit ball",
                error_metric="Saved held-out RMSE divided by the fitting-pair median graph distance",
                euclidean_test_nRMSE=float(errors[0]), hyperbolic_test_nRMSE=float(errors[1]),
                train_scale_mm=float(scales[0]),
                lines="Saved landmark-pair chords in XYZ/E; model geodesics in H; not fiber trajectories",
                selection="Reuse saved split and display pairs; no score-based selection",
                layouts=["row", "column"], new_fitting=False)
    (out / "display_notes.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    sampling.to_csv(out / "landmark_sampling_summary.csv", index=False)
    plt.show()
    print(f"Saved: {out}")
    print("Row and column layouts: preview PNG, 600-dpi PNG, and vector PDF.")
    print(f"Split {split}: held-out E3 nRMSE = {errors[0]:.4f}; H3 nRMSE = {errors[1]:.4f}.")
    print("Anatomical/E3 axes: mm. Poincare ball axes: dimensionless p_x, p_y, p_z.")
    print("\nLandmark sampling (voxel-based selection; node ratios are descriptive):")
    print(sampling.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Faint anatomical context: {len(cloud):,} displayed / {len(context):,} ROI graph nodes.")
    print("Routing used the saved full ROI graph, including non-landmark intermediate nodes.")
    return figures


if __name__ == "__main__":
    figures19 = make_cell19()
