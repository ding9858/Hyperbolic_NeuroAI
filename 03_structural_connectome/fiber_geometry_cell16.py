# Cell 16: summarize existing evidence and display fitted H3 landmark panels.
# No fitting, curvature search, shortest paths, registration, or FSL.
# Ball arcs are model geodesics between selected fitting pairs, not fiber paths.
# Each panel contains only its saved landmarks, not every whole-brain graph node.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def ball16(Z, curvature):
    radius = 1 / np.sqrt(abs(curvature))
    time = np.sqrt(radius**2 + np.sum(Z*Z, axis=1))
    lorentz = np.column_stack((time, Z))
    return Z / (time[:, None] + radius), lorentz


def ball_dist16(P, curvature, i, j):
    radius = 1 / np.sqrt(abs(curvature))
    norm2 = np.sum(P*P, axis=1)
    gap = (1 - norm2[i]) * (1 - norm2[j])
    if np.any(gap <= 0):
        raise ValueError("Poincare points must lie strictly inside the unit display ball.")
    return 2*radius*np.arcsinh(np.linalg.norm(P[i]-P[j], axis=1)/np.sqrt(gap))


def ball_arcs16(lorentz, a, b, points=48):
    # Normalized Lorentz chords trace geodesics; sampling is not constant speed.
    t = np.linspace(0, 1, points)[None, :, None]
    Q = (1-t)*lorentz[a, None, :] + t*lorentz[b, None, :]
    norm = np.sqrt(Q[..., 0]**2 - np.sum(Q[..., 1:]**2, axis=-1))
    return Q[..., 1:] / (Q[..., 0] + norm)[..., None]


def bool16(series):
    return series.astype(str).str.lower().isin(("true", "1"))


def run_cell16(project):
    root = Path(project) / "results"
    matched, controls = root / "cell13_matched_dimensions", root / "cell15_control_geometry"
    out = root / "cell16_evidence_and_ball"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((controls / "protocol.json").read_text())
    seeds, null_seeds = tuple(protocol["panel_seeds"]), tuple(protocol["null_seeds"])
    names, dimensions = ("supported", "cone45"), (3, 5)
    dim_rows = pd.read_csv(matched / "matched_results.csv")
    dim_rows = dim_rows[dim_rows.dimension.isin(dimensions)].copy()
    gains = pd.read_csv(controls / "panel_gains.csv")
    contrasts = pd.read_csv(controls / "original_control_contrasts.csv")
    fits = pd.read_csv(controls / "fit_results.csv")
    if not (bool16(dim_rows.selected_fits_ok).all() and bool16(dim_rows.all_candidates_ok).all() and
            bool16(gains.fit_ok).all() and bool16(contrasts.fit_ok).all()):
        raise ValueError("Saved results contain failed fits; inspect their flags before preparing figures.")
    summary = dim_rows.groupby(["metric", "dimension"], sort=False).agg(
        panels=("panel_seed", "nunique"), mean_E_test_nRMSE=("E_test_nRMSE", "mean"),
        mean_H_test_nRMSE=("H_test_nRMSE", "mean"), mean_H_gain=("E_minus_H", "mean"),
        mean_relative_H_reduction=("relative_H_reduction", "mean")).reset_index()
    summary.to_csv(out / "matched_dimension_summary.csv", index=False)
    dim_rows.to_csv(out / "matched_dimension_panel_data.csv", index=False)
    contrasts.to_csv(out / "control_contrast_panel_data.csv", index=False)
    control_rows = []
    for graph in names:
        part = gains[gains.graph == graph]
        original = part[part.variant == "original"].H_gain.mean()
        null_means = part[part.variant != "original"].groupby("variant").H_gain.mean()
        paired = contrasts[contrasts.graph == graph]
        control_rows.append(dict(graph=graph, original_mean_H_gain=original,
                                 control_mean_H_gain=null_means.mean(),
                                 min_control_mean_H_gain=null_means.min(),
                                 max_control_mean_H_gain=null_means.max(),
                                 original_mean_within_control_mean_range=bool(null_means.min() <= original <= null_means.max()),
                                 mean_paired_original_minus_control=paired.original_minus_control.mean(),
                                 positive_paired_contrasts=int((paired.original_minus_control > 0).sum()),
                                 total_paired_contrasts=len(paired)))
    control_summary = pd.DataFrame(control_rows)
    control_summary.to_csv(out / "control_contrast_summary.csv", index=False)

    # Display all panels. No confidence intervals or hypothesis tests are added.
    colors = plt.cm.tab10(np.arange(len(seeds)))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), constrained_layout=True)
    for row, graph in enumerate(names):
        for color, seed in zip(colors, seeds):
            for suffix, style in (("graph", "o-"), ("spatial", "s--")):
                part = dim_rows[(dim_rows.metric == graph + "_" + suffix) &
                                (dim_rows.panel_seed == seed)].set_index("dimension").loc[list(dimensions)]
                axes[row, 0].plot(dimensions, part.E_minus_H, style, color=color,
                                   alpha=.85 if suffix == "graph" else .45,
                                   label=f"{seed} {suffix}")
            part = contrasts[(contrasts.graph == graph) & (contrasts.panel_seed == seed)].set_index("control_seed").loc[list(null_seeds)]
            axes[row, 1].plot(range(len(null_seeds)), part.original_minus_control, "o-", color=color, label=f"panel {seed}")
        axes[row, 0].set(title=f"{graph}: matched-dimension comparison", xlabel="Embedding dimension",
                         ylabel="E - H test nRMSE", xticks=dimensions)
        axes[row, 1].set(title=f"{graph}: fixed H3 control comparison", xlabel="Rewiring seed",
                         ylabel="Original H gain - control H gain", xticks=range(len(null_seeds)),
                         xticklabels=[str(s) for s in null_seeds])
        for ax in axes[row]:
            ax.axhline(0, color="0.5", ls=":", lw=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7, ncol=2 if ax is axes[row, 0] else 1)
    fig.suptitle("One-brain exploratory evidence | left: Cell 13 selected curvature | right: Cell 15 fixed k=-4")
    fig.savefig(out / "evidence_summary.png", dpi=220)
    fig.savefig(out / "evidence_summary.pdf")
    plt.show()

    with np.load(controls / "pair_split.npz") as z:
        i, j, train, test = [z[k].astype(int) for k in ("i", "j", "train", "test")]
    chosen = np.random.default_rng(1601).choice(train, min(24, len(train)), replace=False)
    pd.DataFrame(dict(pair_index=chosen, landmark_a=i[chosen], landmark_b=j[chosen],
                      role="Illustrative fitting-pair geodesic; not an observed edge")).to_csv(out / "display_pairs.csv", index=False)
    # Reuse the already established display-space transform; do not inspect images.
    with np.load(root / "cell14c_fdt_space/coordinate_transforms.npz") as z:
        world_map = z["dwi_world_to_fdt_world"].copy()
    records, point_tables, conversion_checks = {}, [], []
    for graph in names:
        with np.load(root / f"cell04_orientation_graph/nodes_{graph}.npz") as z:
            node_ids, native_xyz = z["base_node_id"].copy(), z["world_xyz_mm"].copy()
        for seed in seeds:
            path = controls / f"fit_{graph}_original_{seed}_hyperbolic.npz"
            with np.load(path) as z:
                Z, ids = z["coordinates"].astype(float), z["base_node_id"].copy()
                k, scale = float(z["curvature_scaled"]), float(z["train_scale_mm"])
            if Z.shape != (len(ids), 3) or k >= 0:
                raise ValueError("Expected a saved three-dimensional hyperbolic fit.")
            index = pd.Index(node_ids).get_indexer(ids)
            if np.any(index < 0):
                raise ValueError("Saved embedding contains unknown graph node IDs.")
            fdt_xyz = native_xyz[index] @ world_map[:3, :3].T + world_map[:3, 3]
            P, L = ball16(Z, k)
            if np.any(np.linalg.norm(P, axis=1) >= 1):
                raise ValueError("Invalid unit-ball projection.")
            radius = 1/np.sqrt(abs(k))
            lorentz_distance = radius*np.arccosh(np.maximum((L[i, 0]*L[j, 0]-(Z[i]*Z[j]).sum(1))/radius**2, 1))
            ball_distance = ball_dist16(P, k, i, j)
            conversion_error = float(np.max(np.abs(lorentz_distance-ball_distance)))
            if conversion_error > 1e-6:
                raise ValueError("Lorentz-to-ball projection changed distances beyond tolerance.")
            with np.load(controls / f"distances_{graph}_original_{seed}.npz") as z:
                if not np.array_equal(ids, z["base_node_id"]):
                    raise ValueError("Embedding and distance-matrix node ordering differ.")
                observed = z["distance_mm"][i[test], j[test]]
            recovered_score = float(np.sqrt(np.mean((ball_distance[test]*scale-observed)**2))/scale)
            saved = fits[(fits.graph == graph) & (fits.variant == "original") &
                         (fits.panel_seed == seed) & (fits.model == "hyperbolic")]
            if len(saved) != 1 or not np.isclose(recovered_score, saved.test_nRMSE.iloc[0], atol=1e-7, rtol=0):
                raise ValueError("Ball distances do not reproduce the saved held-out score.")
            curves = ball_arcs16(L, i[chosen], j[chosen])
            table = pd.DataFrame(dict(graph=graph, panel_seed=seed, landmark_id=np.arange(len(ids)), base_node_id=ids,
                                      fdt_x_mm=fdt_xyz[:, 0], fdt_y_mm=fdt_xyz[:, 1], fdt_z_mm=fdt_xyz[:, 2],
                                      ball_x=P[:, 0], ball_y=P[:, 1], ball_z=P[:, 2],
                                      ball_radius=np.linalg.norm(P, axis=1), curvature_scaled=k,
                                      train_scale_mm=scale))
            table.to_csv(out / f"landmarks_{graph}_{seed}_poincare.csv", index=False)
            point_tables.append(table)
            np.savez_compressed(out / f"poincare_{graph}_{seed}.npz", base_node_id=ids,
                                ball_coordinates=P, lorentz_coordinates=L, curvature_scaled=k,
                                train_scale_mm=scale, fdt_world_xyz_mm=fdt_xyz,
                                display_pair_indices=chosen, display_geodesics=curves)
            records[graph, seed] = dict(points=P, curves=curves, fdt=fdt_xyz, k=k)
            conversion_checks.append(dict(graph=graph, panel_seed=seed, landmarks=len(ids),
                                          max_ball_radius=float(np.linalg.norm(P, axis=1).max()),
                                          distance_conversion_error_normalized=conversion_error,
                                          recovered_test_nRMSE=recovered_score))
    combined = pd.concat(point_tables, ignore_index=True)
    combined.to_csv(out / "all_landmark_coordinates.csv", index=False)
    checks = pd.DataFrame(conversion_checks)
    checks.to_csv(out / "ball_conversion_checks.csv", index=False)
    norm = Normalize(combined.fdt_z_mm.min(), combined.fdt_z_mm.max())
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(4.3*len(seeds), 8.1), constrained_layout=True)
    axes_ball = []
    u, v = np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
              np.outer(np.ones_like(u), np.cos(v)))
    for row, graph in enumerate(names):
        for col, seed in enumerate(seeds):
            ax = fig.add_subplot(2, len(seeds), row*len(seeds)+col+1, projection="3d")
            axes_ball.append(ax)
            rec = records[graph, seed]
            ax.plot_wireframe(*sphere, color="0.65", linewidth=.35, alpha=.22)
            ax.add_collection3d(Line3DCollection(rec["curves"], colors="#64748b", linewidths=.8, alpha=.5))
            P = rec["points"]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=rec["fdt"][:, 2], cmap=cmap, norm=norm,
                       s=23, edgecolors="white", linewidths=.3, depthshade=False)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                   title=f"{graph} | panel {seed}\n{len(P)} fitted landmarks; k={rec['k']:g}")
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-55)
            ax.set_axis_off()
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes_ball, shrink=.65,
                 label="Anatomical z in fdt_paths world space (mm)", pad=.01)
    fig.suptitle("Saved H3 fits in the Poincare ball\nArcs: sampled fitting-pair geodesics, not fiber trajectories; latent axes are arbitrary")
    fig.savefig(out / "poincare_landmark_panels.png", dpi=220)
    fig.savefig(out / "poincare_landmark_panels.pdf")
    plt.show()
    (out / "README.txt").write_text(
        "Existing results only; no new fitting or model selection.\n"
        "Evidence summary: matched dimensions 3/5 from Cell 13, fixed-curvature controls from Cell 15.\n"
        "Higher-dimensional sensitivity remains available in Cell 12/13; it was not rerun.\n"
        "Panels are landmark samples of one brain, not independent subjects. Reused test pairs are exploratory.\n"
        "Controls are partial rewiring with a protected forest; mixing is not established.\n"
        "Control curves evaluate the frozen k=-4 model, not each graph's optimal curvature.\n"
        "Poincare plots display saved landmark embeddings, not a full-node whole-brain embedding.\n"
        "The unit sphere is a display boundary at infinite hyperbolic distance, not an anatomical surface.\n"
        "Ball distances retain curvature and fitting-distance scale; Euclidean distances inside the ball are not graph distances.\n"
        "Drawn arcs represent selected fitting-pair model geodesics, not graph adjacency or recovered fiber paths.\n"
        "No 2D projection is labelled as a fitted hyperbolic disk.\n"
        "Lorentz/Poincare model reference: https://proceedings.mlr.press/v80/nickel18a.html\n",
        encoding="utf-8")
    (out / "protocol.json").write_text(json.dumps(dict(
        matched_source=str(matched), control_source=str(controls), dimensions_shown=dimensions,
        panels=seeds, ball_source="Cell 15 original-graph H3 fits", display_pair_seed=1601,
        display_pairs=int(len(chosen)), pair_selection="Uniform sample of fitting pairs, identical pair indices across panels",
        projection="Unit display ball p=z/(sqrt(R^2+|z|^2)+R), R=1/sqrt(-k)",
        anatomy="Saved Cell 14c DWI-world to fdt_paths-world transform; no new registration",
        interpretation="Representation advantage; biological specificity requires evidence beyond these descriptive controls"
    ), indent=2), encoding="utf-8")
    print("No new fitting, curvature search, shortest paths, or image checks.")
    print("\nMatched-dimension summary:\n" + summary.to_string(index=False))
    print("\nDescriptive control comparison (not a significance test):\n" + control_summary.to_string(index=False))
    print("\nBall projection checks:\n" + checks.to_string(index=False))
    print("Ball arcs are model geodesics; plotted nodes are saved landmarks only.")
    print(f"Saved: {out}")
    return summary, control_summary, checks


if __name__ == "__main__":
    summary16, controls16, ball_checks16 = run_cell16("/mnt/g/Research/TUM_connectome/HyperbolicTest")
