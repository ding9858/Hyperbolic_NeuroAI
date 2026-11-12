# Cell 17: anatomical XYZ, fitted E3, and fitted H3 side by side.
# Reuse saved fits; no tracking, shortest paths, curvature search, or FSL.
# Displayed lines join selected fitting pairs, not observed fiber trajectories.
# Increase N_LINKS for display only; the saved fits and scores stay unchanged.
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
N_LINKS = 96


def arcs17(L, a, b):
    t = np.linspace(0, 1, 48)[None, :, None]
    Q = (1-t)*L[a, None, :] + t*L[b, None, :]
    norm = np.sqrt(Q[..., 0]**2 - np.sum(Q[..., 1:]**2, axis=-1))
    return Q[..., 1:] / (Q[..., 0]+norm)[..., None]


def bounds17(points):
    lo, hi = points.min(0), points.max(0)
    center = (lo+hi)/2
    half = max(float(np.max(hi-lo))/2*1.13, 1e-6)
    return center-half, center+half


def run_cell17(project, n_links=96):
    root = Path(project) / "results"
    src, balls = root / "cell15_control_geometry", root / "cell16_evidence_and_ball"
    out = root / "cell17_xyz_euclidean_hyperbolic"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((src / "protocol.json").read_text())
    seeds, graphs = tuple(protocol["panel_seeds"]), ("supported", "cone45")
    fits = pd.read_csv(src / "fit_results.csv")
    with np.load(src / "pair_split.npz") as z:
        i, j, train, test = [z[k].astype(int) for k in ("i", "j", "train", "test")]
    old = pd.read_csv(balls / "display_pairs.csv").pair_index.to_numpy(dtype=int)
    if len(np.unique(old)) != len(old) or not np.isin(old, train).all():
        raise ValueError("Cell 16 display pairs must be distinct fitting pairs.")
    n_links = int(n_links)
    if n_links < len(old) or n_links > len(train):
        raise ValueError(f"N_LINKS must be between {len(old)} and {len(train)}.")
    extra = np.random.default_rng(1701).choice(np.setdiff1d(train, old), n_links-len(old), replace=False)
    chosen = np.r_[old, extra]
    a, b = i[chosen], j[chosen]
    pd.DataFrame(dict(pair_index=chosen, landmark_a=a, landmark_b=b,
                      already_displayed_in_cell16=np.isin(chosen, old), split="fitting")).to_csv(
                          out / "display_pairs.csv", index=False)
    records, scores, coordinates = {}, [], []
    for graph in graphs:
        for seed in seeds:
            with np.load(src / f"fit_{graph}_original_{seed}_euclidean.npz") as z:
                X, ids, scale = z["coordinates"].copy(), z["base_node_id"].copy(), float(z["train_scale_mm"])
            with np.load(balls / f"poincare_{graph}_{seed}.npz") as z:
                if not np.array_equal(ids, z["base_node_id"]):
                    raise ValueError("Euclidean and hyperbolic node ordering differs.")
                P, L, xyz = z["ball_coordinates"].copy(), z["lorentz_coordinates"].copy(), z["fdt_world_xyz_mm"].copy()
                k, hscale = float(z["curvature_scaled"]), float(z["train_scale_mm"])
            if X.shape != (len(ids), 3) or not np.isclose(scale, hscale) or k >= 0:
                raise ValueError("Expected matched three-dimensional saved fits.")
            if not np.array_equal(np.c_[i, j], np.column_stack(np.triu_indices(len(ids), 1))):
                raise ValueError("Pair indices do not match landmark count.")
            with np.load(src / f"distances_{graph}_original_{seed}.npz") as z:
                if not np.array_equal(ids, z["base_node_id"]):
                    raise ValueError("Distance-matrix node ordering differs.")
                target = z["distance_mm"][i[test], j[test]] / scale
            pnorm2 = np.sum(P*P, axis=1)
            if np.any(pnorm2 >= 1):
                raise ValueError("Invalid saved ball coordinates.")
            pred_e = np.linalg.norm(X[i[test]]-X[j[test]], axis=1)
            pred_h = 2/np.sqrt(-k)*np.arcsinh(
                np.linalg.norm(P[i[test]]-P[j[test]], axis=1) /
                np.sqrt((1-pnorm2[i[test]])*(1-pnorm2[j[test]])))
            values = {}
            for kind, prediction in (("euclidean", pred_e), ("hyperbolic", pred_h)):
                saved = fits[(fits.graph == graph) & (fits.variant == "original") &
                             (fits.panel_seed == seed) & (fits.model == kind)]
                if len(saved) != 1 or str(saved.fit_ok.iloc[0]).lower() not in ("true", "1"):
                    raise ValueError("Missing or failed saved model fit.")
                score = float(np.sqrt(np.mean((prediction-target)**2)))
                if not np.isclose(score, saved.test_nRMSE.iloc[0], atol=1e-7, rtol=0):
                    raise ValueError("Displayed coordinates do not reproduce the saved score.")
                values[kind] = score
                scores.append(dict(graph=graph, panel_seed=seed, model=kind, landmarks=len(ids),
                                   displayed_pairs=n_links, test_nRMSE=score, train_scale_mm=scale))
            # Euclidean coordinates are expressed in target-distance mm, not anatomical mm.
            E = X*scale
            records[graph, seed] = dict(xyz=xyz, E=E, P=P, curves=arcs17(L, a, b), scores=values, k=k)
            coordinates.append(pd.DataFrame(dict(graph=graph, panel_seed=seed,
                landmark_id=np.arange(len(ids)), base_node_id=ids,
                anatomical_x_mm=xyz[:, 0], anatomical_y_mm=xyz[:, 1], anatomical_z_mm=xyz[:, 2],
                E_x_mm=E[:, 0], E_y_mm=E[:, 1], E_z_mm=E[:, 2],
                H_ball_x=P[:, 0], H_ball_y=P[:, 1], H_ball_z=P[:, 2],
                curvature_scaled=k, train_scale_mm=scale)))
    table, scores = pd.concat(coordinates, ignore_index=True), pd.DataFrame(scores)
    table.to_csv(out / "landmark_coordinates.csv", index=False)
    scores.to_csv(out / "saved_model_scores.csv", index=False)
    color_norm = Normalize(table.anatomical_z_mm.min(), table.anatomical_z_mm.max())
    cmap = plt.get_cmap("viridis")
    anatomy_bounds = bounds17(table[["anatomical_x_mm", "anatomical_y_mm", "anatomical_z_mm"]].to_numpy())
    euclidean_bounds = bounds17(table[["E_x_mm", "E_y_mm", "E_z_mm"]].to_numpy())
    u, v = np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
              np.outer(np.ones_like(u), np.cos(v)))
    for graph in graphs:
        fig = plt.figure(figsize=(13.4, 4.0*len(seeds)+.7), constrained_layout=True)
        axes = []
        for row, seed in enumerate(seeds):
            rec = records[graph, seed]
            locations = (rec["xyz"], rec["E"], rec["P"])
            titles = ("Anatomical XYZ (fdt space)",
                      f"Euclidean E3 fit | test nRMSE={rec['scores']['euclidean']:.4f}",
                      f"Hyperbolic H3 fit | test nRMSE={rec['scores']['hyperbolic']:.4f}")
            for col, points in enumerate(locations):
                ax = fig.add_subplot(len(seeds), 3, row*3+col+1, projection="3d")
                axes.append(ax)
                segments = rec["curves"] if col == 2 else np.stack((points[a], points[b]), axis=1)
                ax.add_collection3d(Line3DCollection(segments, colors="#64748b", linewidths=.55, alpha=.23))
                ax.scatter(*points.T, c=rec["xyz"][:, 2], cmap=cmap, norm=color_norm,
                           s=23, edgecolors="white", linewidths=.35, depthshade=False)
                if col == 2:
                    ax.plot_wireframe(*sphere, color="0.65", linewidth=.35, alpha=.18)
                    lo, hi = -np.ones(3), np.ones(3)
                    labels = ("Ball x", "Ball y", "Ball z")
                else:
                    lo, hi = anatomy_bounds if col == 0 else euclidean_bounds
                    labels = tuple(f"{s} (mm)" for s in ("X", "Y", "Z")) if col == 0 else (
                        "Latent x (mm)", "Latent y (mm)", "Latent z (mm)")
                ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                       xlabel=labels[0], ylabel=labels[1], zlabel=labels[2],
                       title=f"{titles[col]}\nPanel {seed} | {len(points)} landmarks")
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.set_major_locator(MaxNLocator(3))
                    axis.pane.fill = False
                ax.tick_params(labelsize=7, pad=0)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=20, azim=-55)
                ax.set_proj_type("ortho")
                ax.grid(False)
        fig.colorbar(ScalarMappable(norm=color_norm, cmap=cmap), ax=axes, shrink=.6, pad=.015,
                     label="Anatomical z in fdt_paths world space (mm)")
        fig.suptitle(f"{graph}: same landmarks, colors, and {n_links} fitting pairs in every column\n"
                     "Lines: anatomical chords / model geodesics; not reconstructed fibers. Latent axes are arbitrary.", fontsize=12)
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{graph}_xyz_E3_H3.{extension}", dpi=220, bbox_inches="tight", pad_inches=.15)
        plt.show()
        print(f"Saved comparison: {graph}, {len(seeds)} panels, {n_links} lines per panel.")
    (out / "protocol.json").write_text(json.dumps(dict(
        source_fits=str(src), source_ball=str(balls), panels=seeds,
        display_pairs=n_links, original_pairs_retained=len(old), extra_pair_seed=1701,
        selection="Retain Cell 16 pairs, then uniformly sample additional fitting pairs without replacement",
        line_meaning="Anatomical chords / E3 straight geodesics / H3 curved geodesics, not graph edges or fibers",
        coordinates="Anatomical world mm; Euclidean latent target-distance mm; hyperbolic unit display ball",
        comparison="Identical nodes and pairs; no additional fitting, registration, model selection, or inference"
    ), indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "All coordinates come from saved Cell 15 fits and Cell 16 exports.\n"
        "Anatomical XYZ is not the Euclidean embedding; its chords do not follow graph paths.\n"
        "E3 coordinates approximate graph distances; multiplying by the fitting scale gives target-distance mm.\n"
        "H3 ball coordinates preserve the saved hyperbolic metric; Euclidean chord length in the ball is not that metric.\n"
        "Latent axes are arbitrary. Shapes and radii alone do not establish which metric fits better.\n"
        "Every column displays the same pairs, sampled without using test errors or residuals.\n"
        "Increasing line count changes display density only. All 64 landmarks remain unchanged.\n"
        "These are one-brain landmark-panel visualizations, not full-node connectome embeddings.\n"
        "The saved held-out scores remain exploratory after earlier inspection.\n", encoding="utf-8")
    print("\nReproduced saved scores; no models were fitted:\n" + scores.to_string(index=False))
    print(f"Saved: {out}")
    return scores


if __name__ == "__main__":
    scores17 = run_cell17(PROJECT, N_LINKS)
