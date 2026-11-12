# Cell 15: original versus partially rewired graphs at fixed geometry.
# Same Cell 11 panels and fitting/test pairs; E^3 versus H^3 with k=-4.
# No curvature search, FSL, tracking, or use of display-space coordinates.
# Three controls and within-brain panels support descriptive comparisons only.
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.csgraph import dijkstra


def pair15(Z, kind, k, i, j):
    A, B = Z[i], Z[j]
    if kind == "euclidean":
        difference = A - B
        distance = np.linalg.norm(difference, axis=1)
        gradient = difference / np.maximum(distance[:, None], 1e-12)
        return distance, gradient, -gradient
    radius = 1 / np.sqrt(abs(k))
    ta = np.sqrt(radius**2 + (A*A).sum(axis=1))
    tb = np.sqrt(radius**2 + (B*B).sum(axis=1))
    raw = (ta*tb - (A*B).sum(axis=1)) / radius**2
    c = np.maximum(raw, 1 + 1e-12)
    factor = (raw > 1 + 1e-12) / (radius * np.sqrt(c*c - 1))
    return (radius * np.arccosh(c), ((tb/ta)[:, None]*A - B)*factor[:, None],
            ((ta/tb)[:, None]*B - A)*factor[:, None])


def loss15(flat, n, kind, k, i, j, target):
    Z = flat.reshape(n, 3)
    pred, gi, gj = pair15(Z, kind, k, i, j)
    residual = pred - target
    gradient = np.zeros_like(Z)
    weight = 2 * residual[:, None] / len(target)
    np.add.at(gradient, i, weight * gi)
    np.add.at(gradient, j, weight * gj)
    return float(np.mean(residual**2)), gradient.ravel()


def starts15(n, i, j, target, seed):
    # Identical training-only MDS and random initializations to Cell 11.
    W = np.zeros((n, n))
    W[i, j] = W[j, i] = target
    distances = dijkstra(W, directed=False)
    if not np.isfinite(distances).all():
        raise ValueError("Fitting-pair graph is disconnected.")
    J = np.eye(n) - np.ones((n, n))/n
    values, vectors = np.linalg.eigh(-0.5 * J @ distances**2 @ J)
    mds = vectors[:, -3:] * np.sqrt(np.maximum(values[-3:], 0))
    initial = [mds, np.random.default_rng(seed).normal(size=(n, 3))]
    for X in initial:
        X -= X.mean(axis=0)
        current = np.linalg.norm(X[i] - X[j], axis=1)
        X *= np.dot(current, target) / np.dot(current, current)
    return initial


def fit15(initial, kind, k, i, j, target, context, logs, maxiter):
    fits = []
    for start_id, X in enumerate(initial):
        Z = X.copy()
        if kind == "hyperbolic":
            radius = 1 / np.sqrt(abs(k))
            norm = np.linalg.norm(X, axis=1)
            Z = radius*np.sinh(norm/radius)[:, None]*X/np.maximum(norm[:, None], 1e-12)
        result = minimize(loss15, Z.ravel(), args=(len(Z), kind, k, i, j, target),
                          jac=True, method="L-BFGS-B",
                          options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxls=40))
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError(f"Nonfinite fit: {context}, {kind}, start {start_id}")
        logs.append(dict(**context, model=kind, curvature=k, start=start_id,
                         success=bool(result.success), iterations=int(result.nit),
                         training_loss=float(result.fun), message=str(result.message)))
        fits.append(result)
    # Select by fitting loss only; retain convergence flags even for failed minima.
    best = min(fits, key=lambda x: x.fun)
    return best.x.reshape(-1, 3), float(best.fun), bool(best.success), all(x.success for x in fits)


def check_metric15(D):
    if not np.isfinite(D).all() or not np.allclose(D, D.T, atol=1e-5, rtol=0):
        raise ValueError("Landmark distances must be finite and symmetric.")
    if not np.allclose(np.diag(D), 0, atol=1e-6) or np.any(D[np.triu_indices(len(D), 1)] <= 0):
        raise ValueError("Invalid distance diagonal or off-diagonal entries.")
    excess = max(float(np.max(D - D[:, k, None] - D[k, None, :])) for k in range(len(D)))
    if excess > 1e-4:
        raise ValueError(f"Triangle-inequality violation: {excess} mm")


def run_cell15(project):
    root = Path(project) / "results"
    src, panels = root / "cell04_orientation_graph", root / "cell11_new_landmarks"
    controls, out = root / "cell14_graph_controls", root / "cell15_control_geometry"
    protocol11 = json.loads((panels / "protocol.json").read_text())
    panel_seeds = tuple(protocol11["landmark_seeds"])
    null_seeds = (1401, 1402, 1403)
    names = ("supported", "cone45")
    maxiter = int(protocol11.get("maxiter", 600))
    if protocol11.get("dimension", 3) != 3:
        raise ValueError("Expected the three-dimensional Cell 11 protocol.")
    if any(protocol11["fixed_H"][g + "_graph"] != -4 for g in names):
        raise ValueError("Expected frozen graph curvature -4 from Cell 11.")
    with np.load(panels / "pair_split.npz") as z:
        i, j, train, test = [z[k].astype(int) for k in ("i", "j", "train", "test")]
        pair_seed = int(z["pair_seed"]) if "pair_seed" in z else int(protocol11.get("pair_seed", 711))
    if not np.array_equal(np.sort(np.r_[train, test]), np.arange(len(i))):
        raise ValueError("Fitting and test indices must partition all pairs exactly once.")
    qc = pd.read_csv(controls / "control_qc.csv")
    required = ("degree_preserved", "length_hist_preserved", "components_preserved",
                "forest_preserved", "landmarks_connected", "construction_ok")
    for graph in names:
        for seed in null_seeds:
            part = qc[(qc.graph == graph) & (qc.seed == seed)]
            if len(part) != 1 or not part[list(required)].astype(str).apply(
                    lambda column: column.str.lower().isin(("true", "1"))).to_numpy().all():
                raise ValueError(f"Missing or failed Cell 14 control checks: {graph}, {seed}")

    data, landmark_tables = {}, {}
    for seed in panel_seeds:
        with np.load(panels / f"matched_metrics_{seed}.npz") as z:
            data[seed] = {k: z[k].copy() for k in z.files}
        landmark_tables[seed] = pd.read_csv(panels / f"landmarks_{seed}.csv")
        n = len(data[seed]["base_node_id"])
        if not np.array_equal(np.c_[i, j], np.column_stack(np.triu_indices(n, 1))):
            raise ValueError("Saved pair indices do not match panel size/order.")
        if not np.array_equal(landmark_tables[seed].base_node_id, data[seed]["base_node_id"]):
            raise ValueError("Landmark CSV and distance file disagree.")
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(dimension=3, fixed_normalized_curvature=-4, panel_seeds=panel_seeds,
                    null_seeds=null_seeds, fitting_pairs=len(train), test_pairs=len(test),
                    initialization="Training-only MDS and random, same as Cell 11; two starts per model",
                    maxiter=maxiter, initialization_seed=pair_seed,
                    normalization="Each graph/panel fitting-pair median; also report original-median and relative gains",
                    selection="No geometry/dimension/curvature selection; start chosen by fitting loss only",
                    comparison="Original minus control (E3-H3 test nRMSE gain), paired within each panel",
                    scope="Repeated analysis of already-inspected pairs in one brain; descriptive fixed-model perturbation check",
                    curvature_limit="Curvature was selected using original-graph discovery data; this is not a comparison of independently optimized geometries",
                    control_limit="Fixed random spanning forest retains original structure; partial rewiring with unestablished mixing; no p-values or confidence intervals",
                    coordinates="Native graph edge lengths; display-space exports are not analysis inputs")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    np.savez_compressed(out / "pair_split.npz", i=i, j=j, train=train, test=test, pair_seed=pair_seed)
    print(f"Fixed E^3/H^3 comparison; k=-4; panels {panel_seeds}; controls {null_seeds}.")
    print(f"Fitting/test pairs: {len(train)}/{len(test)}; two starts; original fits rerun identically.")
    print("No curvature search. Already-inspected pairs; conditional descriptive comparison.")
    rows, logs, predictions, distance_rows, provenance = [], [], [], [], []
    for graph in names:
        with np.load(src / f"nodes_{graph}.npz") as z:
            ids = z["base_node_id"].copy()
        with np.load(controls / f"node_ids_{graph}.npz") as z:
            if not np.array_equal(ids, z["base_node_id"]):
                raise ValueError("Original and control node ordering differ.")
        variants = [("original", src / f"edge_lengths_{graph}.npz")]
        variants += [(f"null_{seed}", controls / f"edge_lengths_{graph}_null_{seed}.npz") for seed in null_seeds]
        for variant, path in variants:
            begin = perf_counter()
            signature = hashlib.sha256(path.read_bytes()).hexdigest()
            provenance.append(dict(graph=graph, variant=variant, file=str(path), sha256=signature))
            G = None
            if variant != "original":
                G = sparse.load_npz(path).tocsr()
                if (G.shape != (len(ids), len(ids)) or G.diagonal().any() or
                        (G != G.T).nnz or np.any(G.data <= 0) or not np.isfinite(G.data).all()):
                    raise ValueError(f"Invalid control graph: {path}")
            for seed in panel_seeds:
                base_ids = data[seed]["base_node_id"]
                locations = landmark_tables[seed][graph + "_node_id"].to_numpy(dtype=int)
                if np.any(locations < 0) or np.any(locations >= len(ids)) or not np.array_equal(ids[locations], base_ids):
                    raise ValueError("Landmark row indices do not match base node IDs.")
                original = data[seed][graph + "_graph"].astype(float)
                cache = out / f"distances_{graph}_{variant}_{seed}.npz"
                reused = False
                D = original if variant == "original" else None
                if D is None and cache.is_file():
                    with np.load(cache) as z:
                        if str(z["graph_sha256"]) == signature and np.array_equal(z["base_node_id"], base_ids):
                            D, reused = z["distance_mm"].copy(), True
                if D is None:
                    D = np.empty((len(locations), len(locations)))
                    for start in range(0, len(locations), 8):
                        D[start:start+8] = dijkstra(G, directed=False, indices=locations[start:start+8])[:, locations]
                    print(f"Distances: {graph}, {variant}, panel {seed} complete", flush=True)
                check_metric15(D)
                spatial = data[seed][graph + "_spatial"]
                if np.any(D < spatial - 1e-3):
                    raise ValueError("A local graph distance is shorter than its spatial reference.")
                np.savez_compressed(cache, distance_mm=D, base_node_id=base_ids, graph_sha256=signature)
                raw, baseline = D[i, j], original[i, j]
                scale, original_scale = float(np.median(raw[train])), float(np.median(baseline[train]))
                target = raw / scale
                initial = starts15(len(D), i[train], j[train], target[train], pair_seed)
                ratio = raw / baseline
                distance_rows.append(dict(graph=graph, variant=variant, panel_seed=seed,
                                          median_distance_ratio=float(np.median(ratio)),
                                          p05_distance_ratio=float(np.percentile(ratio, 5)),
                                          p95_distance_ratio=float(np.percentile(ratio, 95)),
                                          fraction_pairs_shortened=float(np.mean(ratio < 1 - 1e-6)),
                                          train_scale_mm=scale, original_train_scale_mm=original_scale,
                                          distance_cache_reused=reused))
                for kind, k in (("euclidean", 0.), ("hyperbolic", -4.)):
                    context = dict(graph=graph, variant=variant, panel_seed=seed)
                    Z, loss, converged, all_converged = fit15(initial, kind, k, i[train], j[train],
                                                            target[train], context, logs, maxiter)
                    pred = pair15(Z, kind, k, i[test], j[test])[0]
                    roundoff = 0.
                    if kind == "hyperbolic":
                        A, B = Z[i[test]].astype(np.longdouble), Z[j[test]].astype(np.longdouble)
                        R = 1/np.sqrt(np.longdouble(abs(k)))
                        c = (np.sqrt(R*R+(A*A).sum(1))*np.sqrt(R*R+(B*B).sum(1))-(A*B).sum(1))/(R*R)
                        roundoff = float(np.max(np.abs(R*np.arccosh(np.maximum(c, 1))-pred)))
                    error_mm = (pred - target[test]) * scale
                    rmse = float(np.sqrt(np.mean(error_mm**2)))
                    numerical_ok = bool(np.isfinite(pred).all() and np.isfinite(roundoff) and roundoff <= 1e-6)
                    rows.append(dict(**context, model=kind, curvature_scaled=k,
                                     train_scale_mm=scale, original_train_scale_mm=original_scale,
                                     train_nRMSE=np.sqrt(loss), test_RMSE_mm=rmse,
                                     test_nRMSE=rmse/scale, test_RMSE_over_original_median=rmse/original_scale,
                                     converged=converged, all_starts_converged=all_converged,
                                     numerical_ok=numerical_ok, numerical_error=roundoff,
                                     fit_ok=bool(converged and numerical_ok)))
                    predictions.append(pd.DataFrame(dict(**context, model=kind, pair_index=test,
                                                         landmark_a=i[test], landmark_b=j[test],
                                                         observed_mm=raw[test], predicted_mm=pred*scale)))
                    np.savez_compressed(out / f"fit_{graph}_{variant}_{seed}_{kind}.npz",
                                        coordinates=Z, model=kind, curvature_scaled=k,
                                        train_scale_mm=scale, base_node_id=base_ids, graph_sha256=signature)
                pd.DataFrame(rows).to_csv(out / "fit_results.csv", index=False)
                pd.DataFrame(logs).to_csv(out / "optimizer_runs.csv", index=False)
                print(f"Fitted: {graph}, {variant}, panel {seed}; E/H fit_ok={rows[-2]['fit_ok']}/{rows[-1]['fit_ok']}", flush=True)
            print(f"Finished {graph}, {variant}: {perf_counter()-begin:.1f} s", flush=True)
            del G

    results = pd.DataFrame(rows)
    pd.concat(predictions, ignore_index=True).to_csv(out / "test_pair_predictions.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(out / "distance_changes.csv", index=False)
    pd.DataFrame(provenance).to_csv(out / "graph_provenance.csv", index=False)
    gains = []
    for keys, group in results.groupby(["graph", "variant", "panel_seed"], sort=False):
        E, H = [group[group.model == kind].iloc[0] for kind in ("euclidean", "hyperbolic")]
        gains.append(dict(graph=keys[0], variant=keys[1], panel_seed=keys[2],
                          E_test_nRMSE=E.test_nRMSE, H_test_nRMSE=H.test_nRMSE,
                          H_gain=E.test_nRMSE-H.test_nRMSE,
                          H_gain_original_scale=E.test_RMSE_over_original_median-H.test_RMSE_over_original_median,
                          relative_H_reduction=(E.test_RMSE_mm-H.test_RMSE_mm)/E.test_RMSE_mm,
                          fit_ok=bool(E.fit_ok and H.fit_ok)))
    gains = pd.DataFrame(gains)
    contrasts = []
    for graph in names:
        for seed in panel_seeds:
            group = gains[(gains.graph == graph) & (gains.panel_seed == seed)].set_index("variant")
            original = group.loc["original"]
            for control_seed in null_seeds:
                control = group.loc[f"null_{control_seed}"]
                contrasts.append(dict(graph=graph, panel_seed=seed, control_seed=control_seed,
                                      original_H_gain=original.H_gain, control_H_gain=control.H_gain,
                                      original_minus_control=original.H_gain-control.H_gain,
                                      original_minus_control_shared_scale=original.H_gain_original_scale-control.H_gain_original_scale,
                                      original_minus_control_relative=original.relative_H_reduction-control.relative_H_reduction,
                                      fit_ok=bool(original.fit_ok and control.fit_ok)))
    contrasts = pd.DataFrame(contrasts)
    summary = gains.groupby(["graph", "variant"], sort=False).agg(
        mean_E_test_nRMSE=("E_test_nRMSE", "mean"), mean_H_test_nRMSE=("H_test_nRMSE", "mean"),
        mean_H_gain=("H_gain", "mean"), mean_relative_H_reduction=("relative_H_reduction", "mean"),
        all_fit_ok=("fit_ok", "all")).reset_index()
    gains.to_csv(out / "panel_gains.csv", index=False)
    contrasts.to_csv(out / "original_control_contrasts.csv", index=False)
    summary.to_csv(out / "control_summary.csv", index=False)
    old_path = panels / "test_results.csv"
    if old_path.is_file():
        old = pd.read_csv(old_path)
        old = old[old.metric.isin([g + "_graph" for g in names]) & old.model.isin(("euclidean", "hyperbolic"))].copy()
        old["graph"] = old.metric.str.removesuffix("_graph")
        audit = results[results.variant == "original"].merge(old[["graph", "panel_seed", "model", "test_nRMSE"]],
                    on=["graph", "panel_seed", "model"], suffixes=("_new", "_cell11"), validate="one_to_one")
        audit["absolute_difference"] = abs(audit.test_nRMSE_new - audit.test_nRMSE_cell11)
        audit.to_csv(out / "original_fit_reproduction.csv", index=False)
        if len(audit):
            print(f"Maximum original-fit difference from Cell 11: {audit.absolute_difference.max():.3g} nRMSE")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(len(panel_seeds)))
    variants = ["original"] + [f"null_{s}" for s in null_seeds]
    for row, graph in enumerate(names):
        subset = gains[gains.graph == graph]
        for color, seed in zip(colors, panel_seeds):
            group = subset[subset.panel_seed == seed].set_index("variant").loc[variants]
            good = group.fit_ok.to_numpy(dtype=bool)
            for index, variant in enumerate(variants):
                if good[index]:
                    axes[row, 0].scatter(group.iloc[index].E_test_nRMSE, group.iloc[index].H_test_nRMSE,
                                         c=[color], marker="o" if index == 0 else "+", s=65 if index == 0 else 40)
            axes[row, 1].plot(range(4), group.H_gain.where(group.fit_ok), "o-", color=color, label=f"panel {seed}")
            comparison = contrasts[(contrasts.graph == graph) & (contrasts.panel_seed == seed)].set_index("control_seed").loc[list(null_seeds)]
            axes[row, 2].plot(range(3), comparison.original_minus_control.where(comparison.fit_ok), "o-", color=color)
        maximum = float(subset[["E_test_nRMSE", "H_test_nRMSE"]].max().max()) * 1.08
        axes[row, 0].plot([0, maximum], [0, maximum], "--", color="0.6", lw=1)
        axes[row, 0].set(xlabel="E3 test nRMSE", ylabel="H3 test nRMSE", xlim=(0, maximum), ylim=(0, maximum),
                         title=f"{graph}: prediction errors\nCircles: original; plus signs: controls")
        axes[row, 1].set(xticks=range(4), xticklabels=["Original"]+[str(s) for s in null_seeds],
                         ylabel="E3 - H3 test nRMSE", title="Fixed-model hyperbolic advantage")
        axes[row, 2].set(xticks=range(3), xticklabels=[str(s) for s in null_seeds], xlabel="Control seed",
                         ylabel="Original gain - control gain", title="Positive: larger original-graph advantage")
        for ax in axes[row, 1:]:
            ax.axhline(0, color="0.6", ls="--", lw=1)
        axes[row, 1].legend(fontsize=8)
    fig.suptitle("Fixed H3 curvature -4 | paired panels | partial-rewiring controls | descriptive")
    fig.savefig(out / "control_geometry_comparison.png", dpi=180)
    fig.savefig(out / "control_geometry_comparison.pdf")
    plt.show()
    print("\nModel summary (means over panels of one brain; no inferential error bars):")
    print(summary.to_string(index=False))
    print("\nPaired original-minus-control H advantage (positive favors the original):")
    print(contrasts.to_string(index=False))
    print(f"\nOptimizer convergence: {sum(x['success'] for x in logs)}/{len(logs)}")
    if not results.fit_ok.all():
        print("WARNING: Failed selected fits remain in tables and are omitted from plots; do not interpret affected contrasts.")
    print("A positive contrast concerns this fixed-curvature model and these partially rewired controls.")
    print("No p-values, independent-subject replication, or optimal-curvature claims are made.")
    print(f"Saved: {out}")
    return results, contrasts


if __name__ == "__main__":
    results15, contrasts15 = run_cell15("/mnt/g/Research/TUM_connectome/HyperbolicTest")
