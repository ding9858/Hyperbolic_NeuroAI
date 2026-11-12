# Cell 13: matched dimensions, with curvature selected inside the fitting pairs.
# Same previously inspected test pairs: exploratory sensitivity, not confirmation.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra

if any(k not in globals() for k in
       ("fit_best", "pair_geometry", "encode_start", "OPT_LOG", "MAXITER")):
    raise RuntimeError("Run Cell 9 first in this kernel to load its fitting functions.")


def run_cell13():
    root = Path("/mnt/g/Research/TUM_connectome/HyperbolicTest/results")
    src, old_E = root / "cell11_new_landmarks", root / "cell12_dimension_check"
    out = root / "cell13_matched_dimensions"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((src / "protocol.json").read_text())
    seeds, metrics = protocol["landmark_seeds"], tuple(protocol["fixed_H"])
    dims = (3, 5, 8)
    grid = (-0.015625, -0.0625, -0.25, -1., -4., -8., -16.)
    inner_seed, init_seed = 1313, 711
    with np.load(src / "pair_split.npz") as z:
        i, j, fit, test = (z[k].copy() for k in ("i", "j", "train", "test"))
    assert np.array_equal(np.sort(np.r_[fit, test]), np.arange(len(i)))
    order = np.random.default_rng(inner_seed).permutation(fit)
    train, val = np.split(order, [int(0.75 * len(order))])
    np.savez_compressed(out / "pair_split.npz", i=i, j=j, train=train,
                        validation=val, fit=fit, test=test)
    (out / "protocol.json").write_text(json.dumps(dict(
        dimensions=dims, hyperbolic_grid=grid, panels=seeds,
        inner_seed=inner_seed, initialization_seed=init_seed, maxiter=MAXITER,
        selection="Minimum inner-validation RMSE; starts chosen by training loss only",
        normalization="Inner-training median retained for refit; reported nRMSE uses Cell 11 fitting median",
        starts="Two for inner fits; four for each final E/H fit, including saved fitting-only E coordinates",
        scope="Same one-brain graph and previously inspected test pairs; adaptive exploratory comparison",
        inference="No dimension chosen on test; panels are not independent subjects; no p-values or CIs",
    ), indent=2), encoding="utf-8")

    def starts(n, dim, pair_ids, target):
        W = np.zeros((n, n))
        W[i[pair_ids], j[pair_ids]] = W[j[pair_ids], i[pair_ids]] = target[pair_ids]
        filled = dijkstra(W, directed=False)
        if not np.isfinite(filled).all():
            raise ValueError("Fitting-pair graph is disconnected.")
        J = np.eye(n) - np.ones((n, n))/n
        values, vectors = np.linalg.eigh(-0.5*J@(filled**2)@J)
        mds = vectors[:, -dim:]*np.sqrt(np.maximum(values[-dim:], 0))
        answer = []
        for X in (mds, np.random.default_rng(init_seed).normal(size=(n, dim))):
            X = X-X.mean(axis=0)
            distances = np.linalg.norm(X[i[pair_ids]]-X[j[pair_ids]], axis=1)
            X *= np.dot(distances, target[pair_ids])/np.dot(distances, distances)
            answer.append(X)
        return answer

    def predict(Z, kind, k, pairs):
        p = pair_geometry(Z, kind, k, i[pairs], j[pairs])[0]
        error = 0.
        if kind == "hyperbolic":
            # Check cancellation in Lorentz distances using extended precision.
            A, B = Z[i[pairs]].astype(np.longdouble), Z[j[pairs]].astype(np.longdouble)
            R = 1/np.sqrt(np.longdouble(abs(k)))
            ta, tb = np.sqrt(R*R+(A*A).sum(1)), np.sqrt(R*R+(B*B).sum(1))
            c = (ta*tb-(A*B).sum(1))/(R*R)
            q = np.asarray(R*np.arccosh(np.maximum(c, 1)), dtype=float)
            error = float(np.max(np.abs(p-q)))
        ok = bool(np.isfinite(p).all() and np.isfinite(error) and error <= 1e-6)
        return p, ok

    rows, scans, predictions, log_start = [], [], [], len(OPT_LOG)
    print(f"Matched dimensions: {dims}; curvature chosen without test distances.")
    print(f"Inner train/validation: {len(train)}/{len(val)}; final fit/test: {len(fit)}/{len(test)}")
    print("Already-inspected test pairs; exploratory comparison. No new shortest paths.")
    for seed in seeds:
        with np.load(src / f"matched_metrics_{seed}.npz") as z:
            ids = z["base_node_id"].copy()
            matrices = {m: z[m].astype(float) for m in metrics}
        n = len(ids)
        ei, ej = np.triu_indices(n, 1)
        assert np.array_equal(i, ei) and np.array_equal(j, ej) and max(dims) < n
        for metric in metrics:
            raw = matrices[metric][i, j]
            scale, report_scale = float(np.median(raw[train])), float(np.median(raw[fit]))
            if not np.isfinite(raw).all() or np.min(raw) <= 0:
                raise ValueError("Expected finite positive off-diagonal distances.")
            target = raw/scale
            for dim in dims:
                context = dict(graph=metric, seed=seed, dimension=dim)
                initial = starts(n, dim, train, target)
                candidates = []
                for kind, ks in (("euclidean", (0.,)), ("hyperbolic", grid)):
                    for k in ks:
                        Z, converged, loss = fit_best(
                            [encode_start(X, kind, k) for X in initial], kind, k,
                            i[train], j[train], target[train], dict(**context, stage="cell13_inner"))
                        p, numerical_ok = predict(Z, kind, k, val)
                        score = float(np.sqrt(np.mean((p-target[val])**2)))
                        if not np.isfinite(score):
                            raise ValueError("Nonfinite validation score; inspect the fit.")
                        item = dict(panel_seed=seed, metric=metric, dimension=dim, model=kind,
                                    curvature=k, val_RMSE_mm=score*scale,
                                    converged=converged, numerical_ok=numerical_ok)
                        scans.append(item)
                        candidates.append((item, Z))
                e_inner = candidates[0]
                h_inner = min(candidates[1:], key=lambda x: x[0]["val_RMSE_mm"])
                k_h = h_inner[0]["curvature"]
                search_ok = all(x[0]["converged"] and x[0]["numerical_ok"] for x in candidates)

                # All these initial coordinates use only the original 80% fitting pairs.
                saved_path = (src / f"fit_{metric}_{seed}_euclidean.npz" if dim == 3 else
                              old_E / f"fit_{metric}_{seed}_E{dim}.npz")
                with np.load(saved_path) as z:
                    assert np.array_equal(z["base_node_id"], ids)
                    saved_E = z["coordinates"].copy()*float(z["train_scale_mm"])/scale
                assert saved_E.shape == (n, dim)
                final_starts = starts(n, dim, fit, target)
                scores, checks = {}, []
                for kind, selected in (("euclidean", e_inner), ("hyperbolic", h_inner)):
                    item, inner_Z = selected
                    k = item["curvature"]
                    initial = [encode_start(X, kind, k) for X in final_starts]
                    initial += [inner_Z.copy(), encode_start(saved_E, kind, k)]
                    Z, converged, loss = fit_best(initial, kind, k, i[fit], j[fit], target[fit],
                                                dict(**context, stage="cell13_refit"))
                    p, numerical_ok = predict(Z, kind, k, test)
                    if kind == "euclidean":
                        old_loss = np.mean((np.linalg.norm(saved_E[i[fit]]-saved_E[j[fit]], axis=1)
                                            -target[fit])**2)
                        numerical_ok &= loss <= old_loss+1e-6
                    selected_ok = item["converged"] and item["numerical_ok"]
                    checks.append(bool(converged and numerical_ok and selected_ok))
                    scores[kind] = (np.sqrt(loss)*scale/report_scale,
                                   np.sqrt(np.mean((p*scale-raw[test])**2))/report_scale)
                    np.savez_compressed(out / f"fit_{metric}_{seed}_{kind}{dim}.npz",
                        coordinates=Z, model=kind, curvature_scaled=k, train_scale_mm=scale,
                        report_scale_mm=report_scale, base_node_id=ids, dimension=dim)
                    predictions.append(pd.DataFrame(dict(panel_seed=seed, metric=metric,
                        dimension=dim, model=kind, landmark_a=i[test], landmark_b=j[test],
                        observed_mm=raw[test], predicted_mm=p*scale)))
                etr, ete = scores["euclidean"]
                htr, hte = scores["hyperbolic"]
                rows.append(dict(panel_seed=seed, metric=metric, dimension=dim, H_curvature=k_h,
                    near_flat_boundary=k_h == grid[0], strong_boundary=k_h == grid[-1],
                    inner_val_favors_H=h_inner[0]["val_RMSE_mm"] < e_inner[0]["val_RMSE_mm"],
                    E_train_nRMSE=etr, H_train_nRMSE=htr, E_test_nRMSE=ete, H_test_nRMSE=hte,
                    E_minus_H=ete-hte, relative_H_reduction=(ete-hte)/ete,
                    selected_fits_ok=all(checks), all_candidates_ok=search_ok))
                pd.DataFrame(rows).to_csv(out / "matched_results.csv", index=False)
                pd.DataFrame(scans).to_csv(out / "curvature_validation.csv", index=False)
                pd.DataFrame(OPT_LOG[log_start:]).to_csv(out / "optimizer_runs.csv", index=False)
                print(f"Panel {seed}, {metric}, d={dim}: selected k={k_h:g}; fit_ok={all(checks)}", flush=True)

    results = pd.DataFrame(rows)
    pd.concat(predictions, ignore_index=True).to_csv(out / "test_pair_predictions.csv", index=False)
    summary = results.groupby(["metric", "dimension"], sort=False).agg(
        mean_E_test_nRMSE=("E_test_nRMSE", "mean"), mean_H_test_nRMSE=("H_test_nRMSE", "mean"),
        mean_E_minus_H=("E_minus_H", "mean"), k_min=("H_curvature", "min"), k_max=("H_curvature", "max"),
        near_flat_selections=("near_flat_boundary", "sum"), strong_boundary_selections=("strong_boundary", "sum"),
        all_selected_ok=("selected_fits_ok", "all"), all_candidates_ok=("all_candidates_ok", "all")).reset_index()
    summary.to_csv(out / "matched_summary.csv", index=False)
    contrasts = []
    for graph in ("supported", "cone45"):
        g = results[results.metric == graph+"_graph"].set_index(["panel_seed", "dimension"])
        s = results[results.metric == graph+"_spatial"].set_index(["panel_seed", "dimension"])
        contrasts.append(pd.DataFrame(dict(graph=graph, graph_reduction=g.E_minus_H,
            spatial_reduction=s.E_minus_H, extra_reduction=g.E_minus_H-s.E_minus_H)).reset_index())
    contrast = pd.concat(contrasts, ignore_index=True)
    contrast.to_csv(out / "graph_spatial_contrast.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        for seed in seeds:
            part = results[(results.metric == metric) & (results.panel_seed == seed)]
            ax.plot(part.dimension, part.E_minus_H, "o-", label=f"panel {seed}")
            bad = ~part.selected_fits_ok | ~part.all_candidates_ok
            ax.scatter(part.loc[bad, "dimension"], part.loc[bad, "E_minus_H"], marker="x", c="red", s=65)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set(title=metric, xlabel="Embedding dimension (same for E and H)", xticks=dims,
               ylabel="Test nRMSE: Euclidean minus hyperbolic")
        ax.legend(fontsize=8)
    fig.suptitle("Above zero favors H at the same dimension | previously inspected test pairs")
    fig.savefig(out / "matched_dimension_comparison.png", dpi=180)
    plt.show()
    print("\nMatched-dimension summary (descriptive, not independent confirmation):")
    print(summary.to_string(index=False))
    print("\nGraph minus spatial H advantage (normalized within each metric):")
    print(contrast.to_string(index=False))
    print(f"\nOptimizer convergence: {sum(r['success'] for r in OPT_LOG[log_start:])}/{len(OPT_LOG)-log_start}")
    if not (results.selected_fits_ok.all() and results.all_candidates_ok.all()):
        print("WARNING: Review flagged fits/candidates; do not treat their minima as reliable evidence.")
    print("Near-flat selections do not establish strictly negative curvature; strong-boundary minima are unbracketed.")
    print(f"Saved: {out}")
    return results, summary, contrast


results13, summary13, contrast13 = run_cell13()
