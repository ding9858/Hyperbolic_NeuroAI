# Cell 18: fixed-ROI, within-region E3/H3 comparison.
# Use CC/roi_1.nii.gz, not the historical bin_roi files.
# Keep original local edges whose endpoints both belong to this ROI.
# Fit the largest within-ROI component; report its coverage explicitly.
# No FSL, whole-brain retracking, curvature search, or outcome-based ROI selection.
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

PROJECT = "/mnt/g/Research/TUM_connectome/HyperbolicTest"
ROI_ID, GRAPH = 1, "supported"
MAX_LANDMARKS, MIN_LANDMARKS = 128, 16
CURVATURE = -4.0
SPLIT_SEEDS = (1807, 1817, 1827)


def membership18(coords, image, expected_shape, expected_affine):
    if image.shape != tuple(expected_shape) or not np.allclose(image.affine, expected_affine, atol=1e-4, rtol=0):
        raise ValueError("ROI grid differs from the saved fdt-space grid. No transform was guessed.")
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("ROI image has nonfinite values.")
    if not np.isfinite(coords).all():
        raise ValueError("Saved node coordinates are nonfinite.")
    vox = np.floor(coords + .5).astype(np.int64)
    inside = np.all((vox >= 0) & (vox < np.array(image.shape)), axis=1)
    selected = np.zeros(len(coords), dtype=bool)
    selected[inside] = data[tuple(vox[inside].T)] != 0
    return selected, int(np.count_nonzero(data))


def ball18(Z, k):
    R = 1/np.sqrt(-k)
    time = np.sqrt(R*R + np.sum(Z*Z, axis=1))
    return Z/(time[:, None]+R), np.column_stack((time, Z))


def arcs18(L, a, b):
    t = np.linspace(0, 1, 48)[None, :, None]
    Q = (1-t)*L[a, None, :] + t*L[b, None, :]
    norm = np.sqrt(Q[..., 0]**2-np.sum(Q[..., 1:]**2, axis=-1))
    return Q[..., 1:]/(Q[..., 0]+norm)[..., None]


def limits18(points):
    low, high = points.min(0), points.max(0)
    center, half = (low+high)/2, max(np.max(high-low)*.58, 1e-6)
    return center-half, center+half


def run_cell18(project, roi_id=1, graph="supported", max_landmarks=128):
    root = Path(project)
    src, exports = root / "results/cell04_orientation_graph", root / "results/cell14c_fdt_space"
    out = root / f"results/cell18_roi_{roi_id:03d}_{graph}"
    out.mkdir(parents=True, exist_ok=True)
    roi_path = root / f"CC/roi_{roi_id}.nii.gz"
    roi = nib.load(roi_path)
    with np.load(exports / "coordinate_transforms.npz") as z:
        shape, affine = z["fdt_shape"].copy(), z["fdt_affine"].copy()
    with np.load(src / f"nodes_{graph}.npz") as z:
        ids, ijk, components = z["base_node_id"].copy(), z["ijk"].copy(), z["component"].copy()
    with np.load(exports / f"nodes_{graph}_in_fdt.npz") as z:
        if not np.array_equal(ids, z["base_node_id"]):
            raise ValueError("Native and display-space node IDs differ.")
        vox, world = z["fdt_voxel_xyz"].copy(), z["fdt_world_xyz_mm"].copy()
    belongs, mask_voxels = membership18(vox, roi, shape, affine)
    roi_rows = np.flatnonzero(belongs)
    if len(roi_rows) == 0:
        raise ValueError(f"ROI {roi_id} contains no {graph} graph nodes under nearest-center assignment.")
    full = sparse.load_npz(src / f"edge_lengths_{graph}.npz").tocsr()
    if full.shape != (len(ids), len(ids)):
        raise ValueError("Graph and node table differ.")
    G = full[roi_rows][:, roi_rows].astype(float).tocsr()
    del full
    if G.diagonal().any() or (G != G.T).nnz or np.any(G.data <= 0):
        raise ValueError("Expected a symmetric, positive-length local graph without self-loops.")
    count, labels = connected_components(G, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.argmax())
    local_rows = np.flatnonzero(labels == largest)
    source_rows = roi_rows[local_rows]
    _, inverse = np.unique(ijk[source_rows], axis=0, return_inverse=True)
    unique_voxels = int(inverse.max()+1)
    sparse.save_npz(out / "within_roi_graph.npz", G)
    membership = pd.DataFrame(dict(graph_node_row=roi_rows, base_node_id=ids[roi_rows],
        fiber_component=components[roi_rows], within_roi_component=labels,
        within_roi_degree=np.diff(G.indptr), in_largest=(labels == largest),
        fdt_x_mm=world[roi_rows, 0], fdt_y_mm=world[roi_rows, 1], fdt_z_mm=world[roi_rows, 2]))
    membership.to_csv(out / "roi_node_membership.csv", index=False)
    pd.DataFrame(dict(component_id=np.arange(count), nodes=sizes)).sort_values(
        "nodes", ascending=False).to_csv(out / "within_roi_components.csv", index=False)
    qc = pd.DataFrame([dict(roi_id=roi_id, graph=graph, mask_voxels=mask_voxels,
        roi_nodes=len(roi_rows), roi_voxels=len(np.unique(ijk[roi_rows], axis=0)),
        roi_edges=G.nnz//2, roi_components=count, roi_isolates=int((np.diff(G.indptr) == 0).sum()),
        largest_nodes=len(source_rows), largest_voxels=unique_voxels,
        largest_node_fraction=len(source_rows)/len(roi_rows))])
    qc.to_csv(out / "roi_coverage.csv", index=False)
    print(f"Fixed ROI {roi_id}: {roi_path}")
    print("Distances use only edges between ROI-assigned nodes; paths cannot use outside nodes.")
    print("\nROI coverage:\n" + qc.to_string(index=False))
    protocol = dict(roi_id=roi_id, graph=graph, roi_file=str(roi_path),
        roi_sha256=hashlib.sha256(roi_path.read_bytes()).hexdigest(),
        membership="Nearest voxel center in the saved fdt grid; nonzero support of CC/roi_ID",
        mask_scope="Previously identified GM-masked regional ROI; no mask dilation",
        routing="Induced ROI subgraph; select its largest component before fitting",
        landmark_seed=1801, landmark_max=int(max_landmarks), dimension=3,
        curvature_scaled=CURVATURE, curvature_source="Frozen from the earlier whole-brain model; not optimized for this ROI",
        split_seeds=SPLIT_SEEDS, fit_fraction=.8, starts=2, maxiter=600,
        interpretation="Exploratory within-ROI component comparison in one brain; no optimal-curvature or biological-hyperbolicity claim")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if unique_voxels < MIN_LANDMARKS:
        print(f"No embedding fitted: largest within-ROI component has fewer than {MIN_LANDMARKS} distinct voxels.")
        print("Coverage was saved. No ROI, graph, or routing rule was changed to obtain a fit.")
        return qc, pd.DataFrame()

    rng = np.random.default_rng(1801)
    chosen_voxels = rng.choice(unique_voxels, min(max_landmarks, unique_voxels), replace=False)
    chosen_in_lcc = np.array([rng.choice(np.flatnonzero(inverse == v)) for v in chosen_voxels])
    selected_local = local_rows[chosen_in_lcc]
    selected_global = roi_rows[selected_local]
    landmark_ids, xyz = ids[selected_global], world[selected_global]
    n = len(landmark_ids)
    pd.DataFrame(dict(landmark_id=np.arange(n), base_node_id=landmark_ids,
        graph_node_row=selected_global, within_roi_node_row=selected_local,
        fiber_component=components[selected_global],
        fdt_x_mm=xyz[:, 0], fdt_y_mm=xyz[:, 1], fdt_z_mm=xyz[:, 2])).to_csv(out / "landmarks.csv", index=False)
    D = np.empty((n, n))
    for start in range(0, n, 16):
        D[start:start+16] = dijkstra(G, directed=False, indices=selected_local[start:start+16])[:, selected_local]
    helper_path = Path(__file__).with_name("fiber_geometry_cell15.py")
    spec = importlib.util.spec_from_file_location("roi18_fitter", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.check_metric15(D)
    np.savez_compressed(out / "landmark_distances.npz", distance_mm=D, base_node_id=landmark_ids)
    i, j = np.triu_indices(n, 1)
    raw = D[i, j]
    rows, logs, records, prediction_tables, coord_tables, display_tables = [], [], {}, [], [], []
    print(f"Landmarks: {n}; pairs: {len(i)}; E3 versus H3 with frozen k={CURVATURE:g}.")
    print("Three pair splits reuse these same landmarks and the same regional component.")
    for seed in SPLIT_SEEDS:
        train, test = np.split(np.random.default_rng(seed).permutation(len(i)), [int(.8*len(i))])
        scale = float(np.median(raw[train]))
        target = raw/scale
        starts = helper.starts15(n, i[train], j[train], target[train], seed)
        np.savez_compressed(out / f"pair_split_{seed}.npz", i=i, j=j, train=train, test=test)
        chosen = np.random.default_rng(1802).choice(train, min(96, len(train)), replace=False)
        a, b = i[chosen], j[chosen]
        display_tables.append(pd.DataFrame(dict(split_seed=seed, pair_index=chosen,
            landmark_a=a, landmark_b=b, split="fitting")))
        record = dict(a=a, b=b, scores={})
        for kind, k in (("euclidean", 0.), ("hyperbolic", CURVATURE)):
            context = dict(roi_id=roi_id, graph=graph, split_seed=seed)
            Z, loss, converged, all_ok = helper.fit15(starts, kind, k, i[train], j[train],
                                                    target[train], context, logs, 600)
            pred = helper.pair15(Z, kind, k, i[test], j[test])[0]
            error = pred-target[test]
            numerical_error = 0.
            if kind == "hyperbolic":
                P, L = ball18(Z, k)
                norm2 = np.sum(P*P, axis=1)
                gap = (1-norm2[i[test]])*(1-norm2[j[test]])
                if np.any(norm2 >= 1) or np.any(gap <= 0):
                    raise ValueError("Invalid ball projection; inspect this fit before interpreting it.")
                ball_distance = 2/np.sqrt(-k)*np.arcsinh(np.linalg.norm(P[i[test]]-P[j[test]], axis=1)/np.sqrt(gap))
                numerical_error = float(np.max(np.abs(ball_distance-pred)))
                record.update(P=P, curves=arcs18(L, a, b))
            else:
                record["E"] = Z*scale
            score = float(np.sqrt(np.mean(error**2)))
            numerical_ok = bool(np.isfinite(pred).all() and numerical_error <= 1e-6)
            fit_ok = bool(converged and numerical_ok)
            record["scores"][kind] = (score, fit_ok)
            rows.append(dict(**context, model=kind, curvature_scaled=k, landmarks=n,
                train_pairs=len(train), test_pairs=len(test), train_scale_mm=scale,
                train_nRMSE=np.sqrt(loss), test_nRMSE=score, test_RMSE_mm=score*scale,
                test_MAE_mm=np.mean(np.abs(error))*scale, selected_converged=converged,
                all_starts_converged=all_ok, numerical_error=numerical_error, fit_ok=fit_ok))
            prediction_tables.append(pd.DataFrame(dict(**context, model=kind, pair_index=test,
                landmark_a=i[test], landmark_b=j[test], observed_mm=raw[test], predicted_mm=pred*scale)))
            np.savez_compressed(out / f"fit_{seed}_{kind}.npz", coordinates=Z, model=kind,
                curvature_scaled=k, train_scale_mm=scale, base_node_id=landmark_ids)
        records[seed] = record
        coord_tables.append(pd.DataFrame(dict(split_seed=seed, base_node_id=landmark_ids,
            fdt_x_mm=xyz[:, 0], fdt_y_mm=xyz[:, 1], fdt_z_mm=xyz[:, 2],
            E_x_mm=record["E"][:, 0], E_y_mm=record["E"][:, 1], E_z_mm=record["E"][:, 2],
            H_ball_x=record["P"][:, 0], H_ball_y=record["P"][:, 1], H_ball_z=record["P"][:, 2])))
        print(f"Split {seed}: E={rows[-2]['test_nRMSE']:.4f}; H={rows[-1]['test_nRMSE']:.4f}; "
              f"fit_ok={rows[-2]['fit_ok']}/{rows[-1]['fit_ok']}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "fit_results.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "optimizer_runs.csv", index=False)
    pd.concat(prediction_tables).to_csv(out / "test_pair_predictions.csv", index=False)
    pd.concat(coord_tables).to_csv(out / "embedding_coordinates.csv", index=False)
    pd.concat(display_tables).to_csv(out / "display_pairs.csv", index=False)
    comparison = scores.pivot(index="split_seed", columns="model", values="test_nRMSE").reset_index()
    comparison["E_minus_H"] = comparison.euclidean-comparison.hyperbolic
    comparison["relative_H_reduction"] = comparison.E_minus_H/comparison.euclidean
    comparison["both_fits_ok"] = scores.groupby("split_seed").fit_ok.all().reindex(comparison.split_seed).to_numpy()
    comparison.to_csv(out / "paired_comparison.csv", index=False)

    # Display every split, with the same landmark pairs in all three columns.
    fig = plt.figure(figsize=(13.4, 12.7), constrained_layout=True)
    color_norm, cmap = Normalize(xyz[:, 2].min(), xyz[:, 2].max()), plt.get_cmap("viridis")
    anatomy_limits = limits18(world[roi_rows])
    euclidean_limits = limits18(np.concatenate([r["E"] for r in records.values()]))
    context_xyz = world[roi_rows[::max(1, int(np.ceil(len(roi_rows)/3000)))]]
    u, v = np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
              np.outer(np.ones_like(u), np.cos(v)))
    axes = []
    for row, seed in enumerate(SPLIT_SEEDS):
        record = records[seed]
        for col, points in enumerate((xyz, record["E"], record["P"])):
            ax = fig.add_subplot(3, 3, row*3+col+1, projection="3d")
            axes.append(ax)
            a, b = record["a"], record["b"]
            segments = record["curves"] if col == 2 else np.stack((points[a], points[b]), axis=1)
            ax.add_collection3d(Line3DCollection(segments, colors="#64748b", linewidths=.55, alpha=.23))
            ax.scatter(*points.T, c=xyz[:, 2], cmap=cmap, norm=color_norm, s=20,
                       edgecolors="white", linewidths=.3, depthshade=False)
            if col == 0:
                ax.scatter(*context_xyz.T, c="0.65", s=2, alpha=.13, depthshade=False)
                lo, hi = anatomy_limits
                title = "ROI anatomical XYZ"
            elif col == 1:
                lo, hi = euclidean_limits
                title = f"E3 | test nRMSE={record['scores']['euclidean'][0]:.4f}"
            else:
                ax.plot_wireframe(*sphere, color="0.65", linewidth=.35, alpha=.18)
                lo, hi = -np.ones(3), np.ones(3)
                title = f"H3 (k={CURVATURE:g}) | test nRMSE={record['scores']['hyperbolic'][0]:.4f}"
            if col > 0 and not record["scores"][("euclidean", "hyperbolic")[col-1]][1]:
                title += " | FIT FLAGGED"
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                   title=f"{title}\nPair split {seed}")
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-55)
            ax.set_proj_type("ortho")
            ax.set_axis_off()
    fig.colorbar(ScalarMappable(norm=color_norm, cmap=cmap), ax=axes, shrink=.6, pad=.01,
                 label="Anatomical z in fdt_paths world space (mm)")
    fig.suptitle(f"ROI {roi_id} | {graph} | {n} landmarks from the largest within-ROI component\n"
        f"Component: {len(source_rows)}/{len(roi_rows)} ROI nodes. Lines show landmark-pair geometry, not fiber trajectories.")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"roi_xyz_E3_H3.{ext}", dpi=220, bbox_inches="tight")
    plt.show()
    fig, ax = plt.subplots(figsize=(5.6, 4), constrained_layout=True)
    for r in comparison.itertuples():
        if r.both_fits_ok:
            ax.plot((0, 1), (r.euclidean, r.hyperbolic), "o-", label=f"split {r.split_seed}")
    ax.set(xticks=(0, 1), xticklabels=("Euclidean E3", "Hyperbolic H3"),
           ylabel="Held-out nRMSE", title=f"ROI {roi_id}: fixed-geometry comparison")
    ax.spines[["top", "right"]].set_visible(False)
    if comparison.both_fits_ok.any():
        ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"roi_prediction_comparison.{ext}", dpi=220)
    plt.show()
    (out / "README.txt").write_text(
        "Fixed ROI ID chosen before inspecting its model comparison; no search over regions.\n"
        "The input CC mask was previously identified as GM-masked atlas support.\n"
        "Nearest-center membership and selection of the largest component restrict the scope.\n"
        "Distances use the full within-ROI graph, including intermediate nodes not selected as landmarks.\n"
        "Grey anatomical points show other ROI-assigned nodes, including other components; only colored landmarks are fitted.\n"
        "E3 and H3 use identical landmarks, fitting/test pairs, two starts, and optimizer settings.\n"
        "Fixed normalized curvature -4 is carried over from whole-brain analysis, not optimized for this region.\n"
        "Three pair splits reuse the same brain, component, and nodes; they are not independent biological replicates.\n"
        "Drawn lines are anatomical chords or model geodesics, not observed graph edges or axons.\n"
        "Any fit flag must be resolved before interpreting its comparison.\n", encoding="utf-8")
    print("\nPaired results (descriptive; positive E_minus_H favors H):\n" + comparison.to_string(index=False))
    print(f"Optimizer convergence: {sum(r['success'] for r in logs)}/{len(logs)}")
    if not scores.fit_ok.all():
        print("WARNING: Resolve flagged fits before interpreting model differences.")
    print("Fixed regional component and fixed curvature; no claim about all ROI tissue or optimal regional geometry.")
    print(f"Saved: {out}")
    return qc, scores


if __name__ == "__main__":
    roi_coverage18, roi_results18 = run_cell18(PROJECT, ROI_ID, GRAPH, MAX_LANDMARKS)
