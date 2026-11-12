# Cell 20: All 120 within-ROI E3/H3 comparisons and horizontal figures.
# Same fitting protocol as Cell 18; the plot style follows Cell 19.
# All regions are reported, including E wins, insufficient coverage and failures.
# No FSL, new tractography, curvature search, or outcome-based region selection.
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import hashlib
import itertools
import json
import platform
import shutil
import traceback
import numpy as np
import pandas as pd
import nibabel as nib
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components, dijkstra
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROJECT = "/mnt/g/Research/TUM_connectome/HyperbolicTest"
ROI_IDS = tuple(range(1, 121))
GRAPH = "supported"
MAX_LANDMARKS, MIN_LANDMARKS = 128, 16
SPLIT_SEEDS = (1807, 1817, 1827)
DISPLAY_SPLIT = 1807
CURVATURE = -4.0
NODE_COLOR = "#2475A9"
FRAME_COLOR = "#EEF0F2"
RESUME = True
PROTOCOL_VERSION = "cell20-1.0"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, table):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True).encode()).hexdigest()


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


def unit_ball(Z, k=CURVATURE):
    radius = 1 / np.sqrt(-k)
    time = np.sqrt(radius * radius + np.sum(Z * Z, axis=1))
    return Z / (time[:, None] + radius)


def ball_arcs(P, a, b):
    norm2 = np.sum(P * P, axis=1)
    if not np.isfinite(P).all() or np.any(norm2 >= 1):
        raise ValueError("Invalid Poincare ball coordinates.")
    L = np.column_stack((1 + norm2, 2 * P)) / (1 - norm2[:, None])
    t = np.linspace(0, 1, 64)[None, :, None]
    Q = (1 - t) * L[a, None, :] + t * L[b, None, :]
    norm = np.sqrt(Q[..., 0] ** 2 - np.sum(Q[..., 1:] ** 2, axis=-1))
    return Q[..., 1:] / (Q[..., 0] + norm)[..., None]


def box_segments(low, high):
    # Explicitly draw all 12 edges, including the left rear vertical edge.
    corners = np.array(list(itertools.product((0, 1), repeat=3)), dtype=int)
    points = low + corners * (high - low)
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(corners[i] != corners[j]) == 1]
    return np.array([[points[i], points[j]] for i, j in pairs])


def save_visible(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [fig.bbox]
    for ax in fig.axes:
        boxes.append(ax.get_tightbbox(renderer))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            # Do not re-query tick labels after 3D tight-bbox projection.
            for artist in (axis.label, axis.offsetText):
                if artist.get_visible() and artist.get_text():
                    boxes.append(artist.get_window_extent(renderer))
    boxes.extend(t.get_window_extent(renderer) for t in fig.texts)
    bounds = Bbox.union([b for b in boxes if b is not None])
    bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(.16)
    for suffix, dpi in ((".png", 200), ("_600dpi.png", 600), (".pdf", 200)):
        fig.savefig(str(stem) + suffix, dpi=dpi, bbox_inches=bounds, facecolor="white")


def draw_row(out, context, xyz=None, record=None, reason="Not fitted"):
    # Fixed display split, viewpoint and pair selection; no best-looking selection.
    context = np.asarray(context, dtype=float).reshape(-1, 3)
    cloud = context[::max(1, int(np.ceil(len(context) / 3000)))]
    xyz = np.empty((0, 3)) if xyz is None else np.asarray(xyz, dtype=float)
    record = {} if record is None else record
    scores = record.get("scores", {})
    arrays = (xyz, record.get("E"), record.get("P"))
    captions = ["Anatomical XYZ"]
    for kind, name in (("euclidean", r"Euclidean $E^3$ fit"),
                       ("hyperbolic", r"Hyperbolic $H^3$ fit")):
        score, ok = scores.get(kind, (np.nan, False))
        label = f"{name} | nRMSE = {score:.4f}" if np.isfinite(score) else f"{name} | not fitted"
        if np.isfinite(score) and not ok:
            label += " | flagged"
        captions.append(label)
    a = np.asarray(record.get("a", []), dtype=int)
    b = np.asarray(record.get("b", []), dtype=int)
    theta, phi = np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (np.outer(np.cos(theta), np.sin(phi)), np.outer(np.sin(theta), np.sin(phi)),
              np.outer(np.ones_like(theta), np.cos(phi)))
    style = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
             "figure.autolayout": False, "figure.constrained_layout.use": False,
             "savefig.bbox": None, "text.color": "black", "axes.labelcolor": "black"}
    with matplotlib.rc_context(style):
        width, height = 1904, 626
        fig = Figure(figsize=(width / 200, height / 200), dpi=200, facecolor="white")
        FigureCanvasAgg(fig)
        try:
            side = .70 * 626
            for col, points in enumerate(arrays):
                center_x = (col + .5) * width / 3
                ax = fig.add_axes([(center_x - .04 * 626 - side / 2) / width,
                                   .24 * 626 / height, side / width, side / height], projection="3d")
                fig.text(center_x / width, .045 * 626 / height, captions[col],
                         ha="center", va="center", fontsize=8.5)
                available = points is not None and len(points) > 0 and np.isfinite(points).all()
                if col == 0 and len(cloud):
                    ax.scatter(*cloud.T, color=NODE_COLOR, s=1.7, alpha=.12,
                               linewidths=0, depthshade=False, rasterized=False)
                if available:
                    if len(a):
                        segments = (ball_arcs(points, a, b) if col == 2 else
                                    np.stack((points[a], points[b]), axis=1))
                        ax.add_collection3d(Line3DCollection(segments, colors="#64748b",
                                                             linewidths=.45, alpha=.27))
                    ax.scatter(*points.T, color=NODE_COLOR, s=16, edgecolors="white",
                               linewidths=.25, depthshade=False, rasterized=False)
                elif col > 0 or not len(context):
                    ax.text2D(.5, .55, reason, transform=ax.transAxes, ha="center",
                              va="center", fontsize=7, color="#666666", wrap=True)
                if col == 2:
                    if available:
                        ax.plot_wireframe(*sphere, color="#b7c1cc", linewidth=.3, alpha=.36)
                    low, high = -np.ones(3), np.ones(3)
                else:
                    extent = context if col == 0 else points
                    if extent is None or not len(extent):
                        low, high = -np.ones(3), np.ones(3)
                    else:
                        middle = (extent.min(0) + extent.max(0)) / 2
                        half = max(float(np.ptp(extent, axis=0).max()) * .56, 1e-6)
                        low, high = middle - half, middle + half
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
                ax.set_box_aspect((1, 1, 1))
                ax.set_proj_type("ortho")
                ax.view_init(elev=20, azim=-55)
                ax.grid(False)
                ax.add_collection3d(Line3DCollection(box_segments(low, high),
                                                     colors=FRAME_COLOR, linewidths=.5, alpha=1.0))
                labels = ("X (mm)", "Y (mm)", "Z (mm)") if col < 2 else (r"$p_x$", r"$p_y$", r"$p_z$")
                for dim, (axis, label) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), labels)):
                    axis.set_label_text(label, fontsize=8)
                    axis.labelpad = 5
                    axis.pane.fill = False
                    axis.pane.set_edgecolor((1, 1, 1, 0))
                    axis.pane.set_linewidth(0)
                    axis.line.set_color("#888888")
                    axis.line.set_linewidth(.45)
                    ticks = (np.array([-1, 0, 1]) if col == 2 else
                             MaxNLocator(nbins=3).tick_values(low[dim], high[dim]))
                    axis.set_ticks(ticks[(ticks >= low[dim]) & (ticks <= high[dim])])
                ax.tick_params(labelsize=6.5, pad=0)
            save_visible(fig, out / "roi_xyz_E3_H3_row_labeled")
        finally:
            fig.clear()
    write_json(out / "display_notes.json", dict(
        split_seed=DISPLAY_SPLIT, captions=captions, node_color=NODE_COLOR,
        frame_edges=12, frame_color=FRAME_COLOR, vector_artists=True,
        context_points_drawn=len(cloud), context_nodes_available=len(context),
        context_sampling="Stride through all ROI graph nodes; display only, not a fitting sample",
        anatomical_units="Saved fdt/T1 reference world XYZ in mm",
        euclidean_units="Graph-distance mm; fitted axes are not anatomical axes",
        hyperbolic_units="Dimensionless Cartesian Poincare unit-ball p_x, p_y, p_z",
        lines="Same sampled training-pair chords in anatomical/E; model geodesics in H; not streamlines",
        score="Held-out nRMSE for the displayed split, not the three-split mean",
        selection="Split 1807 is fixed before fitting; no outcome-based display selection"))


def load_inputs(root, graph):
    src = root / "results/cell04_orientation_graph"
    exp = root / "results/cell14c_fdt_space"
    paths = {"nodes": src / f"nodes_{graph}.npz",
             "graph": src / f"edge_lengths_{graph}.npz",
             "transforms": exp / "coordinate_transforms.npz",
             "display_nodes": exp / f"nodes_{graph}_in_fdt.npz"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required saved input is missing: {path}")
    with np.load(paths["nodes"]) as z:
        ids, ijk, components = (z[k].copy() for k in ("base_node_id", "ijk", "component"))
    with np.load(paths["transforms"]) as z:
        shape, affine = tuple(z["fdt_shape"].astype(int)), z["fdt_affine"].copy()
    with np.load(paths["display_nodes"]) as z:
        if not np.array_equal(ids, z["base_node_id"]):
            raise ValueError("Native and exported node orders disagree.")
        vox, world = z["fdt_voxel_xyz"].copy(), z["fdt_world_xyz_mm"].copy()
    if (len(shape) != 3 or affine.shape != (4, 4) or len(np.unique(ids)) != len(ids)
            or ijk.shape != (len(ids), 3) or components.shape != (len(ids),)
            or vox.shape != (len(ids), 3) or world.shape != (len(ids), 3)
            or not all(np.isfinite(x).all() for x in (affine, ijk, vox, world))):
        raise ValueError("Invalid saved node or coordinate tables.")
    full = sparse.load_npz(paths["graph"]).tocsr().astype(float)
    full.sum_duplicates()
    full.eliminate_zeros()
    if (full.shape != (len(ids), len(ids)) or full.diagonal().any() or (full != full.T).nnz
            or np.any(full.data <= 0) or not np.isfinite(full.data).all()):
        raise ValueError("Expected symmetric positive-length graph without self-loops.")
    rounded = np.floor(vox + .5).astype(np.int64)
    inside = np.all((rounded >= 0) & (rounded < np.array(shape)), axis=1)
    return dict(ids=ids, ijk=ijk, components=components, world=world,
                shape=shape, affine=affine, rounded=rounded, inside=inside, full=full), {
                    name: dict(path=str(path), sha256=sha256_file(path)) for name, path in paths.items()}


def region_graph(inputs, roi_path, roi_id, graph, out):
    image = nib.load(roi_path)
    if image.shape != inputs["shape"] or not np.allclose(image.affine, inputs["affine"], atol=1e-4, rtol=0):
        raise ValueError("ROI grid differs from the saved fdt grid. No transform was guessed.")
    mask = np.asarray(image.dataobj)
    if not np.isfinite(mask).all():
        raise ValueError("Nonfinite ROI mask values.")
    inside, vox = inputs["inside"], inputs["rounded"]
    belongs = np.zeros(len(inside), dtype=bool)
    belongs[inside] = mask[tuple(vox[inside].T)] != 0
    roi_rows = np.flatnonzero(belongs)
    G = inputs["full"][roi_rows][:, roi_rows].tocsr()
    sparse.save_npz(out / "within_roi_graph.npz", G)
    if len(roi_rows):
        count, labels = connected_components(G, directed=False)
        sizes = np.bincount(labels)
        largest = int(sizes.argmax())
        local_rows = np.flatnonzero(labels == largest)
        source_rows = roi_rows[local_rows]
        _, inverse = np.unique(inputs["ijk"][source_rows], axis=0, return_inverse=True)
        unique_voxels = int(inverse.max() + 1)
    else:
        count, largest, unique_voxels = 0, -1, 0
        labels = sizes = local_rows = source_rows = inverse = np.empty(0, dtype=int)
    context = inputs["world"][roi_rows]
    coverage = dict(roi_id=roi_id, graph=graph, mask_voxels=int(np.count_nonzero(mask)),
                    roi_nodes=len(roi_rows), roi_voxels=len(np.unique(inputs["ijk"][roi_rows], axis=0)),
                    roi_edges=G.nnz // 2, roi_components=count,
                    roi_isolates=int((np.diff(G.indptr) == 0).sum()),
                    largest_component_id=largest, largest_nodes=len(source_rows), largest_voxels=unique_voxels,
                    largest_node_fraction=len(source_rows) / len(roi_rows) if len(roi_rows) else np.nan)
    coverage["largest_voxel_fraction"] = unique_voxels / coverage["roi_voxels"] if coverage["roi_voxels"] else np.nan
    write_csv(out / "roi_coverage.csv", pd.DataFrame([coverage]))
    write_csv(out / "roi_node_membership.csv", pd.DataFrame(dict(
        graph_node_row=roi_rows, base_node_id=inputs["ids"][roi_rows],
        fiber_component=inputs["components"][roi_rows], within_roi_component=labels,
        within_roi_degree=np.diff(G.indptr), in_largest=(labels == largest),
        fdt_x_mm=context[:, 0], fdt_y_mm=context[:, 1], fdt_z_mm=context[:, 2])))
    write_csv(out / "within_roi_components.csv", pd.DataFrame(
        dict(component_id=np.arange(count), nodes=sizes)).sort_values("nodes", ascending=False))
    return G, roi_rows, local_rows, inverse, context, coverage


def fit_region(inputs, G, roi_rows, local_rows, inverse, out, roi_id, graph, max_landmarks):
    unique_voxels = int(inverse.max() + 1)
    rng = np.random.default_rng(1801)
    chosen = rng.choice(unique_voxels, min(max_landmarks, unique_voxels), replace=False)
    chosen_nodes = np.array([rng.choice(np.flatnonzero(inverse == v)) for v in chosen])
    selected_local = local_rows[chosen_nodes]
    selected_global = roi_rows[selected_local]
    ids, xyz = inputs["ids"][selected_global], inputs["world"][selected_global]
    n = len(ids)
    print(f"    ROI nodes/edges: {G.shape[0]:,}/{G.nnz // 2:,}; "
          f"landmarks: {n}/{unique_voxels} largest-component voxels "
          f"({100 * n / unique_voxels:.2f}%).", flush=True)
    write_csv(out / "landmarks.csv", pd.DataFrame(dict(
        landmark_id=np.arange(n), base_node_id=ids, graph_node_row=selected_global,
        within_roi_node_row=selected_local, fiber_component=inputs["components"][selected_global],
        fdt_x_mm=xyz[:, 0], fdt_y_mm=xyz[:, 1], fdt_z_mm=xyz[:, 2])))
    D = np.empty((n, n))
    for start in range(0, n, 16):
        D[start:start + 16] = dijkstra(G, directed=False, indices=selected_local[start:start + 16])[:, selected_local]
    check_metric15(D)
    np.savez_compressed(out / "landmark_distances.npz", distance_mm=D, base_node_id=ids)
    i, j = np.triu_indices(n, 1)
    raw = D[i, j]
    rows, logs, predictions, coordinates, display_pairs, comparisons = [], [], [], [], [], []
    display_record = None
    for seed in SPLIT_SEEDS:
        train, test = np.split(np.random.default_rng(seed).permutation(len(i)), [int(.8 * len(i))])
        scale = float(np.median(raw[train]))
        target = raw / scale
        starts = starts15(n, i[train], j[train], target[train], seed)
        np.savez_compressed(out / f"pair_split_{seed}.npz", i=i, j=j, train=train, test=test)
        chosen_pairs = np.random.default_rng(1802).choice(train, min(96, len(train)), replace=False)
        a, b = i[chosen_pairs], j[chosen_pairs]
        display_pairs.append(pd.DataFrame(dict(split_seed=seed, pair_index=chosen_pairs,
                                               landmark_a=a, landmark_b=b, split="fitting")))
        record = dict(a=a, b=b, scores={})
        split_rows = {}
        for kind, k in (("euclidean", 0.), ("hyperbolic", CURVATURE)):
            context = dict(roi_id=roi_id, graph=graph, split_seed=seed)
            row = dict(**context, model=kind, curvature_scaled=k, landmarks=n,
                       train_pairs=len(train), test_pairs=len(test), train_scale_mm=scale,
                       train_nRMSE=np.nan, test_nRMSE=np.nan, test_RMSE_mm=np.nan,
                       test_MAE_mm=np.nan, selected_converged=False, all_starts_converged=False,
                       numerical_ok=False, numerical_error=np.nan, fit_ok=False, error="")
            try:
                Z, loss, converged, all_ok = fit15(starts, kind, k, i[train], j[train],
                                                  target[train], context, logs, 600)
                np.savez_compressed(out / f"fit_{seed}_{kind}.npz", coordinates=Z, model=kind,
                                    curvature_scaled=k, train_scale_mm=scale, base_node_id=ids)
                pred = pair15(Z, kind, k, i[test], j[test])[0]
                error = pred - target[test]
                numerical_error = 0.
                if kind == "hyperbolic":
                    P = unit_ball(Z, k)
                    norm2 = np.sum(P * P, axis=1)
                    gap = (1 - norm2[i[test]]) * (1 - norm2[j[test]])
                    if np.any(norm2 >= 1) or np.any(gap <= 0) or not np.isfinite(P).all():
                        raise ValueError("Invalid ball projection.")
                    ball_distance = 2 / np.sqrt(-k) * np.arcsinh(
                        np.linalg.norm(P[i[test]] - P[j[test]], axis=1) / np.sqrt(gap))
                    numerical_error = float(np.max(np.abs(ball_distance - pred)))
                    record["P"] = P
                else:
                    record["E"] = Z * scale
                score = float(np.sqrt(np.mean(error ** 2)))
                numerical_ok = bool(np.isfinite(pred).all() and np.isfinite(numerical_error) and numerical_error <= 1e-6)
                row.update(train_nRMSE=float(np.sqrt(loss)), test_nRMSE=score,
                           test_RMSE_mm=score * scale, test_MAE_mm=float(np.mean(np.abs(error)) * scale),
                           selected_converged=converged, all_starts_converged=all_ok,
                           numerical_ok=numerical_ok, numerical_error=numerical_error,
                           fit_ok=bool(converged and numerical_ok))
                predictions.append(pd.DataFrame(dict(**context, model=kind, pair_index=test,
                    landmark_a=i[test], landmark_b=j[test], base_node_a=ids[i[test]], base_node_b=ids[j[test]],
                    observed_mm=raw[test], predicted_mm=pred * scale, residual_mm=error * scale)))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                (out / f"fit_error_{seed}_{kind}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            record["scores"][kind] = (row["test_nRMSE"], row["fit_ok"])
            split_rows[kind] = row
            rows.append(row)
        E = record.get("E", np.full((n, 3), np.nan))
        P = record.get("P", np.full((n, 3), np.nan))
        coordinates.append(pd.DataFrame(dict(split_seed=seed, base_node_id=ids,
            fdt_x_mm=xyz[:, 0], fdt_y_mm=xyz[:, 1], fdt_z_mm=xyz[:, 2],
            E_x_mm=E[:, 0], E_y_mm=E[:, 1], E_z_mm=E[:, 2], H_ball_x=P[:, 0], H_ball_y=P[:, 1], H_ball_z=P[:, 2])))
        er, hr = split_rows["euclidean"], split_rows["hyperbolic"]
        gain = er["test_nRMSE"] - hr["test_nRMSE"]
        comparisons.append(dict(roi_id=roi_id, graph=graph, split_seed=seed,
            E_test_nRMSE=er["test_nRMSE"], H_test_nRMSE=hr["test_nRMSE"], E_minus_H=gain,
            relative_H_reduction=gain / er["test_nRMSE"] if er["test_nRMSE"] > 0 else np.nan,
            E_test_RMSE_mm=er["test_RMSE_mm"], H_test_RMSE_mm=hr["test_RMSE_mm"],
            E_test_MAE_mm=er["test_MAE_mm"], H_test_MAE_mm=hr["test_MAE_mm"],
            both_fits_ok=bool(er["fit_ok"] and hr["fit_ok"])))
        if seed == DISPLAY_SPLIT:
            display_record = record
        # Save after each split so an interruption still leaves reviewable data.
        write_csv(out / "fit_results.csv", pd.DataFrame(rows))
        write_csv(out / "optimizer_runs.csv", pd.DataFrame(logs))
        if predictions:
            write_csv(out / "test_pair_predictions.csv", pd.concat(predictions, ignore_index=True))
        write_csv(out / "embedding_coordinates.csv", pd.concat(coordinates, ignore_index=True))
        write_csv(out / "display_pairs.csv", pd.concat(display_pairs, ignore_index=True))
        write_csv(out / "paired_comparison.csv", pd.DataFrame(comparisons))
        print(f"    Split {seed}: E={er['test_nRMSE']:.5f}; H={hr['test_nRMSE']:.5f}; "
              f"fit_ok={er['fit_ok']}/{hr['fit_ok']}", flush=True)
    return xyz, display_record, pd.DataFrame(comparisons)


def summary_template(roi_id, graph):
    return dict(roi_id=roi_id, graph=graph, status="pending", reason="", landmarks=0,
                valid_splits=0, all_three_splits_valid=False, mean_E_test_nRMSE=np.nan,
                mean_H_test_nRMSE=np.nan, mean_E_minus_H=np.nan, mean_relative_H_reduction=np.nan,
                mean_E_test_RMSE_mm=np.nan, mean_H_test_RMSE_mm=np.nan,
                mean_E_test_MAE_mm=np.nan, mean_H_test_MAE_mm=np.nan,
                sd_split_E_minus_H=np.nan, H_better_splits=0, comparison="not_evaluable",
                landmark_fraction_of_largest_voxels=np.nan, landmark_fraction_of_roi_voxels=np.nan)


def analyze_region(inputs, roi_path, roi_id, graph, out, max_landmarks):
    summary = summary_template(roi_id, graph)
    if not roi_path.is_file():
        summary.update(status="missing_roi", reason=f"Missing {roi_path.name}")
        draw_row(out, [], reason="ROI file missing")
        return summary
    G, roi_rows, local_rows, inverse, context, coverage = region_graph(inputs, roi_path, roi_id, graph, out)
    summary.update(coverage)
    if coverage["largest_voxels"] < MIN_LANDMARKS:
        summary.update(status="insufficient_component",
                       reason=f"Largest within-ROI component has {coverage['largest_voxels']} distinct voxels; minimum is {MIN_LANDMARKS}.")
        draw_row(out, context, reason=f"Not fitted\nFewer than {MIN_LANDMARKS} connected voxels")
        return summary
    xyz, record, comparison = fit_region(inputs, G, roi_rows, local_rows, inverse, out, roi_id, graph, max_landmarks)
    summary["landmarks"] = len(xyz)
    summary["landmark_fraction_of_largest_voxels"] = len(xyz) / coverage["largest_voxels"]
    summary["landmark_fraction_of_roi_voxels"] = len(xyz) / coverage["roi_voxels"]
    write_csv(out / "landmark_sampling_summary.csv", pd.DataFrame([
        dict(reference=label, landmarks=len(xyz), denominator=coverage[key],
             sampled_percent=100 * len(xyz) / coverage[key])
        for label, key in (("Largest-component voxels", "largest_voxels"),
                           ("All ROI graph voxels", "roi_voxels"), ("All ROI graph nodes", "roi_nodes"))]))
    valid = comparison[comparison.both_fits_ok]
    summary.update(valid_splits=len(valid), all_three_splits_valid=len(valid) == len(SPLIT_SEEDS),
                   H_better_splits=int((valid.E_minus_H > 0).sum()))
    # Main means require all three paired splits to pass; no selective averaging.
    if len(valid) == len(SPLIT_SEEDS):
        for key in ("E_test_nRMSE", "H_test_nRMSE", "E_minus_H", "relative_H_reduction",
                    "E_test_RMSE_mm", "H_test_RMSE_mm", "E_test_MAE_mm", "H_test_MAE_mm"):
            summary["mean_" + key] = float(comparison[key].mean())
        summary["sd_split_E_minus_H"] = float(comparison.E_minus_H.std(ddof=1))
        gain = summary["mean_E_minus_H"]
        summary.update(status="ok", comparison="H_lower" if gain > 0 else "E_lower" if gain < 0 else "equal")
    else:
        summary.update(status="fit_flagged", reason="At least one selected fit failed convergence or numerical checks; inspect per-split tables.")
    try:
        draw_row(out, context, xyz, record, reason="Fit unavailable\nSee fit_results.csv")
        summary["figure_ok"] = True
    except Exception:
        (out / "figure_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        summary["figure_ok"] = False
    return summary


def write_batch_summary(out, rows, roi_ids):
    table = pd.DataFrame(rows).sort_values("roi_id")
    write_csv(out / "roi_summary.csv", table)
    fit_tables, paired_tables = [], []
    for roi_id in table.roi_id.to_numpy(dtype=int):
        folder = out / f"roi_{roi_id:03d}"
        for filename, tables in (("fit_results.csv", fit_tables), ("paired_comparison.csv", paired_tables)):
            path = folder / filename
            if path.is_file():
                tables.append(pd.read_csv(path))
    if fit_tables:
        write_csv(out / "all_fit_results.csv", pd.concat(fit_tables, ignore_index=True))
    if paired_tables:
        write_csv(out / "all_paired_comparisons.csv", pd.concat(paired_tables, ignore_index=True))
    valid = table[table.all_three_splits_valid.fillna(False).astype(bool)]
    counts = table.status.value_counts().to_dict()
    report = dict(requested_ROIs=len(roi_ids), processed_ROIs=len(table), status_counts=counts,
                  evaluable_ROIs=len(valid), H_lower_ROIs=int((valid.comparison == "H_lower").sum()),
                  E_lower_ROIs=int((valid.comparison == "E_lower").sum()),
                  equal_ROIs=int((valid.comparison == "equal").sum()),
                  H_lower_fraction_of_evaluable=float((valid.comparison == "H_lower").mean()) if len(valid) else None,
                  median_ROI_mean_E_minus_H=float(valid.mean_E_minus_H.median()) if len(valid) else None,
                  interpretation="Descriptive comparison of within-ROI graph components in one brain. ROIs and pair splits are not independent subjects.")
    write_json(out / "batch_summary.json", report)
    return table, report


README = """Cell 20: batch within-ROI E3/H3 geometry comparison

Each requested ROI is included in roi_summary.csv, including missing masks,
insufficient connected coverage, fitting failures and Euclidean-favored results.
ROI files are CC/roi_ID.nii.gz; historical bin_roi masks are not used.
Nodes are assigned using their saved nearest fdt-space voxel center. Original
supported-graph lengths are used for analysis. Display coordinates do not replace
native graph lengths. Paths are constrained to the ROI-induced graph. Only the
largest within-ROI component supplies landmarks; fragmentation and coverage are
reported. This differs from whole-brain routing between endpoints inside an ROI.

Up to 128 distinct native DWI voxels are selected uniformly without replacement
using seed 1801, with one eligible orientation node selected per voxel. At least
16 distinct voxels are required. Full within-ROI graph nodes, including unsampled
intermediate nodes, participate in shortest-path calculation. Three 80/20 pair
splits (1807, 1817, 1827) reuse the same landmarks. Test pairs, not unseen nodes or
new subjects, are held out. Scale is the median training-pair distance in mm.

E3 and H3 use the exact Cell 15 fitting functions used by Cell 18: training-only
MDS and random initializations, L-BFGS-B with maxiter 600, minimum training loss
selection. H curvature is fixed at -4 in normalized distance units. There is no
per-ROI curvature tuning, dimension search or best-test-seed selection. Physical
curvature would be -4 / train_scale_mm**2. A failed minimum remains flagged.

test_nRMSE = test_RMSE_mm / median(training-pair distances_mm).
E_minus_H > 0 means a lower H test error; < 0 means a lower E test error.
relative_H_reduction = (E_test_nRMSE - H_test_nRMSE) / E_test_nRMSE.
Main ROI means and the better-model label require all three paired fits to pass.
Per-split raw results are saved even when a region is flagged. The mean relative
reduction is the mean of the three split-specific ratios, not a ratio of means.
Across-split SD is descriptive and is not a confidence interval.

Figure: anatomical XYZ, Euclidean E3, Poincare H3, one horizontal row. Display
split is always 1807. The displayed errors belong to that split, not the ROI
mean. All points are the same blue. Faint anatomy is a strided context cloud
(up to 3000 ROI graph nodes); bright points are the sampled landmarks. The same
up to 96 training pairs are drawn as anatomical/E chords and H geodesics. These
lines are not tractography streamlines or direct graph edges. Anatomical and E
axes use mm but E axes are not anatomical axes; Poincare axes are dimensionless
Cartesian ball coordinates. All 12 cube edges are drawn explicitly, lighter than
pair lines and sphere wireframe. Exports are 200-dpi preview, 600-dpi PNG and
vector PDF, with explicit margins for all labels. No vertical layout is made.
Unfittable regions get labeled placeholder panels, never invented embeddings.

Key files in each roi_NNN folder:
  roi_summary.csv                 One-row status, coverage and E/H means
  fit_results.csv                 Model/split RMSE, nRMSE, MAE and checks
  paired_comparison.csv           Paired E-H and relative reductions
  test_pair_predictions.csv       Observed/predicted distances for audit
  optimizer_runs.csv              Both starts, losses and convergence
  embedding_coordinates.csv       Anatomy, E3 and H3 for every split
  fit_SEED_MODEL.npz              Raw fitted coordinates and scale
  pair_split_SEED.npz             Exact train/test indices
  landmark_distances.npz          Full landmark distance matrix in mm
  landmarks.csv                  Landmark IDs and graph row mapping
  landmark_sampling_summary.csv   Actual sampling percentages
  roi_coverage.csv                All-component and largest-component coverage
  roi_node_membership.csv         Every assigned node and component
  within_roi_graph.npz            Induced CSR graph, mm edge lengths
  protocol.json / status.json     Input fingerprints, settings and run status
  display_notes.json              Plot units, captions and line meaning

Missing/unfittable regions have only outputs that can legitimately be produced.
Top-level summaries are updated after every ROI. Completed matching regions are
reused on rerun. Changed/incomplete region folders are moved to _previous before
retrying. Original input files and previous Cells 18/19 outputs are unchanged.

These results test distance representation of MRI-derived orientation graphs,
not individual neurons or a proof of biological hyperbolicity. A lower H error
is an empirical outcome, not an enforced objective. This is one-brain exploratory
evidence; shared anatomy, graph construction, reused data and partial graph
coverage limit generalization. No significance tests or independent-subject
claims are generated automatically.
"""


def run_cell20(project=PROJECT, roi_ids=ROI_IDS, graph=GRAPH,
               max_landmarks=MAX_LANDMARKS, resume=RESUME):
    root = Path(project).expanduser().resolve()
    roi_ids = tuple(int(x) for x in roi_ids)
    if not roi_ids or len(set(roi_ids)) != len(roi_ids) or any(x < 1 or x > 120 for x in roi_ids):
        raise ValueError("ROI IDs must be unique integers from 1 to 120.")
    if max_landmarks < MIN_LANDMARKS:
        raise ValueError("max_landmarks must be at least 16.")
    out = root / f"results/cell20_all120_rois_{graph}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading saved {graph} graph and fdt-space node assignments ...", flush=True)
    inputs, provenance = load_inputs(root, graph)
    settings = dict(protocol_version=PROTOCOL_VERSION, graph=graph, dimension=3,
                    curvature_scaled=CURVATURE, landmark_seed=1801, max_landmarks=int(max_landmarks),
                    min_landmarks=MIN_LANDMARKS, split_seeds=SPLIT_SEEDS, fit_fraction=.8,
                    display_split=DISPLAY_SPLIT, display_pair_seed=1802, starts=2, maxiter=600,
                    routing="Largest component of the ROI-induced graph; no outside-ROI paths",
                    normalization="Training-pair median distance in mm, separately for each split",
                    score_scope="Held-out pairs among the same landmarks; one-brain descriptive evidence")
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    pandas=pd.__version__, nibabel=nib.__version__, matplotlib=matplotlib.__version__)
    source = Path(__file__).resolve()
    common = dict(settings=settings, inputs=provenance, versions=versions, source_sha256=sha256_file(source))
    write_json(out / "batch_protocol.json", dict(**common, requested_roi_ids=roi_ids))
    (out / "README.txt").write_text(README, encoding="utf-8")
    shutil.copyfile(source, out / "source_cell20.py")
    rows, measured_times = [], []
    print(f"ROIs: {len(roi_ids)} | E3/H3 | fixed k={CURVATURE:g} | up to {max_landmarks} landmarks per ROI.")
    print("Three paired 80/20 splits; figures always show split 1807. No curvature search.")
    print("Each ROI saves its own results, horizontal preview/600-dpi PNG and vector PDF.", flush=True)
    for index, roi_id in enumerate(roi_ids, 1):
        begin = perf_counter()
        roi_path = root / f"CC/roi_{roi_id}.nii.gz"
        region_out = out / f"roi_{roi_id:03d}"
        protocol = dict(**common, roi_id=roi_id, roi_file=str(roi_path),
                        roi_sha256=sha256_file(roi_path) if roi_path.is_file() else None)
        fingerprint = signature(protocol)
        saved = None
        status_path = region_out / "status.json"
        if resume and status_path.is_file():
            try:
                previous = json.loads(status_path.read_text())
                if (previous.get("fingerprint") == fingerprint and previous.get("completed")
                        and all((region_out / f).is_file() and (region_out / f).stat().st_size == size
                                for f, size in previous.get("output_sizes", {}).items())):
                    saved = previous["summary"]
            except (OSError, ValueError, KeyError):
                saved = None
        if saved is not None:
            rows.append(saved)
            print(f"[{index:03d}/{len(roi_ids):03d}] ROI {roi_id:03d}: reused ({saved['status']}).", flush=True)
            table, report = write_batch_summary(out, rows, roi_ids)
            continue
        if region_out.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archived = out / "_previous" / f"roi_{roi_id:03d}_{stamp}"
            archived.parent.mkdir(exist_ok=True)
            shutil.move(str(region_out), str(archived))
        region_out.mkdir()
        write_json(region_out / "protocol.json", protocol)
        write_json(status_path, dict(fingerprint=fingerprint, completed=False, status="running"))
        print(f"[{index:03d}/{len(roi_ids):03d}] ROI {roi_id:03d}: analyzing ...", flush=True)
        try:
            summary = analyze_region(inputs, roi_path, roi_id, graph, region_out, max_landmarks)
        except Exception as exc:
            summary = summary_template(roi_id, graph)
            summary.update(status="error", reason=f"{type(exc).__name__}: {exc}", figure_ok=False)
            (region_out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            coverage_path = region_out / "roi_coverage.csv"
            if coverage_path.is_file():
                summary.update(pd.read_csv(coverage_path).iloc[0].to_dict())
            try:
                context = []
                membership_path = region_out / "roi_node_membership.csv"
                if membership_path.is_file():
                    context = pd.read_csv(membership_path)[["fdt_x_mm", "fdt_y_mm", "fdt_z_mm"]].to_numpy()
                draw_row(region_out, context, reason="Analysis failed\nSee error.txt")
                summary["figure_ok"] = True
            except Exception:
                (region_out / "figure_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        elapsed = perf_counter() - begin
        summary.update(seconds=elapsed, output_folder=str(region_out))
        summary.setdefault("figure_ok", True)
        write_csv(region_out / "roi_summary.csv", pd.DataFrame([summary]))
        sizes = {p.name: p.stat().st_size for p in region_out.iterdir() if p.is_file() and p.name != "status.json"}
        write_json(status_path, dict(fingerprint=fingerprint,
                   completed=summary["status"] != "error" and summary["figure_ok"],
                   summary=summary, output_sizes=sizes))
        rows.append(summary)
        measured_times.append(elapsed)
        table, report = write_batch_summary(out, rows, roi_ids)
        eta = np.mean(measured_times) * (len(roi_ids) - index) / 60
        gain = summary["mean_E_minus_H"]
        gain_text = f"{gain:+.5f}" if gain is not None and np.isfinite(gain) else "not evaluable"
        print(f"  {summary['status']} | mean E-H={gain_text} | {elapsed:.1f} s | rough remaining {eta:.1f} min", flush=True)
    table, report = write_batch_summary(out, rows, roi_ids)
    print("\nCompleted. Descriptive ROI summary:")
    print(json.dumps(json_ready(report), indent=2))
    print(f"\nSaved: {out}")
    print("Primary table: roi_summary.csv. Positive mean_E_minus_H favors H; negative favors E.")
    print("ROIs are regions of one brain, not independent subjects. No superiority is assumed.")
    return table


if __name__ == "__main__":
    roi_results20 = run_cell20()
