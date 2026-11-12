# Cell 14: local degree- and length-preserving graph controls.
# A fixed random spanning forest preserves all original connected components.
# This is a conservative, backbone-conditioned pilot, not a fully mixed null ensemble.
# All comments and output labels are in English. No FSL or embedding fits are used.
from pathlib import Path
from itertools import product
from time import perf_counter
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def swap_local14(adj, degree, free_degree, ijk, length_code, anchors_a, anchors_c, slots_b, slots_d):
    accepted = 0
    for t in range(len(anchors_a)):
        a, c = int(anchors_a[t]), int(anchors_c[t])
        sb, sd = int(slots_b[t]), int(slots_d[t])
        b, d = int(adj[a, sb]), int(adj[c, sd])
        if a == c or a == d or b == c or b == d:
            continue
        ax, ay, az = (int(v) for v in ijk[a])
        bx, by, bz = (int(v) for v in ijk[b])
        cx, cy, cz = (int(v) for v in ijk[c])
        dx, dy, dz = (int(v) for v in ijk[d])
        u, v, w = ax-dx, ay-dy, az-dz
        p, q, r = cx-bx, cy-by, cz-bz
        if max(abs(u), abs(v), abs(w)) != 1 or max(abs(p), abs(q), abs(r)) != 1:
            continue
        old1 = length_code[ax-bx+1, ay-by+1, az-bz+1]
        old2 = length_code[cx-dx+1, cy-dy+1, cz-dz+1]
        new1, new2 = length_code[u+1, v+1, w+1], length_code[p+1, q+1, r+1]
        if not ((old1 == new1 and old2 == new2) or (old1 == new2 and old2 == new1)):
            continue
        duplicate = False
        for s in range(int(degree[a])):
            if adj[a, s] == d:
                duplicate = True
                break
        if duplicate:
            continue
        for s in range(int(degree[c])):
            if adj[c, s] == b:
                duplicate = True
                break
        if duplicate:
            continue
        rb, rd = -1, -1
        for s in range(int(free_degree[b])):
            if adj[b, s] == a:
                rb = s
                break
        for s in range(int(free_degree[d])):
            if adj[d, s] == c:
                rd = s
                break
        if rb < 0 or rd < 0:
            raise ValueError("Broken undirected free-edge bookkeeping.")
        adj[a, sb], adj[d, rd], adj[c, sd], adj[b, rb] = d, a, b, c
        accepted += 1
    return accepted


def run_cell14():
    root = Path("/mnt/g/Research/TUM_connectome/HyperbolicTest/results")
    src, panels = root / "cell04_orientation_graph", root / "cell11_new_landmarks"
    out = root / "cell14_graph_controls"
    out.mkdir(parents=True, exist_ok=True)
    null_seeds, checkpoints, attempts_per_free_edge = (1401, 1402, 1403), 5, 100
    metadata = json.loads((src / "metadata.json").read_text())
    affine = np.asarray(metadata["affine"], dtype=float)
    step_cos = np.cos(np.deg2rad(metadata["max_axis_to_displacement_deg"]))-1e-7
    axis_cos = np.cos(np.deg2rad(metadata["max_interaxis_deg"]))-1e-7
    offsets = np.array(list(product((-1, 0, 1), repeat=3)), dtype=np.int32)
    step_mm = np.linalg.norm(offsets @ affine[:3, :3].T, axis=1).reshape(3, 3, 3)
    codes = np.rint(step_mm*1e5).astype(np.int64)
    (out / "protocol.json").write_text(json.dumps(dict(
        null_seeds=null_seeds, checkpoints=checkpoints, attempts_per_free_edge=attempts_per_free_edge,
        source=str(src), node_definition="Same voxel-orientation nodes as Cell 4, including isolates",
        fixed="Node coordinates, per-node degree, connected-component partition, protected forest, global edge-length histogram",
        changed="Pairing of unprotected edges; local orientation compatibility is not imposed",
        forest="Minimum spanning forest with independent random edge costs; not a uniform spanning-tree sample",
        proposal="Uniform active anchor; uniform radius-2 voxel offset and component; uniform free neighbors; symmetric local two-switch",
        lengths="Same global counts of physical step-length classes; original stored edge weights retained exactly",
        limitations="Forest retains original structure; per-node strength and per-node length distributions are not fixed; mixing is not established",
        inference="Construction pilot only; no embedding scores, p-values, or biological inference",
    ), indent=2), encoding="utf-8")
    rows, traces = [], []
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    print(f"Graph controls: {len(null_seeds)} per graph; {attempts_per_free_edge} proposals per free edge.")
    print("Fixed spanning forest; exact degree/length/component checks; no curvature search.")

    for graph_index, name in enumerate(("supported", "cone45")):
        G = sparse.load_npz(src / f"edge_lengths_{name}.npz").tocsr()
        G.sort_indices()
        with np.load(src / f"nodes_{name}.npz") as z:
            ids, ijk, xyz, axes, comp = [z[k].copy() for k in
                ("base_node_id", "ijk", "world_xyz_mm", "direction_world", "component")]
        ijk = ijk.astype(np.int32)
        n, degree = len(ids), np.diff(G.indptr).astype(np.int32)
        if G.shape != (n, n) or G.diagonal().any() or (G != G.T).nnz or np.any(G.data <= 0):
            raise ValueError("Invalid input graph.")
        if not np.isfinite(G.data).all() or not np.allclose(xyz, ijk @ affine[:3, :3].T+affine[:3, 3]):
            raise ValueError("Invalid input weights or coordinates.")
        if not np.isin(comp, (1, 2)).all() or len(np.unique(np.c_[ijk, comp], axis=0)) != n:
            raise ValueError("Expected unique voxel-component nodes with component 1 or 2.")
        if not np.allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-5):
            raise ValueError("Invalid unit orientation axes.")
        nc, labels = connected_components(G, directed=False)
        U = sparse.triu(G, k=1).tocoo()
        original_edges = np.c_[U.row, U.col].astype(np.int32)
        edge_count = len(original_edges)
        old_keys = U.row.astype(np.int64)*n+U.col
        dr = ijk[U.row]-ijk[U.col]
        if not np.all(np.max(np.abs(dr), axis=1) == 1):
            raise ValueError("Expected 26-neighbor edges with no within-voxel edges.")
        original_codes = codes[tuple((dr+1).T)]
        physical = np.linalg.norm(xyz[U.row]-xyz[U.col], axis=1)
        if not np.allclose(physical, U.data, atol=1e-5, rtol=0):
            raise ValueError("Input edge values are not physical step lengths.")
        weight_by_code = {}
        for code in np.unique(original_codes):
            weights = np.unique(U.data[original_codes == code])
            if len(weights) != 1:
                raise ValueError("An edge-length class contains multiple stored weights.")
            weight_by_code[int(code)] = float(weights[0])
        step_weight = np.zeros((3, 3, 3), dtype=G.dtype)
        for code, value in weight_by_code.items():
            step_weight[codes == code] = value
        original_hist = np.unique(U.data, return_counts=True)
        pad_ijk = ijk-ijk.min(axis=0)+2
        lookup = np.full((*tuple(pad_ijk.max(axis=0)+3), 2), -1, dtype=np.int32)
        lookup[tuple(pad_ijk.T)+(comp.astype(int)-1,)] = np.arange(n, dtype=np.int32)
        np.savez_compressed(out / f"node_ids_{name}.npz", base_node_id=ids)
        panel_ids = json.loads((panels / "protocol.json").read_text())["landmark_seeds"]
        landmarks = {}
        for panel in panel_ids:
            table = pd.read_csv(panels / f"landmarks_{panel}.csv")
            ix = pd.Index(ids).get_indexer(table.base_node_id)
            if np.any(ix < 0) or len(np.unique(labels[ix])) != 1:
                raise ValueError("Original panel endpoints are missing or disconnected.")
            landmarks[panel] = ix
        row_index = np.repeat(np.arange(n, dtype=np.int32), degree)
        slots = np.arange(G.nnz)-np.repeat(G.indptr[:-1], degree)

        def edge_snapshot(adj):
            destination = adj[row_index, slots]
            upper = row_index < destination
            return np.c_[row_index[upper], destination[upper]]

        def direction_stats(edges):
            a, b = edges.T
            step = xyz[b]-xyz[a]
            step /= np.linalg.norm(step, axis=1)[:, None]
            ca, cb = np.abs(np.sum(axes[a]*step, axis=1)), np.abs(np.sum(axes[b]*step, axis=1))
            cab = np.abs(np.sum(axes[a]*axes[b], axis=1))
            compatible = (ca >= step_cos) & (cb >= step_cos) & (cab >= axis_cos)
            angle = np.degrees(np.arccos(np.clip(np.r_[ca, cb], 0, 1))).mean()
            return float(compatible.mean()), float(angle)

        baseline_compat, baseline_angle = direction_stats(original_edges)
        if baseline_compat < 1-1e-6:
            raise ValueError("Cell 4 direction rules do not reproduce the saved graph.")
        trace_ax = fig.add_subplot(2, 3, graph_index*3+1)
        print(f"\n{name}: {n:,} nodes, {edge_count:,} edges, {nc:,} components", flush=True)
        for seed in null_seeds:
            start = perf_counter()
            rng = np.random.default_rng(seed)
            costs = rng.uniform(.1, 1., edge_count)
            R = sparse.coo_matrix((costs, (U.row, U.col)), shape=G.shape).tocsr()
            T = minimum_spanning_tree(R+R.T)
            forest = (T+T.T).tocsr()
            forest.data[:] = 1
            protected = sparse.triu(forest, k=1).tocoo()
            assert len(protected.data) == n-nc
            free = (G-G.multiply(forest)).tocsr()
            free.eliminate_zeros()
            free_degree = np.diff(free.indptr).astype(np.int32)
            free_edges = free.nnz//2
            active = np.flatnonzero(free_degree).astype(np.int32)
            adj = np.full((n, max(1, int(degree.max()))), -1, dtype=np.int32)
            for source, shift in ((free, np.zeros(n, dtype=int)), (forest, free_degree)):
                counts = np.diff(source.indptr)
                rr = np.repeat(np.arange(n), counts)
                ss = np.arange(source.nnz)-np.repeat(source.indptr[:-1], counts)+shift[rr]
                adj[rr, ss] = source.indices
            accepted, proposals = 0, 0
            xs, ys = [0.], [0.]
            for checkpoint in range(1, checkpoints+1):
                count = (checkpoint*attempts_per_free_edge*free_edges)//checkpoints-proposals
                proposals += count
                # Anchor proposals depend only on fixed node positions and free degrees.
                for begin in range(0, count, 200_000):
                    batch = min(200_000, count-begin)
                    aa = rng.choice(active, batch)
                    location = pad_ijk[aa]+rng.integers(-2, 3, size=(batch, 3))
                    cc = lookup[tuple(location.T)+(rng.integers(0, 2, size=batch),)]
                    ok = cc >= 0
                    aa, cc = aa[ok], cc[ok]
                    ok = (aa != cc) & (free_degree[cc] > 0) & (labels[aa] == labels[cc])
                    aa, cc = aa[ok], cc[ok]
                    sb = (rng.random(len(aa))*free_degree[aa]).astype(np.int32)
                    sd = (rng.random(len(cc))*free_degree[cc]).astype(np.int32)
                    accepted += swap_local14(adj, degree, free_degree, ijk, codes, aa, cc, sb, sd)
                edges = edge_snapshot(adj)
                keys = edges[:, 0].astype(np.int64)*n+edges[:, 1]
                added = ~np.isin(keys, old_keys, assume_unique=True)
                replaced = int(added.sum())
                compat, angle = direction_stats(edges)
                traces.append(dict(graph=name, seed=seed, checkpoint=checkpoint, proposals=proposals,
                    accepted_swaps=accepted, changed_edge_fraction=replaced/edge_count,
                    replaced_free_fraction=replaced/free_edges if free_edges else 0.,
                    compatible_edge_fraction=compat, mean_axis_step_angle_deg=angle))
                xs.append(proposals/max(free_edges, 1)); ys.append(replaced/edge_count)
                pd.DataFrame(traces).to_csv(out / "rewiring_trace.csv", index=False)
                print(f"  seed {seed}, {checkpoint}/{checkpoints}: swaps={accepted:,}, "
                      f"changed={100*replaced/edge_count:.1f}% of all edges", flush=True)

            dr = ijk[edges[:, 0]]-ijk[edges[:, 1]]
            weights = step_weight[tuple((dr+1).T)]
            H = sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=G.shape).tocsr()
            H = H+H.T
            hc, hl = connected_components(H, directed=False)
            degree_ok = np.array_equal(np.diff(H.indptr), degree)
            hist = np.unique(weights, return_counts=True)
            length_ok = all(np.array_equal(a, b) for a, b in zip(hist, original_hist))
            component_ok = bool(hc == nc and np.all(labels[edges[:, 0]] == labels[edges[:, 1]]))
            forest_keys = protected.row.astype(np.int64)*n+protected.col
            forest_ok = bool(np.isin(forest_keys, keys, assume_unique=True).all())
            landmark_ok = all(len(np.unique(hl[ix])) == 1 for ix in landmarks.values())
            valid = bool(degree_ok and length_ok and component_ok and forest_ok and landmark_ok
                         and H.nnz == G.nnz and not H.diagonal().any() and not (H != H.T).nnz
                         and np.all(weights > 0) and len(np.unique(keys)) == edge_count)
            if not valid:
                raise ValueError("A graph-control invariant failed; no invalid graph was saved.")
            sparse.save_npz(out / f"edge_lengths_{name}_null_{seed}.npz", H)
            np.savez_compressed(out / f"protected_edges_{name}_{seed}.npz",
                                edges=np.c_[protected.row, protected.col], base_node_id=ids)
            rows.append(dict(graph=name, seed=seed, nodes=n, edges=edge_count,
                protected_fraction=1-free_edges/edge_count, free_edges=free_edges,
                proposals=proposals, accepted_swaps=accepted,
                acceptance_fraction=accepted/max(proposals, 1), changed_edge_fraction=replaced/edge_count,
                replaced_free_fraction=replaced/free_edges if free_edges else 0.,
                original_axis_step_angle_deg=baseline_angle, null_axis_step_angle_deg=angle,
                compatible_edge_fraction=compat, degree_preserved=degree_ok,
                length_hist_preserved=length_ok, components_preserved=component_ok,
                forest_preserved=forest_ok, landmarks_connected=landmark_ok,
                construction_ok=valid, seconds=perf_counter()-start))
            pd.DataFrame(rows).to_csv(out / "control_qc.csv", index=False)
            trace_ax.plot(xs, ys, "o-", label=f"seed {seed}")

            if seed == null_seeds[0]:
                # This crop is chosen to show changed edges, not anatomical representativeness.
                incident = np.bincount(edges[added].ravel(), minlength=n)
                center = xyz[int(incident.argmax())] if incident.any() else np.median(xyz, axis=0)
                inside = np.linalg.norm(xyz-center, axis=1) <= 12.
                for col, ee, flag, title in ((2, original_edges, np.zeros(edge_count, bool), "Original"),
                                             (3, edges, added, "Control; added edges in orange")):
                    ax = fig.add_subplot(2, 3, graph_index*3+col, projection="3d")
                    visible = inside[ee[:, 0]] & inside[ee[:, 1]]
                    for selected, color, width in ((visible & ~flag, "#64748b", .6),
                                                    (visible & flag, "#e87824", 1.3)):
                        if selected.any():
                            ax.add_collection3d(Line3DCollection(xyz[ee[selected]], colors=color, linewidths=width))
                    ax.set(xlim=(center[0]-12, center[0]+12), ylim=(center[1]-12, center[1]+12),
                           zlim=(center[2]-12, center[2]+12), title=f"{name}: {title}",
                           xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
                    ax.set_box_aspect((1, 1, 1))
                    ax.view_init(25, -55)
        trace_ax.axhline(free_edges/edge_count, ls="--", color="gray", label="Unprotected edge fraction")
        trace_ax.set(title=f"{name}: edge replacement", xlabel="Proposals per free edge",
                     ylabel="Fraction of all edges replaced", ylim=(0, 1))
        trace_ax.legend(fontsize=8)

    qc = pd.DataFrame(rows)
    fig.suptitle("Backbone-conditioned graph controls | fixed degree, edge lengths and components\n"
                 "Local 12-mm-radius previews use seed 1401; crops selected to show changed edges")
    fig.savefig(out / "rewiring_control_qc.png", dpi=180)
    plt.show()
    print("\nConstruction summary:")
    print(qc.to_string(index=False))
    if (qc.changed_edge_fraction == 0).any():
        print("WARNING: Some controls did not change; they cannot support a null comparison.")
    print("Successful construction does not establish sufficient mixing or biological specificity.")
    print("No curvature search, embedding fit, or significance test was run.")
    print(f"Saved: {out}")
    return qc, pd.DataFrame(traces)


qc14, trace14 = run_cell14()
