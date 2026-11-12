# Check existing Cell 14c exports without changing images or registration.
# Original cone45 is a subgraph of original supported; rewired controls are not.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import sparse


def check_alignment14c(project):
    root = Path(project)
    source = root / "results/cell04_orientation_graph"
    exported = root / "results/cell14c_fdt_space"
    out = exported / "alignment_check"
    out.mkdir(parents=True, exist_ok=True)
    ref_path = root / "PROBTRACKX/fdt_paths.nii.gz"
    ref = nib.load(ref_path)
    print("Reference:", ref_path)
    print("Checking exports in:", exported)
    rows = []
    for path in sorted(exported.glob("*.nii.gz")):
        img = nib.load(path)
        forms_ok = all(int(img.header[f + "_code"]) == int(ref.header[f + "_code"]) and
                       np.allclose(getattr(img, "get_" + f)(), getattr(ref, "get_" + f)(), atol=1e-4, rtol=0)
                       for f in ("qform", "sform"))
        rows.append(dict(file=path.name, shape=str(img.shape),
                         shape_match=img.shape == ref.shape,
                         affine_match=np.allclose(img.affine, ref.affine, atol=1e-4, rtol=0),
                         voxel_size_match=np.allclose(img.header.get_zooms()[:3], ref.header.get_zooms()[:3], atol=1e-5, rtol=0),
                         forms_match=forms_ok,
                         units_match=img.header.get_xyzt_units()[0] == ref.header.get_xyzt_units()[0]))
    table = pd.DataFrame(rows)
    table.to_csv(out / "header_checks.csv", index=False)
    print("\nHeader checks:\n" + table.to_string(index=False))
    if table.empty or not table.drop(columns=["file", "shape"]).to_numpy(dtype=bool).all():
        raise ValueError("Missing or inconsistent exports. Send the header-check table before changing any transform.")
    with np.load(exported / "coordinate_transforms.npz") as z:
        voxel_map, target_affine = z["dwi_voxel_to_fdt_voxel"].copy(), z["fdt_affine"].copy()
    checks = dict(saved_reference_affine_matches=bool(np.allclose(target_affine, ref.affine, atol=1e-4, rtol=0)))
    nodes, native, volumes = {}, {}, {}
    for name in ("supported", "cone45"):
        with np.load(exported / f"nodes_{name}_in_fdt.npz") as z:
            nodes[name] = {k: z[k].copy() for k in ("base_node_id", "native_ijk", "fdt_world_xyz_mm")}
        with np.load(source / f"nodes_{name}.npz") as z:
            native[name] = {k: z[k].copy() for k in ("base_node_id", "ijk")}
        a, b = nodes[name], native[name]
        checks[name + "_source_nodes_match"] = bool(np.array_equal(a["base_node_id"], b["base_node_id"]) and
                                                     np.array_equal(a["native_ijk"], b["ijk"]))
        expected = nib.affines.apply_affine(ref.affine @ voxel_map, a["native_ijk"])
        checks[name + "_export_coordinate_error_mm"] = float(np.linalg.norm(expected-a["fdt_world_xyz_mm"], axis=1).max())
        path = exported / f"{name}_original_in_fdt.nii.gz"
        print("Original overlay:", path)
        volumes[name] = np.asarray(nib.load(path).dataobj)
    common, si, ci = np.intersect1d(nodes["supported"]["base_node_id"], nodes["cone45"]["base_node_id"], return_indices=True)
    checks["all_cone_nodes_in_supported"] = bool(len(common) == len(nodes["cone45"]["base_node_id"]))
    checks["common_node_count"] = int(len(common))
    checks["common_node_max_position_error_mm"] = (float(np.linalg.norm(
        nodes["supported"]["fdt_world_xyz_mm"][si]-nodes["cone45"]["fdt_world_xyz_mm"][ci], axis=1).max()) if len(common) else None)
    if checks["all_cone_nodes_in_supported"] and all(checks[n + "_source_nodes_match"] for n in nodes):
        G = sparse.load_npz(source / "edge_lengths_supported.npz").tocsr()
        H = sparse.load_npz(source / "edge_lengths_cone45.npz").tocsr()
        checks["original_graph_subset_matches"] = bool((G[si][:, si] != H[ci][:, ci]).nnz == 0)
    S, C = volumes["supported"], volumes["cone45"]
    checks["supported_positive_voxels"] = int(np.count_nonzero(S))
    checks["cone45_positive_voxels"] = int(np.count_nonzero(C))
    checks["cone45_voxels_absent_from_supported"] = int(np.count_nonzero((C > 0) & (S == 0)))
    checks["cone45_count_exceeds_supported_voxels"] = int(np.count_nonzero(C > S))
    fdt = np.asarray(ref.dataobj, dtype=np.float32)
    for name, volume in volumes.items():
        positive = volume > 0
        checks[name + "_fraction_on_positive_fdt"] = float(np.mean(fdt[positive] > 0)) if positive.any() else None
    (out / "alignment_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("\nContent checks:\n" + json.dumps(checks, indent=2))

    support = np.argwhere(fdt > 0)
    center = np.median(support, axis=0).astype(int) if len(support) else np.array(ref.shape)//2
    background = np.log1p(np.maximum(fdt, 0))
    vmax = max(float(np.percentile(background[fdt > 0], 99)), 1.) if len(support) else 1.
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
    for row, name in enumerate(("supported", "cone45")):
        for axis in range(3):
            ax = axes[row, axis]
            ax.imshow(np.take(background, center[axis], axis=axis).T, origin="lower", cmap="gray", vmin=0, vmax=vmax)
            overlay = np.take(volumes[name] > 0, center[axis], axis=axis).T
            ax.imshow(np.ma.masked_where(~overlay, overlay), origin="lower",
                      cmap=ListedColormap(["#00c5ed" if row == 0 else "#ff9f43"]),
                      vmin=0, vmax=1, alpha=.65, interpolation="nearest")
            ax.set_title(f"{name} | {'ijk'[axis]}={center[axis]}")
            ax.axis("off")
    fig.suptitle("Same fdt_paths voxel slices | gray: log(1+fdt_paths) | color: original graph")
    fig.savefig(out / "same_slice_overlays.png", dpi=160)
    plt.show()
    print("\nGrid agreement and subset checks do not independently validate anatomical registration.")
    print("Overlap with positive fdt_paths is descriptive; graph coverage need not equal tracking density support.")
    print(f"Saved: {out}")
    return table, checks


if __name__ == "__main__":
    alignment_headers, alignment_checks = check_alignment14c("/mnt/g/Research/TUM_connectome/HyperbolicTest")
