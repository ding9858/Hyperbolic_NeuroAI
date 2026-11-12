# Cell 14c: export graph edges in the original fdt_paths coordinate system.
# Reuse the registration recorded by PROBTRACKX; no FSL execution or tracking.
# NIfTI outputs use the exact fdt_paths grid. TCK vertices use its world space.
# Analysis graphs and native-space edge weights are not modified.
from pathlib import Path
import hashlib
import json
import shlex
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse, ndimage


def voxel_to_fsl14c(img):
    """Voxel to FSL scaled-mm coordinates, including the handedness convention."""
    pix = np.asarray(img.header.get_zooms()[:3], dtype=float)
    mat = np.diag(np.r_[pix, 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        mat[0, 0] = -pix[0]
        mat[0, 3] = (img.shape[0] - 1) * pix[0]
    return mat


def log_option14c(tokens, *names):
    values = []
    for i, token in enumerate(tokens):
        for name in names:
            if token == name and i + 1 < len(tokens):
                values.append(tokens[i + 1])
            elif token.startswith(name + "="):
                values.append(token.split("=", 1)[1])
    values = list(dict.fromkeys(values))
    if len(values) > 1:
        raise ValueError(f"Ambiguous logged option {names}: {values}")
    return values[0] if values else None


def local_file14c(root, logged):
    """Relocate historical paths only through known project directory names."""
    if not logged:
        return None
    parts = Path(logged.replace("\\", "/")).parts
    for anchor in ("BEDPOSTX.bedpostX", "BEDPOSTX", "T1", "PROBTRACKX"):
        if anchor in parts:
            k = len(parts) - 1 - list(reversed(parts)).index(anchor)
            path = root.joinpath(*parts[k:])
            stem = str(path)
            if stem.endswith(".nii.gz"):
                stem = stem[:-7]
            elif stem.endswith(".nii"):
                stem = stem[:-4]
            for candidate in dict.fromkeys((path, Path(stem + ".nii.gz"), Path(stem + ".nii"))):
                if candidate.is_file():
                    return candidate
    return None


def same_grid14c(a, b):
    return (a.shape == b.shape and
            np.allclose(a.affine, b.affine, atol=1e-4, rtol=0) and
            np.allclose(a.header.get_zooms()[:3], b.header.get_zooms()[:3], atol=1e-5, rtol=0))


def save_grid14c(data, reference, path, description):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    header.set_slope_inter(1, 0)
    header.set_intent("none")
    header["descrip"] = description.encode("ascii")[:79]
    img = nib.Nifti1Image(data, reference.affine, header)
    # Preserve both reference forms, including their codes and any differences.
    for form in ("qform", "sform"):
        matrix = getattr(reference, "get_" + form)()
        getattr(img, "set_" + form)(matrix, code=int(reference.header[form + "_code"]))
    img.header["cal_min"], img.header["cal_max"] = 0, float(data.max())
    nib.save(img, path)
    check = nib.load(path)
    if not same_grid14c(check, reference):
        raise ValueError(f"Saved grid differs from fdt_paths: {path}")
    for form in ("qform", "sform"):
        if int(check.header[form + "_code"]) != int(reference.header[form + "_code"]):
            raise ValueError(f"Changed {form} code: {path}")
        if not np.allclose(getattr(check, "get_" + form)(),
                           getattr(reference, "get_" + form)(), atol=1e-4, rtol=0):
            raise ValueError(f"Changed {form}: {path}")


def raster_edges14c(points, edges, shape):
    """Approximate line occupancy; count each edge once per visited target voxel."""
    volume = np.zeros(shape, dtype=np.uint32)
    start = points[edges[:, 0]]
    delta = points[edges[:, 1]] - start
    steps = np.maximum(1, np.ceil(4 * np.max(np.abs(delta), axis=1)).astype(int))
    if steps.max() > 1000:
        raise ValueError("Unexpected transformed edge length; inspect registration.")
    for count in np.unique(steps):
        rows = np.flatnonzero(steps == count)
        a, d = start[rows], delta[rows]
        previous = None
        for j in range(count + 1):
            voxel = np.floor(a + (j / count) * d + 0.5).astype(np.int32)
            keep = np.all((voxel >= 0) & (voxel < np.asarray(shape)), axis=1)
            if previous is not None:
                keep &= np.any(voxel != previous, axis=1)
            np.add.at(volume, tuple(voxel[keep].T), 1)
            previous = voxel
    return volume


def export_fdt_space14c(root):
    root = Path(root)
    result = root / "results"
    src = result / "cell04_orientation_graph"
    controls = result / "cell14_graph_controls"
    out = result / "cell14c_fdt_space"
    ref_path = root / "PROBTRACKX/fdt_paths.nii.gz"
    native_path = root / "BEDPOSTX.bedpostX/nodif_brain_mask.nii.gz"
    log_path = root / "PROBTRACKX/probtrackx.log"
    ref, native = nib.load(ref_path), nib.load(native_path)
    for img in (ref, native):
        if len(img.shape) != 3 or img.header.get_xyzt_units()[0] != "mm":
            raise ValueError("Expected 3D reference images in millimeters.")
        if not (int(img.header["sform_code"]) or int(img.header["qform_code"])):
            raise ValueError("Missing coded image coordinate transform.")
    graph_meta = json.loads((src / "metadata.json").read_text())
    if not np.allclose(native.affine, graph_meta["affine"], atol=1e-5, rtol=0):
        raise ValueError("Graph coordinates do not match the native DWI reference.")

    tokens = shlex.split(log_path.read_text(errors="replace").replace("\\\n", " "))
    logged_xfm = log_option14c(tokens, "--xfm")
    xfm_path = local_file14c(root, logged_xfm)
    if xfm_path is None or xfm_path.suffix != ".mat":
        raise ValueError("Cannot resolve the logged linear --xfm. Send probtrackx.log; no transform guessed.")
    logged_native = local_file14c(root, log_option14c(tokens, "-m", "--mask"))
    if logged_native is None or not same_grid14c(nib.load(logged_native), native):
        raise ValueError("Logged diffusion mask does not match the graph reference.")
    matched_masks = []
    for names in (("-x", "--seed"), ("--seedref",), ("--target3",), ("--lrtarget3",), ("--stop",), ("--avoid",)):
        path = local_file14c(root, log_option14c(tokens, *names))
        if path is None:
            continue
        if not same_grid14c(nib.load(path), ref):
            raise ValueError(f"Logged seed-space image differs from fdt_paths: {path}")
        matched_masks.append(str(path.relative_to(root)))
    if not matched_masks:
        raise ValueError("No logged seed-space image found to verify the fdt_paths grid. Send probtrackx.log and T1 filenames.")

    seed_to_diff = np.loadtxt(xfm_path)
    if (seed_to_diff.shape != (4, 4) or not np.isfinite(seed_to_diff).all() or
            not np.allclose(seed_to_diff[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError("Expected a finite 4x4 FLIRT affine matrix.")
    diff_to_seed = np.linalg.inv(seed_to_diff)
    # The logged transform acts on FSL coordinates, not NIfTI world coordinates.
    voxel_map = np.linalg.inv(voxel_to_fsl14c(ref)) @ diff_to_seed @ voxel_to_fsl14c(native)
    world_map = ref.affine @ voxel_map @ np.linalg.inv(native.affine)
    inverse_error = None
    inverse_path = xfm_path.with_name("diff2str.mat")
    if xfm_path.name == "str2diff.mat" and inverse_path.is_file():
        supplied_inverse = np.loadtxt(inverse_path)
        corners = np.array(np.meshgrid(*[(0, n - 1) for n in native.shape], indexing="ij")).reshape(3, -1).T
        fsl_corners = nib.affines.apply_affine(voxel_to_fsl14c(native), corners)
        inverse_error = float(np.linalg.norm(nib.affines.apply_affine(supplied_inverse, fsl_corners) -
                                             nib.affines.apply_affine(diff_to_seed, fsl_corners), axis=1).max())
        if inverse_error > 0.05:
            raise ValueError(f"diff2str.mat disagrees with inverse(str2diff.mat): {inverse_error:.4f} mm. Send both matrices.")

    print("Registration: inverse of the seed-to-DWI FLIRT matrix recorded in probtrackx.log.")
    print(f"Transform: {xfm_path}")
    print(f"Target: {ref.shape}; voxel sizes: {ref.header.get_zooms()[:3]} mm")
    print(f"Verified seed-space masks: {len(set(matched_masks))}")
    if inverse_error is not None:
        print(f"Forward/inverse agreement at DWI corners: {inverse_error:.6g} mm")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "coordinate_transforms.npz", dwi_voxel_to_fdt_voxel=voxel_map,
                        dwi_world_to_fdt_world=world_map, native_affine=native.affine,
                        fdt_affine=ref.affine, native_shape=native.shape, fdt_shape=ref.shape)
    back = np.linalg.inv(voxel_map)
    aligned_mask = ndimage.affine_transform(np.asarray(native.dataobj) > 0, back[:3, :3], back[:3, 3],
                                          output_shape=ref.shape, order=0, mode="constant", cval=0,
                                          prefilter=False).astype(np.uint8)
    save_grid14c(aligned_mask, ref, out / "dwi_brain_mask_in_fdt.nii.gz", "DWI brain mask in fdt_paths space; nearest neighbor")
    fdt_positive = np.asarray(ref.dataobj) > 0
    overlap = float(aligned_mask[fdt_positive].mean()) if fdt_positive.any() else None
    if overlap is not None:
        print(f"Positive fdt_paths voxels inside transformed DWI mask: {overlap:.2%} (descriptive QC)")

    rows = []
    for name in ("supported", "cone45"):
        with np.load(src / f"nodes_{name}.npz") as z:
            ijk, xyz, ids = [z[k].copy() for k in ("ijk", "world_xyz_mm", "base_node_id")]
        if not np.allclose(nib.affines.apply_affine(native.affine, ijk), xyz, atol=1e-4, rtol=0):
            raise ValueError(f"Stored coordinates are inconsistent: {name}")
        target_vox = nib.affines.apply_affine(voxel_map, ijk)
        target_xyz = nib.affines.apply_affine(ref.affine, target_vox)
        if not np.allclose(target_xyz, nib.affines.apply_affine(world_map, xyz), atol=1e-4, rtol=0):
            raise ValueError("Voxel and world coordinate routes disagree.")
        inside = np.all((target_vox >= -0.5) & (target_vox < np.asarray(ref.shape) - 0.5), axis=1)
        np.savez_compressed(out / f"nodes_{name}_in_fdt.npz", base_node_id=ids,
                            native_ijk=ijk, fdt_voxel_xyz=target_vox, fdt_world_xyz_mm=target_xyz)
        graphs = [("original", src / f"edge_lengths_{name}.npz")]
        nulls = sorted(controls.glob(f"edge_lengths_{name}_null_*.npz"))
        if nulls:
            with np.load(controls / f"node_ids_{name}.npz") as z:
                if not np.array_equal(ids, z["base_node_id"]):
                    raise ValueError("Control graph node ordering differs from the original.")
        graphs += [("null_" + p.stem.rsplit("_", 1)[1], p) for p in nulls]
        for variant, path in graphs:
            G = sparse.load_npz(path).tocsr()
            if G.shape != (len(ids), len(ids)) or G.diagonal().any() or (G != G.T).nnz:
                raise ValueError(f"Invalid graph: {path}")
            U = sparse.triu(G, k=1).tocoo()
            edges = np.c_[U.row, U.col].astype(np.int32)
            if not len(edges):
                raise ValueError(f"Empty graph: {path}")
            prefix = f"{name}_{variant}_in_fdt"
            volume = raster_edges14c(target_vox, edges, ref.shape)
            save_grid14c(volume, ref, out / f"{prefix}.nii.gz", "Sampled local graph-edge occupancy in fdt_paths grid")
            segments = target_xyz[edges].astype(np.float32)
            nib.streamlines.save(nib.streamlines.Tractogram(segments, affine_to_rasmm=np.eye(4)),
                                 str(out / f"{prefix}.tck"))
            np.savez_compressed(out / f"{prefix}_edge_index.npz", node_rows=edges,
                                base_node_ids=ids[edges], native_edge_length_mm=U.data,
                                display_edge_length_mm=np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1))
            clipped = int(np.count_nonzero(~np.all(inside[edges], axis=1)))
            rows.append(dict(graph=name, variant=variant, edges=len(edges), target_grid_match=True,
                             nodes_inside_fdt_FOV=float(inside.mean()), edges_with_endpoint_outside_FOV=clipped,
                             occupied_voxels=int(np.count_nonzero(volume)), nifti=f"{prefix}.nii.gz", tck=f"{prefix}.tck"))
            print(f"{name}, {variant}: {len(edges):,} edges; exact target grid; {clipped:,} edges touch/exceed FOV", flush=True)
            del volume, segments, G

    previews = []
    for folder, stem in (("cell05_path_geometry", "model_shortest_paths"),
                         ("cell06_turn60_paths", "model_turn_constrained")):
        for name in ("supported", "cone45"):
            path = result / folder / f"{stem}_{name}.tck"
            if not path.is_file():
                continue
            tractogram = nib.streamlines.load(str(path)).tractogram.to_world()
            mapped_paths = [nib.affines.apply_affine(world_map, p).astype(np.float32)
                            for p in tractogram.streamlines]
            destination = out / f"{path.stem}_in_fdt.tck"
            nib.streamlines.save(nib.streamlines.Tractogram(mapped_paths, affine_to_rasmm=np.eye(4)),
                                 str(destination))
            previews.append(destination.name)
    manifest = dict(reference=str(ref_path), native_reference=str(native_path),
                    source_log=str(log_path), log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
                    logged_seed_to_diffusion=logged_xfm, resolved_transform=str(xfm_path),
                    transform_sha256=hashlib.sha256(xfm_path.read_bytes()).hexdigest(),
                    matched_seed_space_masks=sorted(set(matched_masks)),
                    fdt_shape=list(ref.shape), fdt_affine=ref.affine.tolist(),
                    dwi_world_to_fdt_world=world_map.tolist(), dwi_voxel_to_fdt_voxel=voxel_map.tolist(),
                    inverse_agreement_max_mm=inverse_error, fdt_positive_inside_DWI_mask_fraction=overlap,
                    converted_preview_files=previews, coordinates="TCK: fdt_paths world RAS+ mm; NIfTI: exact fdt_paths grid",
                    rasterization="Nearest-voxel sampling, <=0.25 target voxels per axis per step; each edge counted once per visited voxel",
                    clipping="TCK retains every edge; NIfTI can display only the reference FOV; out-of-FOV counts are reported",
                    interpretation="Coordinate conversion for visualization; original model graphs, weights, and analyses unchanged; not tractography")
    (out / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "export_summary.csv", index=False)
    print(f"\nConverted prior preview TCK files: {len(previews)}")
    print(summary[["graph", "variant", "edges", "target_grid_match", "nodes_inside_fdt_FOV"]].to_string(index=False))
    print("FSLeyes: load fdt_paths, then *_in_fdt.nii.gz or *_in_fdt.tck.")
    print("TCK Reference image = fdt_paths; Coordinate space = World.")
    print("Grid checks establish coordinate consistency; inspect anatomical alignment in FSLeyes.")
    print(f"Saved: {out}")
    return summary


if __name__ == "__main__":
    export14c = export_fdt_space14c("/mnt/g/Research/TUM_connectome/HyperbolicTest")
