# Cell 14b: export whole-brain graph edges for FSLeyes.
# NIfTI shows rasterized edge occupancy; TCK retains separate two-point segments.
# These are local graph edges, not reconstructed axons or PROBTRACKX streamlines.
# A finer display grid does not increase the underlying MRI resolution.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse


def export_graph_views14b():
    root = Path("/mnt/g/Research/TUM_connectome/HyperbolicTest")
    src = root / "results/cell04_orientation_graph"
    out = root / "results/cell14b_connectivity_nifti"
    out.mkdir(parents=True, exist_ok=True)
    upsample = 4
    ref_path = root / "BEDPOSTX.bedpostX/nodif_brain_mask.nii.gz"
    ref = nib.load(ref_path)
    metadata = json.loads((src / "metadata.json").read_text())
    if len(ref.shape) != 3 or ref.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Expected a 3D reference image with spatial units in mm.")
    if not np.allclose(ref.affine, metadata["affine"], atol=1e-5, rtol=0):
        raise ValueError("Reference image and graph coordinate systems disagree.")
    # Native voxel centers map exactly to display-grid indices multiplied by upsample.
    display_shape = tuple(int(v) for v in (np.array(ref.shape)-1)*upsample+1)
    display_affine = ref.affine.copy()
    display_affine[:3, :3] /= upsample
    zooms = np.linalg.norm(display_affine[:3, :3], axis=0)
    print(f"Native shape: {ref.shape}; native voxel sizes: {tuple(float(v) for v in ref.header.get_zooms()[:3])} mm")
    print(f"Display shape: {display_shape}; display voxel sizes: {tuple(float(v) for v in zooms)} mm")
    print("Exporting all original graph edges, without landmark subsampling.")
    rows = []
    for name in ("supported", "cone45"):
        G = sparse.load_npz(src / f"edge_lengths_{name}.npz").tocsr()
        with np.load(src / f"nodes_{name}.npz") as z:
            ijk, xyz, ids, component = [z[k].copy() for k in
                ("ijk", "world_xyz_mm", "base_node_id", "component")]
        n = len(ids)
        if G.shape != (n, n) or G.diagonal().any() or (G != G.T).nnz:
            raise ValueError("Invalid graph or node-table mismatch.")
        if not np.isfinite(G.data).all() or np.any(G.data <= 0):
            raise ValueError("Invalid graph edge lengths.")
        if not np.isin(component, (1, 2)).all() or len(np.unique(np.c_[ijk, component], axis=0)) != n:
            raise ValueError("Expected unique voxel-component nodes with at most two components.")
        if not np.all((ijk >= 0) & (ijk < np.array(ref.shape))):
            raise ValueError("Graph coordinates fall outside the reference image.")
        if not np.allclose(nib.affines.apply_affine(ref.affine, ijk), xyz, rtol=0, atol=1e-4):
            raise ValueError("Stored world coordinates do not match the reference affine.")
        U = sparse.triu(G, k=1).tocoo()
        edges = np.c_[U.row, U.col].astype(np.int32)
        delta = ijk[edges[:, 1]].astype(np.int32)-ijk[edges[:, 0]].astype(np.int32)
        if not len(edges) or not np.all(np.max(np.abs(delta), axis=1) == 1):
            raise ValueError("Expected nonempty 26-neighbor graph edges.")
        if not np.allclose(np.linalg.norm(xyz[U.row]-xyz[U.col], axis=1), U.data, rtol=0, atol=1e-5):
            raise ValueError("Edge lengths disagree with endpoint positions.")

        # Each local edge contributes once at each sampled display-grid location.
        # With 26-neighbor edges and <=2 nodes per native voxel, uint16 is sufficient.
        volume = np.zeros(display_shape, dtype=np.uint16)
        first = ijk[edges[:, 0]].astype(np.int32)*upsample
        for step in range(upsample+1):
            points = first+step*delta
            np.add.at(volume, tuple(points.T), 1)
        visits = int(volume.sum(dtype=np.uint64))
        if visits != len(edges)*(upsample+1):
            raise ValueError("Rasterization count check failed.")
        nifti_path = out / f"{name}_graph_edges.nii.gz"
        header = nib.Nifti1Header()
        header.set_data_dtype(np.uint16)
        header.set_xyzt_units("mm")
        header["descrip"] = b"Local graph edge visits; visualization only; not tractography"
        img = nib.Nifti1Image(volume, display_affine, header)
        img.set_sform(display_affine, code=int(ref.header["sform_code"]) or 2)
        qcode = int(ref.header["qform_code"])
        if qcode:
            img.set_qform(display_affine, code=qcode)
        img.header["cal_min"], img.header["cal_max"] = 0, int(volume.max())
        nib.save(img, nifti_path)
        check = nib.load(nifti_path)
        if check.shape != display_shape or not np.allclose(check.affine, display_affine, atol=1e-5, rtol=0):
            raise ValueError("Saved NIfTI geometry check failed.")
        del check, img

        # TCK coordinates are already in world RAS+ millimeters.
        segments = xyz[edges].astype(np.float32)
        tractogram = nib.streamlines.Tractogram(segments, affine_to_rasmm=np.eye(4))
        tck_path = out / f"{name}_graph_edges.tck"
        nib.streamlines.save(tractogram, str(tck_path))
        np.savez_compressed(out / f"{name}_edge_index.npz",
            node_rows=edges, base_node_ids=ids[edges], length_mm=U.data,
            interpretation="One TCK segment per graph edge, in this row order")
        rows.append(dict(graph=name, nodes=n, undirected_edges=len(edges),
            isolated_nodes=int((np.diff(G.indptr) == 0).sum()),
            occupied_display_voxels=int(np.count_nonzero(volume)), max_edge_visits=int(volume.max()),
            nifti_file=nifti_path.name, tck_file=tck_path.name))
        print(f"{name}: {len(edges):,} edges -> {nifti_path.name} and {tck_path.name}", flush=True)
        del volume, segments, tractogram, G

    (out / "metadata.json").write_text(json.dumps(dict(
        reference=str(ref_path), source_graph_directory=str(src), variants=["supported", "cone45"],
        native_shape=list(ref.shape), native_affine=ref.affine.tolist(),
        display_shape=[int(s) for s in display_shape], display_affine=display_affine.tolist(),
        display_voxel_mm=zooms.tolist(), upsample=upsample,
        coordinate_mapping="Original voxel centers align exactly; display extent spans the native voxel centers",
        nifti_values="Number of local graph-edge centerlines sampled at each display voxel; not streamline probability or connection strength",
        tck="Each two-point segment is one local graph edge; endpoint IDs are in the accompanying edge_index NPZ",
        scope="All original graph edges, including small components; isolated nodes have no edge to draw",
        interpretation="BEDPOSTX-derived compatibility graph visualization; not original tractography and not a hyperbolic embedding",
    ), indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "export_summary.csv", index=False)
    print("\n"+summary.to_string(index=False))
    print("In FSLeyes, open the native mean_S0samples image, then one exported overlay.")
    print("For NIfTI, clip values below 0.5 to hide the background. For separate lines, open the TCK in a 3D view.")
    print(f"Saved: {out}")
    return summary


export14b = export_graph_views14b()
