"""Assemble a portable handoff from the artifacts already in this workspace.

Large files are NTFS hard-linked to avoid using another ~40 GB locally. Copying
the resulting handoff directory to another disk/cloud transfers their bytes.
"""

from __future__ import annotations

import hashlib
import filecmp
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deliverable" / "handoff"
VERIFIED = ROOT.parent / "M_thesis_deliverable"
COPIED = []


def add(source: str | Path, target: str | Path | None = None) -> None:
    source = Path(source)
    src = ROOT / source
    rel = Path(target) if target else source
    dst = OUT / rel
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not os.path.samefile(src, dst):
            dst.unlink()
    if not dst.exists():
        if src.suffix.lower() in {".py", ".sh", ".md", ".txt", ".tex", ".json"}:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
    COPIED.append({"path": rel.as_posix(), "bytes": dst.stat().st_size,
                   "source": source.as_posix()})


def add_tree(path: str, include) -> None:
    base = ROOT / path
    for src in sorted(base.rglob("*")):
        if src.is_file() and include(src):
            add(src.relative_to(ROOT))


def add_verified_tree() -> None:
    """Keep the independently verified package intact under one subdirectory."""
    if not VERIFIED.is_dir():
        raise FileNotFoundError(f"Verified source deliverable not found: {VERIFIED}")

    # These bytes already have canonical paths in the parent handoff. The
    # verified scripts are pointed at those paths below, so the ZIP need not
    # contain a second copy of the same training data and final strip.
    omitted = (
        Path("Data/1_segmentation_training/nnUNet_raw/Dataset501_MickeyScroll"),
        Path("Data/1_segmentation_training/nnUNet_raw/Dataset502_MickeyScroll3D"),
        Path("Results/2_final_model_full_width/strip/wholeroll_tw4.npy"),
    )
    for rel in omitted:
        target = OUT / "verified_reference" / rel
        if target.exists():
            if target.is_symlink() or not target.resolve().is_relative_to(OUT.resolve()):
                raise RuntimeError(f"Refusing to remove path outside handoff: {target}")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    def existing_equivalent(rel: Path) -> Path | None:
        parts = rel.parts
        mapped = None
        if parts[:3] == ("Data", "1_segmentation_training", "nnUNet_raw"):
            mapped = OUT / "Segmentation/nnUNet_data/nnUNet_raw" / Path(*parts[3:])
        elif parts[:3] == ("Data", "2_trained_models", "nnUNet_results"):
            mapped = OUT / "Segmentation/nnUNet_data/nnUNet_results" / Path(*parts[3:])
        elif parts[:3] == ("Data", "2_trained_models", "monai_results"):
            mapped = OUT / "Segmentation/monai_results" / Path(*parts[3:])
        elif rel.as_posix() == "Data/4_optical_ground_truth/optical_scan_h265_1080p.mp4":
            mapped = OUT / "data/h265_1080p.mp4"
        elif rel.as_posix() == "Data/4_optical_ground_truth/optical_scan_perflocked_picture.mp4":
            mapped = OUT / "unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4"
        elif rel.as_posix() == "Results/2_final_model_full_width/strip/wholeroll_tw4.npy":
            mapped = OUT / "unwrapping/inr/results/newscan/render/wholeroll_tw4.npy"
        elif rel.as_posix() == "Results/0_requested_videos/1b_early_version_emulsion_walk__picture_area_scan__as_in_presentation.mp4":
            mapped = OUT / "unwrapping/inr/results/walk_dense_v9/film_v9.mp4"
        return mapped

    for src in sorted(VERIFIED.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        rel = src.relative_to(VERIFIED)
        if any(rel == path or path in rel.parents for path in omitted):
            continue
        dst = OUT / "verified_reference" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in {".py", ".sh", ".md", ".txt", ".json"}:
            if rel == Path("Code/env.sh"):
                content = src.read_text(encoding="utf-8")
                content = content.replace(
                    'export nnUNet_raw="$THESIS_ROOT/Data/1_segmentation_training/nnUNet_raw"',
                    'export nnUNet_raw="$THESIS_ROOT/../Segmentation/nnUNet_data/nnUNet_raw"',
                )
                dst.write_text(content, encoding="utf-8", newline="\n")
            elif rel == Path("Code/slurm/eval_01_final_vs_gt.sh"):
                content = src.read_text(encoding="utf-8")
                content = content.replace(
                    '$THESIS_ROOT/Results/2_final_model_full_width/strip/wholeroll_tw4.npy',
                    '$THESIS_ROOT/../unwrapping/inr/results/newscan/render/wholeroll_tw4.npy',
                )
                dst.write_text(content, encoding="utf-8", newline="\n")
            elif rel == Path("README.md"):
                content = src.read_text(encoding="utf-8")
                content = content.replace(
                    "# Virtual Unrolling of Motion-Picture Film — Master's Thesis Deliverable",
                    "# Virtual Unrolling of Motion-Picture Film — Master's Thesis Deliverable\n\n"
                    "**Combined handoff layout.** This verified reference sits inside the main "
                    "handoff. To avoid duplicate files, the ready-to-train Dataset501 and Dataset502 "
                    "are at `../Segmentation/nnUNet_data/nnUNet_raw/`, and the final strip is at "
                    "`../unwrapping/inr/results/newscan/render/wholeroll_tw4.npy`. "
                    "`Code/env.sh` and `Code/slurm/eval_01_final_vs_gt.sh` use those locations. "
                    "The raw full-width CT is excluded and must be supplied separately at "
                    "`Data/3_full_width_ct_scan/ct_slices/` for CT-dependent reruns. "
                    "The original Dataset502 CT and Ilastik probability exports are included "
                    "at `../data/01_Mickey_3d/`. See the [main README](../README.md) "
                    "for the complete combined handoff instructions.\n",
                )
                dst.write_text(content, encoding="utf-8")
            else:
                shutil.copy2(src, dst)
        elif not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            tmp = dst.with_name(dst.name + ".part")
            if tmp.exists():
                tmp.unlink()
            candidate = existing_equivalent(rel)
            if (candidate is not None and candidate.is_file()
                    and candidate.stat().st_size == src.stat().st_size
                    and filecmp.cmp(src, candidate, shallow=False)):
                os.link(candidate, tmp)
            else:
                shutil.copy2(src, tmp)
            tmp.replace(dst)
        COPIED.append({"path": (Path("verified_reference") / rel).as_posix(),
                       "bytes": dst.stat().st_size,
                       "source": f"M_thesis_deliverable/{rel.as_posix()}"})


def small_source(src: Path) -> bool:
    return "results" not in src.relative_to(ROOT).parts and src.suffix.lower() in {".py", ".sh"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # The 63.5 GiB full-width CT archive remains in the working repository.
    # The handoff assumes the recipient already has it; remove only the old
    # handoff link if this directory was built by an earlier version.
    old_ct_link = OUT / "data" / "Mickey_merged_rigid_nobin-tif.zip"
    if old_ct_link.is_file():
        if not old_ct_link.resolve().is_relative_to(OUT.resolve()):
            raise RuntimeError(f"Refusing to remove path outside handoff: {old_ct_link}")
        old_ct_link.unlink()
    for path in ("Scripts", "unwrapping", "video_restoration/scripts"):
        add_tree(path, small_source)
    add("requirements.txt")
    add("Segmentation/reported_fold0_split.json")
    add("Segmentation/fair_comparison/fair_comparison_results.json")
    add("Segmentation/segmentation_validation_recomputed.json")
    add("Segmentation/segmentation_validation_recomputed.json",
        "metrics/segmentation_validation_recomputed.json")
    add("_share/walk_algo_share/README.md")
    add_tree("_share/walk_algo_share/sample_data", lambda p: p.suffix == ".h5")
    add_tree("docs/images", lambda p: p.is_file())
    add_tree("docs/media", lambda p: p.is_file())
    add("docs/thesisdraft.tex")
    add("presentations/Kai Wen Masters Thesis Presentation October 1 2026 - Revised.pdf")
    add("presentations/Kai Wen Masters Thesis Presentation October 1 2026 - Revised.pptx")
    for weight in ("torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
                   "torch/hub/pyiqa/LPIPS_v0.1_alex-df73285e.pth"):
        add(Path("deliverable/weights") / weight, Path("weights") / weight)

    dataset = "Segmentation/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D"
    add_tree(dataset, lambda p: p.is_file())
    # Preserve the 25 CT volumes and ilastik probability exports from which
    # Dataset502's categorical masks can be regenerated.
    add_tree("data/01_Mickey_3d", lambda p: p.suffix.lower() == ".h5")
    # The original 2D model and MONAI results are included for the five-model
    # comparison. The matched 2D dataset can be rebuilt from Dataset502.
    add_tree("Segmentation/nnUNet_data/nnUNet_raw/Dataset501_MickeyScroll",
             lambda p: p.is_file())
    add_tree("Segmentation/monai_results",
             lambda p: p.suffix in {".json", ".pt"} or p.name.endswith(".nii.gz"))
    for model, config in (("Dataset502_MickeyScroll3D", "3d_fullres"),
                          ("Dataset501_MickeyScroll", "2d")):
        folder = (ROOT / "Segmentation/nnUNet_data/nnUNet_results" / model /
                  f"nnUNetTrainerProgress__nnUNetPlans__{config}")
        for name in ("dataset.json", "dataset_fingerprint.json", "plans.json"):
            add((folder / name).relative_to(ROOT))
        fold = folder / "fold_0"
        add((fold / "checkpoint_final.pth").relative_to(ROOT))
        for src in sorted((fold / "validation").glob("*.nii.gz")):
            add(src.relative_to(ROOT))
        summary = fold / "validation/summary.json"
        if summary.is_file():
            add(summary.relative_to(ROOT))

    for source in (
        "data/h265_1080p.mp4",
        "data/roll2_probs.zip",
        "data/00200.tiff", "data/00700.tiff",
        "data/01200.tiff", "data/01700.tiff",
        "unwrapping/inr/results/arc_simple/wholeroll_v2/film_final_p902.mp4",
        "unwrapping/inr/results/walk_dense_v9/film_v9.mp4",
        "unwrapping/inr/results/walk_dense_v9/wholeroll.npy",
        "unwrapping/inr/results/walk_dense_v9/matched_walks.npz",
        "unwrapping/inr/results/walk_dense_v13/film_v13.mp4",
        "unwrapping/inr/results/walk_dense_v13_tw3/film_v13_tw3.mp4",
        "unwrapping/inr/results/walk_dense_v13_tw3/wholeroll.npy",
        "unwrapping/inr/results/newscan/render/wholeroll_tw4.npy",
        "unwrapping/inr/results/newscan/render/z_index_tw4.npy",
        "unwrapping/inr/results/newscan/stabilized/film_borderlock_m02.mp4",
        "unwrapping/inr/results/newscan/stabilized/FLAGSHIP_borderlock_affine.mp4",
        "unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4",
        "unwrapping/eval/results/affine_method_comparison/scores.json",
        "unwrapping/eval/results/newscan_affine_vs_gt_trim/scores.json",
        "unwrapping/eval/results/newscan_affine_vs_gt_trim/scores_clean.json",
    ):
        add(source)

    add_verified_tree()

    add("unwrapping/inr/results/walk_dense_v9/film_v9.mp4",
        "videos/01_emulsion_walk_ablation.mp4")
    add(OUT / "verified_reference/Results/0_requested_videos/1a_early_version_emulsion_walk__full_width.mp4",
        "videos/01a_emulsion_walk_full_width.mp4")
    COPIED[-1]["source"] = "verified_reference/Results/0_requested_videos/1a_early_version_emulsion_walk__full_width.mp4"
    add(OUT / "verified_reference/Results/0_requested_videos/2a_final_model_WITH_gt_frame_alignment__per_frame_affine.mp4",
        "videos/02_final_gt_assisted_alignment.mp4")
    COPIED[-1]["source"] = "verified_reference/Results/0_requested_videos/2a_final_model_WITH_gt_frame_alignment__per_frame_affine.mp4"
    add(OUT / "verified_reference/Results/0_requested_videos/3b_final_model_NO_gt__border_locked.mp4",
        "videos/03_final_gt_free_alignment.mp4")
    COPIED[-1]["source"] = "verified_reference/Results/0_requested_videos/3b_final_model_NO_gt__border_locked.mp4"

    readme = (ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")
    public_note = (
        "**GitHub contains source code, documentation, and the overview's showcase\n"
        "figures.** CT scans, annotations, trained weights, quantitative result files,\n"
        "and deliverable videos are distributed separately where rights permit. Data\n"
        "paths below describe that separate handoff and are absent from GitHub.\n\n"
    )
    if public_note not in readme:
        raise ValueError("Reproduction guide's public data note changed")
    readme = readme.replace(
        public_note,
        "**This handoff contains the training data, models, outputs, and videos "
        "described below.** The separately supplied raw full-width CT archive "
        "is required for CT-dependent reruns.\n\n",
    )
    public_location = (
        "The portable data package is assembled locally in `deliverable/handoff/` and is\n"
        "not committed to GitHub. Run the commands below from a checkout populated with\n"
        "the separately supplied handoff data, or from the handoff package root."
    )
    if public_location not in readme:
        raise ValueError("Reproduction guide's public location note changed")
    readme = readme.replace(
        public_location,
        "This directory is the portable handoff package. Run the commands below "
        "from this directory.",
    )
    readme = readme.replace("deliverable/handoff/videos/", "videos/")
    readme = readme.replace("deliverable/handoff/MANIFEST.json", "MANIFEST.json")
    readme = readme.replace("[`deliverable/handoff/videos/`](videos/)", "[`videos/`](videos/)")
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    COPIED.append({"path": "README.md", "bytes": (OUT / "README.md").stat().st_size,
                   "source": "REPRODUCTION.md (path-adjusted copy)"})

    # Hash the requested videos and LPIPS weights for an auditable transfer.
    for item in COPIED:
        if item["path"].startswith(("videos/", "weights/")):
            digest = hashlib.sha256()
            with (OUT / item["path"]).open("rb") as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            item["sha256"] = digest.hexdigest()
    manifest = {
        "format": "workspace handoff, paths relative to this directory",
        "file_count": len(COPIED),
        "logical_bytes": sum(item["bytes"] for item in COPIED),
        "large_files": "NTFS hard links locally; copy/upload the directory to transfer bytes",
        "files": sorted(COPIED, key=lambda item: item["path"]),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"Handoff: {OUT}")
    print(f"Files: {manifest['file_count']}; logical bytes: {manifest['logical_bytes']:,}")


if __name__ == "__main__":
    main()
