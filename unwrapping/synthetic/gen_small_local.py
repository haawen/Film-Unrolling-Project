"""Generate a small synthetic video volume locally for sanity checks.

Smaller than the HQ presets so it renders in ~1 min on a CPU laptop, but uses
real video frames (Big Buck Bunny) so the unrolled strip shows recognizable
content. Outer emulsion + flat slab to match the current canonical generator.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.synthetic.generate import SpiralParams, generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir",
                    default="unwrapping/synthetic/results/small_local")
    ap.add_argument("--image-size", type=int, default=1400)
    ap.add_argument("--n-windings", type=int, default=18)
    ap.add_argument("--n-z", type=int, default=48)
    ap.add_argument("--imperfect", action="store_true")
    args = ap.parse_args()

    kw = dict(
        image_size=args.image_size, n_windings=args.n_windings,
        film_thickness=20, air_gap=8, n_z_slices=args.n_z,
        pattern="video", emulsion_side="outer", video_path=args.video,
    )
    if args.imperfect:
        kw.update(noise_std=0.03, eccentricity=8.0, radial_jitter=4.0)
    params = SpiralParams(**kw)
    generate_dataset(params, args.output_dir)


if __name__ == "__main__":
    main()
