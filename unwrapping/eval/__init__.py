"""Video-vs-GT evaluation suite for the unrolled film.

Compares the unrolled CT film video (`film_final_p902.mp4`) against the
ground-truth optical scan (`data/h265_1080p.mp4`) of the SAME physical reel.

The two videos are cross-modal (CT-derived positive vs optical positive),
different resolution / frame rate / field-of-view (GT keeps sprockets), so the
pipeline is: normalize -> temporal-align (DTW) -> spatial-register -> score with
modality/alignment-robust metrics. See `compare_videos.py` for the CLI and
`metrics.py` for the metric definitions.
"""
