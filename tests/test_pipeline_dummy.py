"""
Smoke test: runs the full 6-stage pipeline on synthetic data in dummy mode.
No GPU, no downloaded weights required — this is meant to catch plumbing
breakage (shape mismatches, missing handoff fields, CRS bugs) fast, in CI or
locally, well before anyone waits on a BIT/SAM2.1 training run.

Run with: pytest tests/test_pipeline_dummy.py -v
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.pipeline import load_config, force_dummy_mode, run_pipeline
from scripts.run_demo import make_synthetic_pair


def test_pipeline_end_to_end(tmp_path):
    t1_path, t2_path, parcel_path = make_synthetic_pair(tmp_path, size=256)

    cfg = load_config(str(pathlib.Path(__file__).resolve().parents[1] / "config.yaml"))
    cfg = force_dummy_mode(cfg)
    cfg["stage0_align"]["target_resolution_m"] = 0.10

    out_dir = tmp_path / "out"
    paths = run_pipeline(str(t1_path), str(t2_path), str(parcel_path), str(out_dir), cfg)

    assert pathlib.Path(paths["geojson"]).exists()
    assert pathlib.Path(paths["overlay_png"]).exists()
    assert pathlib.Path(paths["summary_csv"]).exists()

    # The synthetic structure was built to cross the parcel boundary, so we
    # expect at least one detected + measured region flagged as encroaching.
    import geopandas as gpd

    gdf = gpd.read_file(paths["geojson"])
    assert len(gdf) >= 1
    assert gdf["encroaches_parcel"].any(), (
        "Expected the synthetic structure to be flagged as encroaching the parcel"
    )
