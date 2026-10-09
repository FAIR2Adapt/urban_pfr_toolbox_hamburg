"""CLI for the post-output static visualization subsystem: python -m urban_pfr.visualization"""

import argparse
import os
import sys

# Deterministic headless backend before any heavy import
os.environ.setdefault("MPLBACKEND", "Agg")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m urban_pfr.visualization",
        description="Render static maps from an existing urban_pfr run directory "
                    "(post-output only; never recomputes scientific values).")
    parser.add_argument("--run-dir", required=True, help="Run directory containing private/ and public/")
    parser.add_argument("--tier", required=True, choices=["private", "public"])
    parser.add_argument("--view", required=True,
                        choices=["risk_ma", "risk_wb", "risk_ma_raw", "risk_wb_raw", "vulnerability",
                                 "hazard_ma", "hazard_wb", "exposure_ma", "exposure_wb",
                                 "risk_map_pfr", "vulnerability_drivers"],
                        help="public risk_ma/risk_wb draw the SAVED smoothed columns on HEALPix cells; "
                             "risk_*_raw are public raw-validation views; private risk views draw raw "
                             "values on building footprints")
    parser.add_argument("--output", required=True, help="Output figure path (.png or .pdf)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--bbox-crs", help="CRS of the bbox (mandatory when --bbox is given)")
    parser.add_argument("--framing", default="fit", choices=["fit", "focus"])
    parser.add_argument("--renderer", default="auto", choices=["auto", "vector", "rasterized"])
    parser.add_argument("--classification-mode", default="head_tail", choices=["head_tail", "precomputed"])
    parser.add_argument("--breaks-file", help="JSON/YAML breaks file (precomputed mode only)")
    parser.add_argument("--source-format", default="auto", choices=["auto", "gpkg", "fgb"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--style-file", help="JSON/YAML per-view palette/label overrides")
    parser.add_argument("--context-file", help="Optional vector file drawn as grey display-only background (e.g. streets)")
    parser.add_argument("--boundary-file", help="Optional vector file drawn as a dashed outline on top (e.g. an Esri admin boundary)")
    parser.add_argument("--basemap", nargs="?", const="positron", default=None,
                        help="Draw web-tile basemap (opt-in network call). Presets like the dashboard: "
                             "positron (Light CARTO), voyager (CARTO Voyager), esri-topo (Esri Topographic), "
                             "satellite (Esri imagery); or any contextily provider key")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    # Heavy imports only after parsing
    from urban_pfr.visualization.static import visualize_output
    result = visualize_output(
        run_dir=args.run_dir, tier=args.tier, view=args.view, output=args.output,
        bbox=list(args.bbox) if args.bbox else None, bbox_crs=args.bbox_crs,
        framing=args.framing, renderer=args.renderer,
        classification_mode=args.classification_mode, breaks_file=args.breaks_file,
        source_format=args.source_format, dpi=args.dpi,
        figsize=tuple(args.figsize) if args.figsize else None,
        style_file=args.style_file, context_file=args.context_file, basemap=args.basemap,
        boundary_file=args.boundary_file)
    print("figure    : %s" % result.output_path)
    print("provenance: %s" % result.provenance_path)
    print("source    : %s (%s, %s)" % (result.source_filename, result.source_role, result.source_crs))
    print("renderer  : %s (%s)" % (result.renderer_used, result.renderer_reason))
    print("features  : %d | breaks: %s" % (result.feature_count,
                                           [round(b, 4) for b in result.classification_breaks]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
