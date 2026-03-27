"""
thiessen.py – Thiessen (Voronoi) polygon generation from building centroids.

Replaces building footprints with Thiessen polygons clipped to the study area,
creating a continuous tessellation for visualization (matching the ArcGIS workflow).
"""

import numpy as np
import geopandas as gpd
from shapely.ops import voronoi_diagram, unary_union
from shapely.geometry import MultiPoint
from shapely.strtree import STRtree
from shapely import prepared


def create_thiessen_polygons(buildings_gdf, boundary=None, simplify_tolerance=10):
    """
    Replace building footprints with Thiessen (Voronoi) polygons.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with risk values. Geometries will be replaced.
    boundary : Geometry or GeoDataFrame, optional
        Study area boundary to clip Thiessen polygons.
        If None, derived from the convex hull of all buildings (buffered).
        If GeoDataFrame, uses unary_union of its geometries.
    simplify_tolerance : float
        Tolerance (in CRS units, typically meters) for simplifying the
        boundary before clipping. Speeds up intersection operations.

    Returns
    -------
    GeoDataFrame
        Same attributes as input, but with Thiessen polygon geometries.
    """
    n = len(buildings_gdf)
    print(f"  Thiessen: generating polygons for {n} buildings...")

    # Resolve boundary
    if boundary is None:
        print("    Deriving boundary from building footprints...")
        boundary = unary_union(buildings_gdf.geometry).convex_hull.buffer(200)
    elif isinstance(boundary, gpd.GeoDataFrame):
        boundary = unary_union(boundary.geometry)

    # Simplify boundary for fast clipping
    boundary_simple = boundary.simplify(simplify_tolerance)
    n_verts_orig = _count_vertices(boundary)
    n_verts_simple = _count_vertices(boundary_simple)
    print(f"    Boundary: {boundary_simple.area / 1e6:.1f} km², "
          f"simplified {n_verts_orig} → {n_verts_simple} vertices")

    # Compute Voronoi from building centroids
    centroids = buildings_gdf.geometry.centroid
    mp = MultiPoint(list(zip(centroids.x, centroids.y)))
    regions = voronoi_diagram(mp, envelope=boundary_simple.buffer(500))
    voronoi_geoms = list(regions.geoms)
    print(f"    Voronoi: {len(voronoi_geoms)} regions")

    # Match each building centroid to its Voronoi cell
    cell_tree = STRtree(voronoi_geoms)
    print("    Matching buildings to Voronoi cells...")
    matched = [voronoi_geoms[cell_tree.nearest(centroids.iloc[i])]
               for i in range(n)]

    # Clip to boundary using prepared geometry for speed
    print("    Clipping to study area boundary...")
    prep_boundary = prepared.prep(boundary_simple)
    clipped = []
    n_inside = 0
    n_clipped = 0
    for i, g in enumerate(matched):
        if i % 50000 == 0 and i > 0:
            print(f"      {i}/{n}...")
        if prep_boundary.contains(g):
            clipped.append(g)
            n_inside += 1
        elif prep_boundary.intersects(g):
            clipped.append(g.intersection(boundary_simple))
            n_clipped += 1
        else:
            clipped.append(g)

    print(f"    {n_inside} fully inside, {n_clipped} clipped at boundary")

    # Build result
    result = buildings_gdf.copy()
    result['geometry'] = clipped
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=buildings_gdf.crs)
    result = result[~result.geometry.is_empty & result.geometry.notna()]

    area = result.geometry.area
    print(f"  Thiessen complete: {len(result)} polygons, "
          f"median area={area.median():.0f} m², "
          f"total={area.sum() / 1e6:.1f} km²")

    return result


def _count_vertices(geom):
    """Count total vertices in a geometry."""
    if hasattr(geom, 'geoms'):
        return sum(_count_vertices(g) for g in geom.geoms)
    if hasattr(geom, 'exterior'):
        return len(geom.exterior.coords) + sum(
            len(r.coords) for r in geom.interiors)
    return 0
