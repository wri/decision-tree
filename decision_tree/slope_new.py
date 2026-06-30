"""Slope batch flow — computes per-polygon slope statistics from Copernicus DEM.

Downloads Copernicus DEM GLO-30 (same source as the TTC canopy-cover step) via
the Earth Search STAC API, computes slope using a gradient operator, and returns
one row of slope statistics per polygon.

This module is the Copernicus-DEM counterpart to ``opentopo_pull_wrapper`` in
``slope.py``. It is intended for side-by-side comparison with the existing
OpenTopo/NASADEM methodology, so the public entrypoint
(:func:`copernicus_pull_wrapper`) mirrors that wrapper's signature and returns a
``feats_df`` merged on ``(project_id, poly_id)``.

NOTE: This pass aligns *schema and I/O only*. The methodological steps (tile
identification, COP-DEM download, the gradient-based slope operator, and the
zonal-stats precision modes) are intentionally left unchanged from the original
script so they can be addressed separately.

Pipeline steps:
  1. identify_polygon_tiles       — X_tile/Y_tile covering each polygon
  2. download_and_compute_slope   — cache per-tile slope GeoTIFF at dest
  3. compute_polygon_slope_stats  — zonal stats per polygon
"""

from __future__ import annotations

import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from odc.stac import configure_rio, stac_load
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.transform import from_bounds
from shapely.geometry import box
from decision_tree.constants import NODATA, HALF_TILE_DEG, DEM_COLLECTION, EARTH_SEARCH_V1, DEFAULT_TILEDB_PATH

def _load_tiledb(path: str):
    """Load tiledb parquet from S3 or local filesystem."""
    if path.startswith("s3://"):
        bucket, _, key = path.removeprefix("s3://").partition("/")
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body), columns=["X_tile", "Y_tile", "X", "Y"])
    return pd.read_parquet(path, columns=["X_tile", "Y_tile", "X", "Y"])


# ---------------------------------------------------------------------------
# Tile identification
# ---------------------------------------------------------------------------

def identify_polygon_tiles(
    gdf: gpd.GeoDataFrame,
    tiledb_path: str = DEFAULT_TILEDB_PATH,
) -> list[dict]:
    """Return the unique (X_tile, Y_tile, lon, lat) tiles covering these polygons.

    Uses tiledb.parquet to map polygon footprints to the tile grid. A polygon
    larger than one tile will generate multiple rows.

    Args:
        gdf: Polygon geometries (with a ``geometry`` column) in any CRS;
            reprojected to EPSG:4326 internally for the tile-grid join.
        tiledb_path: Path to tiledb.parquet (S3 or local).

    Returns:
        List of unique tile dicts with keys: X_tile, Y_tile, lon, lat.
    """
    if gdf.empty:
        return []

    gdf = gdf.to_crs("EPSG:4326")

    # Load tiledb and build bbox geometry per tile
    tiledb = _load_tiledb(tiledb_path)
    tiles_gdf = gpd.GeoDataFrame(
        tiledb,
        geometry=[
            box(r.X - HALF_TILE_DEG, r.Y - HALF_TILE_DEG,
                r.X + HALF_TILE_DEG, r.Y + HALF_TILE_DEG)
            for r in tiledb.itertuples()
        ],
        crs="EPSG:4326",
    )

    # Spatial join: which tiles does each polygon touch
    joined = gpd.sjoin(tiles_gdf, gdf, how="inner", predicate="intersects")
    unique_tiles = (
        joined[["X_tile", "Y_tile", "X", "Y"]]
        .drop_duplicates(subset=["X_tile", "Y_tile"])
        .rename(columns={"X": "lon", "Y": "lat"})
        .to_dict("records")
    )
    print(f"{len(gdf)} polygons → {len(unique_tiles)} unique DEM tiles")
    return unique_tiles


# ---------------------------------------------------------------------------
# DEM download + slope computation
# ---------------------------------------------------------------------------


def _tile_key(X_tile: int, Y_tile: int) -> str:
    return f"tiles/{X_tile}/{Y_tile}/slope_{X_tile}X{Y_tile}Y.tif"


def _s3_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _compute_slope_for_tile(tile: dict, dest: str) -> tuple[int, int, bool]:
    """Download COP-DEM for one tile, compute slope, upload as GeoTIFF.

    Returns (X_tile, Y_tile, skipped) where skipped=True if output already existed.
    """
    X_tile = int(tile["X_tile"])
    Y_tile = int(tile["Y_tile"])
    lon    = float(tile["lon"])
    lat    = float(tile["lat"])

    # Parse dest into bucket/prefix
    bucket, _, prefix = dest.removeprefix("s3://").partition("/")
    key = f"{prefix.rstrip('/')}/{_tile_key(X_tile, Y_tile)}"

    if _s3_exists(bucket, key):
        return X_tile, Y_tile, True

    # Bounds of the tile — slight expansion avoids edge effects in slope computation
    pad = HALF_TILE_DEG * 0.1
    bbox = [lon - HALF_TILE_DEG - pad, lat - HALF_TILE_DEG - pad,
            lon + HALF_TILE_DEG + pad, lat + HALF_TILE_DEG + pad]

    configure_rio(cloud_defaults=True, aws={"requester_pays": True, "region_name": "eu-central-1"})
    client = Client.open(EARTH_SEARCH_V1)
    items  = client.search(collections=[DEM_COLLECTION], bbox=bbox).item_collection()
    if not items:
        raise RuntimeError(f"No COP-DEM items for tile {X_tile}X{Y_tile}Y bbox={bbox}")

    ds = stac_load(items, bands=["data"], bbox=bbox, resampling="bilinear", chunks={})
    elev = ds["data"].isel(time=0).transpose("latitude", "longitude").values.astype("float32")
    if not np.isfinite(elev).all():
        raise ValueError(f"Non-finite values in DEM for tile {X_tile}X{Y_tile}Y")

    # Gradient operator — slope in percent
    dz_dy, dz_dx = np.gradient(elev)
    # Approximate pixel size in metres (COP-DEM 30m at tile latitude)
    dx_m = 30.0 * np.cos(np.deg2rad(lat))
    dy_m = 30.0
    slope_pct = np.sqrt((dz_dx / dx_m) ** 2 + (dz_dy / dy_m) ** 2) * 100.0

    # Crop back to original tile (unpad)
    h, w = slope_pct.shape
    pad_h = int(h * (pad / (HALF_TILE_DEG * 2 + 2 * pad)))
    pad_w = int(w * (pad / (HALF_TILE_DEG * 2 + 2 * pad)))
    if pad_h > 0 and pad_w > 0:
        slope_pct = slope_pct[pad_h:h - pad_h, pad_w:w - pad_w]

    h2, w2 = slope_pct.shape
    tile_bounds = (
        lon - HALF_TILE_DEG, lat - HALF_TILE_DEG,
        lon + HALF_TILE_DEG, lat + HALF_TILE_DEG,
    )
    transform = from_bounds(*tile_bounds, width=w2, height=h2)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        local = tmp.name
    with rasterio.open(
        local, "w", driver="GTiff",
        width=w2, height=h2, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform, compress="lzw", nodata=NODATA,
    ) as dst:
        dst.write(slope_pct.astype("float32"), 1)

    boto3.client("s3").upload_file(local, bucket, key)
    os.unlink(local)
    return X_tile, Y_tile, False


def download_and_compute_slope(tiles: list[dict], dest: str, max_workers: int = 8) -> None:
    """Parallel per-tile DEM download + slope computation. Skips cached tiles."""
    if not tiles:
        print("No tiles to process.")
        return

    print(f"Starting slope compute for {len(tiles)} tiles (max_workers={max_workers})")
    done, skipped, failed = 0, 0, 0
    first_error: str | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_compute_slope_for_tile, t, dest): t for t in tiles}
        for fut in as_completed(futures):
            tile = futures[fut]
            try:
                _, _, was_skipped = fut.result()
                if was_skipped:
                    skipped += 1
                else:
                    done += 1
            except Exception as exc:
                failed += 1
                msg = f"Tile {tile['X_tile']}X{tile['Y_tile']}Y failed: {type(exc).__name__}: {exc}"
                print(msg)
                if first_error is None:
                    first_error = msg

    print(f"Slope tiles: {done} computed, {skipped} cached, {failed} failed")
    if failed and first_error and done == 0 and skipped == 0:
        # All attempts failed — raise so the caller sees the error rather than
        # silently continuing with no slope data.
        raise RuntimeError(f"All {failed} tiles failed. First error: {first_error}")


# ---------------------------------------------------------------------------
# Zonal stats
# ---------------------------------------------------------------------------


def _stats_from_values(vals: np.ndarray, steep_threshold: float) -> dict:
    """Compute the output stats dict from a 1-D array of per-pixel slope values."""
    if vals.size == 0:
        return {"mean_slope": None, "max_slope": None, "median_slope": None,
                "slope_area": None}
    pct_gt = float(np.mean(vals > steep_threshold) * 100.0)
    return {
        "mean_slope":   round(float(np.mean(vals)), 2),
        "max_slope":    round(float(np.max(vals)), 2),
        "median_slope": round(float(np.median(vals)), 2),
        "slope_area":   round(pct_gt, 2),
    }


def _compute_stats_numpy(gdf, mosaic, transform, steep_threshold: float) -> dict[str, dict]:
    """Binary rasterized polygon mask. Fast; small edge error at polygon boundaries."""
    arr = mosaic[0]
    results: dict[str, dict] = {}
    for poly_id, geom in zip(gdf["poly_id"], gdf.geometry, strict=False):
        try:
            mask = geometry_mask(
                [geom], out_shape=arr.shape, transform=transform,
                invert=True, all_touched=False,
            )
            vals = arr[mask & (arr > -9000)]
            results[poly_id] = _stats_from_values(vals, steep_threshold)
        except Exception as exc:
            print(f"WARNING: Stats failed for polygon {poly_id}: {exc}")
            results[poly_id] = _stats_from_values(np.array([]), steep_threshold)
    return results


def _compute_stats_exactextract(gdf, mosaic, transform, crs, steep_threshold: float) -> dict[str, dict]:
    """Fractional-coverage weighting at polygon edges (higher precision).

    Writes two in-memory rasters to temp GeoTIFFs:
      1. slope values (for mean/max/median)
      2. binary `slope > steep_threshold` (so ``mean`` on it gives the
         fractional area above threshold — exactextract has no native
         threshold op, but a binary raster + ``mean`` is equivalent).
    """
    arr = mosaic[0]
    valid_mask = arr > -9000

    # Temp: slope raster
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as t:
        slope_path = t.name
    with rasterio.open(
        slope_path, "w", driver="GTiff",
        width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32",
        crs=crs, transform=transform, nodata=NODATA, compress="lzw",
    ) as dst:
        dst.write(arr.astype("float32"), 1)

    # Temp: binary "steep" raster. Set nodata pixels to nodata so they don't
    # bias the weighted mean.
    binary = np.where(valid_mask & (arr > steep_threshold), 1.0, 0.0).astype("float32")
    binary[~valid_mask] = NODATA
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as t:
        binary_path = t.name
    with rasterio.open(
        binary_path, "w", driver="GTiff",
        width=binary.shape[1], height=binary.shape[0], count=1, dtype="float32",
        crs=crs, transform=transform, nodata=NODATA, compress="lzw",
    ) as dst:
        dst.write(binary, 1)

    # exactextract wants poly_id as str
    ee_gdf = gdf.copy()
    ee_gdf["poly_id"] = ee_gdf["poly_id"].astype(str)

    stats_df = exact_extract(
        slope_path, ee_gdf,
        ["mean(min_coverage_frac=0.05, coverage_weight=fraction)",
         "max",
         "quantile(q=0.5)"],  # median
        include_cols=["poly_id"], output="pandas",
    )
    steep_df = exact_extract(
        binary_path, ee_gdf,
        "mean(min_coverage_frac=0.05, coverage_weight=fraction)",
        include_cols=["poly_id"], output="pandas",
    )

    steep_frac = dict(zip(steep_df["poly_id"], steep_df["mean"], strict=False))

    results: dict[str, dict] = {}
    for _, r in stats_df.iterrows():
        pid = r["poly_id"]
        mean_v   = r.get("mean")
        max_v    = r.get("max")
        median_v = r.get("quantile_50")
        pct_gt   = steep_frac.get(pid)
        if mean_v is None or (isinstance(mean_v, float) and np.isnan(mean_v)):
            results[pid] = _stats_from_values(np.array([]), steep_threshold)
            continue
        pct_gt_pct = float(pct_gt) * 100.0 if pct_gt is not None else 0.0
        results[pid] = {
            "mean_slope":   round(float(mean_v), 2),
            "max_slope":    round(float(max_v), 2),
            "median_slope": round(float(median_v), 2),
            "slope_area":   round(pct_gt_pct, 2),
        }

    for p in (slope_path, binary_path):
        try:
            os.unlink(p)
        except OSError:
            pass
    return results


def compute_polygon_slope_stats(
    gdf: gpd.GeoDataFrame,
    dest: str,
    steep_threshold: float,
    precision: str = "numpy",
) -> dict[str, dict]:
    """Zonal slope statistics per polygon.

    Args:
        gdf: Polygon geometries with a ``poly_id`` column. Reprojected to
            EPSG:4326 internally to match the cached slope tiles.
        dest: S3 prefix where per-tile slope GeoTIFFs are cached.
        steep_threshold: % slope above which a pixel counts as "steep"
            (``params['criteria']['slope_thresh']``).
        precision: ``"numpy"`` (default — fast, binary mask) or
            ``"exactextract"`` (fractional-coverage weighting at polygon
            edges; slightly more accurate, slightly slower).

    Returns:
        Mapping of poly_id → stats dict with keys: mean_slope, max_slope,
        median_slope, slope_area.
    """
    if gdf.empty:
        return {}

    gdf = gdf.to_crs("EPSG:4326")

    tile_paths = _download_slope_tiles_for_polygons(gdf, dest)
    if not tile_paths:
        return {}

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(srcs, nodata=NODATA)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    if precision == "exactextract":
        results = _compute_stats_exactextract(gdf, mosaic, transform, crs, steep_threshold)
    elif precision == "numpy":
        results = _compute_stats_numpy(gdf, mosaic, transform, steep_threshold)
    else:
        raise ValueError(f"Unknown precision={precision!r} — use 'numpy' or 'exactextract'")

    for p in tile_paths:
        try:
            os.unlink(p)
        except OSError:
            pass

    print(f"Computed slope stats for {len(results)} polygons (precision={precision})")
    return results


def _download_slope_tiles_for_polygons(gdf, dest: str) -> list[str]:
    """Look up which tiles cover the polygons, download each slope GeoTIFF locally."""
    bucket, _, prefix = dest.removeprefix("s3://").partition("/")
    prefix = prefix.rstrip("/")

    tiledb = _load_tiledb(DEFAULT_TILEDB_PATH)
    tiles_gdf = gpd.GeoDataFrame(
        tiledb,
        geometry=[box(r.X - HALF_TILE_DEG, r.Y - HALF_TILE_DEG,
                      r.X + HALF_TILE_DEG, r.Y + HALF_TILE_DEG)
                  for r in tiledb.itertuples()],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(tiles_gdf, gdf, how="inner", predicate="intersects")
    unique = joined[["X_tile", "Y_tile"]].drop_duplicates()

    s3 = boto3.client("s3")
    local_paths: list[str] = []
    tmp_dir = tempfile.mkdtemp(prefix="slope_tiles_")
    for row in unique.itertuples():
        key = f"{prefix}/{_tile_key(int(row.X_tile), int(row.Y_tile))}"
        local = os.path.join(tmp_dir, f"slope_{row.X_tile}X{row.Y_tile}Y.tif")
        try:
            s3.download_file(bucket, key, local)
            local_paths.append(local)
        except Exception as exc:
            print(f"WARNING: Could not download {key}: {exc}")
    return local_paths



def copernicus_pull_wrapper(
    params,
    geojson_dir,
    feats_df,
    precision: str = "numpy",
    max_workers: int = 8,
):
    """Copernicus-DEM slope statistics, returned as a feats_df merge.

    Drop-in counterpart to ``slope.opentopo_pull_wrapper``: iterates the
    projects in ``feats_df``, reads each project's polygons from
    ``geojson_dir``, computes per-polygon slope statistics from the Copernicus
    DEM, and merges the results back into ``feats_df`` on
    ``(project_id, poly_id)``.

    Args:
        params: Parsed params.yaml.
        secrets: Parsed secrets (currently unused; kept for signature parity
            with opentopo_pull_wrapper. COP-DEM access uses default boto3
            credentials / requester-pays).
        geojson_dir: Directory of per-project ``{name}_{data_version}.geojson``.
        feats_df: Feature table with project_name, project_id, poly_id.
        dest: S3 prefix for cached slope GeoTIFF tiles (shared across projects).
        precision: ``"numpy"`` (default) or ``"exactextract"``.
        max_workers: Tile-download concurrency.

    Returns:
        ``feats_df`` left-merged with columns: mean_slope, max_slope,
        median_slope, slope_area. Polygons with no slope data are NaN.
    """
    slope_thresh = params['criteria']['slope_thresh']
    data_version = params['outfile']['data_version']
    dest = params['s3']['slope']

    out_cols = ['mean_slope', 'max_slope', 'median_slope', 'slope_area']
    project_names = feats_df['project_name'].unique()
    dfs_to_concat = []

    for name in project_names:
        project_df = feats_df[feats_df.project_name == name]
        project_id = project_df.project_id.iloc[0]

        geojson_path = os.path.join(geojson_dir, f"{name}_{data_version}.geojson")
        if not os.path.exists(geojson_path):
            print(f"WARNING: geojson not found for {name}: {geojson_path}")
            continue
        project_polygons = gpd.read_file(geojson_path)
        print(f"Processing {name} ({len(project_polygons)} polygons)")

        tiles = identify_polygon_tiles(project_polygons)
        if not tiles:
            print(f"No DEM tiles for {name}, skipping.")
            continue

        download_and_compute_slope(tiles, dest, max_workers=max_workers)
        stats = compute_polygon_slope_stats(
            project_polygons, dest, slope_thresh, precision=precision
        )
        if not stats:
            continue

        # Build rows keyed on the project's original poly_id dtype so the merge
        # into feats_df is reliable regardless of how the stats dict was keyed.
        records = []
        for pid in project_polygons['poly_id']:
            s = stats.get(pid)
            if s is None:
                s = stats.get(str(pid))
            if s is None:
                continue
            records.append({
                'project_id': project_id,
                'poly_id': pid,
                **{c: s[c] for c in out_cols},
            })

        if records:
            dfs_to_concat.append(pd.DataFrame(records))

    if not dfs_to_concat:
        # Nothing computed — return feats_df with empty slope columns so the
        # downstream merge / classification still works.
        comb = feats_df.copy()
        for c in out_cols:
            comb[c] = np.nan
        return comb

    result_df = pd.concat(dfs_to_concat, ignore_index=True)

    # merge slope results into feats df (polys with missing slope become NaN)
    comb = feats_df.merge(
        result_df,
        on=['project_id', 'poly_id'],
        how='left',
    )
    return comb


def apply_slope_classification(params, df, slope_stats):
    '''
    each polygon already has a pre-computed number identifying the percentage of the polygon's
    area that has a steep slope (>threshold). this function converts that number into simple
    flat/steep label using the threshold

    - "steep" if the % of area > threshold.
    - "flat" if the % of area <= threshold.
    - NaN if slope_area is NaN (no data).

    Applies to all polygons, but downstream decision tree will filter for "remote" rows.
    '''
    n_projects = slope_stats['project_id'].nunique()
    n_polys = slope_stats['poly_id'].nunique()
    print(f"Analyzing slope for {n_projects} projects and {n_polys} polygons...")

    slope_thresh = params['criteria']['slope_thresh']
    slope_stats = slope_stats[['project_id', 'poly_id', 'slope_area']].copy()

    def classify_slope(val):
        if pd.isna(val):
            return 'missing'
        elif val > slope_thresh:
            return 'steep'
        else:
            return 'flat'

    # Apply classification
    slope_stats.loc[:, 'slope'] = slope_stats['slope_area'].apply(classify_slope)

    # Merge into main DataFrame
    comb = df.merge(
        slope_stats[['project_id', 'poly_id', 'slope_area', 'slope']],
        on=['project_id', 'poly_id'],
        how='left'
    )

    return comb
