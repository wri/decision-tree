#from src.logs import get_logger
import pandas as pd
import geopandas as gpd
import fiona
import pyproj
from shapely import buffer
from shapely.geometry import Polygon, MultiPolygon, LineString, shape, mapping
from shapely.ops import unary_union
import ast


def shp_to_gdf(shapefile_path, driver):
    shapefile_features = []
    with fiona.collection(shapefile_path, driver=driver) as shp_in:
        shp_crs = pyproj.crs.CRS(shp_in.crs)
        shpschema = shp_in.schema.copy()
        for obj in shp_in:
            shapefile_features.append(obj)
    shp_gdf = gpd.GeoDataFrame.from_features(
        [feature for feature in shapefile_features], crs=shp_crs
    )
    shp_gdf["bounds"] = shp_gdf.geometry.apply(lambda x: x.bounds)
    shp_gdf["bounds"] = [list(i) for i in shp_gdf["bounds"]]
    return shp_gdf


def calculate_area_km2(geometry):
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    area_m2 = gdf.geometry.area.values[0]
    area_km2 = area_m2 * 10**-6
    return area_km2

def join_polygons_by_shortest_line(polygons):
    """
    Join all polygons into a single polygon by connecting them with the shortest lines.
    This function handles both Polygons and MultiPolygons.

    :param polygons: List of Shapely Polygon or MultiPolygon objects.
    :return: A single Shapely Polygon that connects all input polygons.
    """
    # Flatten the list of polygons to handle both Polygon and MultiPolygon
    all_polygons = []
    for geom in polygons:
        if isinstance(geom, MultiPolygon):
            # If the geometry is a MultiPolygon, iterate over geom.geoms to get individual Polygons
            for geom2 in geom.geoms:
                all_polygons.append(buffer(geom2, 0.001))  # 0.001
            # all_polygons.extend(geom.geoms)
        else:
            all_polygons.append(buffer(geom, 0.001))  # 0.001

    # Create a list of all the polygons as initial geometries
    all_geometries = all_polygons[:]

    # Helper function to find the closest pair of polygons by distance
    def find_closest_pair_of_polygons():
        closest_distance = float("inf")
        closest_pair = None
        for i, poly1 in enumerate(all_polygons):
            for j, poly2 in enumerate(all_polygons):
                if i != j:
                    # Check if poly1 and poly2 are individual polygons or MultiPolygons
                    if isinstance(poly1, MultiPolygon):
                        poly1_exteriors = [p.exterior for p in poly1.geoms]
                    else:
                        poly1_exteriors = [poly1.exterior]

                    if isinstance(poly2, MultiPolygon):
                        poly2_exteriors = [p.exterior for p in poly2.geoms]
                    else:
                        poly2_exteriors = [poly2.exterior]

                    # Calculate the distance between each pair of exteriors
                    for ext1 in poly1_exteriors:
                        for ext2 in poly2_exteriors:
                            distance = ext1.distance(ext2)
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_pair = (i, j)
        return closest_pair

    # Keep connecting the closest polygons until all are joined
    while len(all_polygons) > 1:
        # Find the closest pair of polygons
        i, j = find_closest_pair_of_polygons()

        # Get the polygons to connect
        poly1 = all_polygons[i]
        poly2 = all_polygons[j]

        # Create a LineString connecting the two closest polygons (centroids for simplicity)
        line = LineString([poly1.centroid, poly2.centroid])

        # Add the line to the geometries list
        all_geometries.append(line)

        # Remove the two polygons from the list and add a union of them
        new_polygon = poly1.union(poly2)
        all_polygons.remove(poly1)
        all_polygons.remove(poly2)
        all_polygons.append(new_polygon)

    # Once all polygons are connected, return the union of all geometries as a single polygon
    final_geometry = unary_union(all_geometries)

    # Return the final single polygon
    return final_geometry

def convert_geometry(poly_geom, debug=False):
    """
    Converts various geometry formats (WKT, stringified GeoJSON, dict, or Shapely object)
    into a proper Shapely geometry object.
    """
    if isinstance(poly_geom, str):
        poly_geom = poly_geom.strip()

        if poly_geom.startswith('POLYGON') or poly_geom.startswith('MULTIPOLYGON'):
            if debug:
                print("Detected WKT format. Converting using wkt.loads()...")
            return wkt.loads(poly_geom)
        else:
            try:
                if debug:
                    print("Assuming GeoJSON format. Converting using ast.literal_eval() and shape()...")
                return shape(ast.literal_eval(poly_geom))
            except (SyntaxError, ValueError, TypeError, json.JSONDecodeError) as e:
                if debug:
                    print(f"❌ Could not parse geometry string: {e}")
                return None
    elif isinstance(poly_geom, dict):
        return shape(poly_geom)
    elif hasattr(poly_geom, "geom_type"):  # likely already a Shapely geometry
        return poly_geom
    
    if debug:
        print("❌ Unrecognized geometry format.")
    return None

def df_to_geojson(df, geometry_col='geometry', output_path='output.geojson', crs='EPSG:4326'):
    """
    Converts a DataFrame with geometries in various formats into a GeoDataFrame and exports to GeoJSON.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a geometry column
        geometry_col (str): Name of the column containing geometry
        output_path (str): File path to save the output GeoJSON
        crs (str): Coordinate reference system to assign
        debug (bool): If True, print conversion steps and errors
    Returns:
        gpd.GeoDataFrame: The resulting GeoDataFrame
    """
    df = df.copy()

    # Convert geometry column to shapely objects
    df[geometry_col] = df[geometry_col].apply(lambda g: convert_geometry(g, debug=False))

    # Drop rows where conversion failed
    df = df[df[geometry_col].notnull()]

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry_col)
    gdf.set_crs(crs, inplace=True)

    # Export to GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"Exported {len(gdf)} features to {output_path}")