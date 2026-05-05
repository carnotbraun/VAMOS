import os
import json
import osmnx as ox
import geopandas as gpd
import warnings

# Resolve the project root from this source file's location so that paths
# work correctly regardless of the current working directory.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)

def _load_config() -> dict:
    config_path = os.path.join(_PROJECT_ROOT, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

_CONFIG = _load_config()

LOCATION = _CONFIG.get("routing", {}).get("location", "São Paulo, Brazil")
_cache_subdir = _CONFIG.get("routing", {}).get("graph_cache_dir", "graph_data")
GRAPH_CACHE_DIR = os.path.join(_PROJECT_ROOT, _cache_subdir, LOCATION.split(',')[0].strip())
os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

_location_slug = LOCATION.split(',')[0].strip().lower()
GRAPH_FILENAME = os.path.join(GRAPH_CACHE_DIR, f"{_location_slug}_drive_graph.graphml")
POI_FILENAME = os.path.join(GRAPH_CACHE_DIR, f"{_location_slug}_pois.gpkg")

# OSM tags for points of interest to download and cache.
POI_TAGS = {
    "amenity": ["hospital", "pharmacy", "fuel", "restaurant", "cafe", "bar"],
    "shop": ["supermarket", "bakery", "mall"],
    "leisure": ["park", "stadium"]
}


def get_graph(force_download: bool = False):
    """Load the street graph from local cache or download it from OpenStreetMap."""
    if os.path.exists(GRAPH_FILENAME) and not force_download:
        print(f"Loading graph from cache: '{GRAPH_FILENAME}'")
        return ox.load_graphml(GRAPH_FILENAME)

    print(f"Downloading street graph for '{LOCATION}' from OpenStreetMap...")
    graph = ox.graph_from_place(LOCATION, network_type='drive')

    print("Adding speed and travel time data to edges...")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)

    print(f"Saving graph to cache: '{GRAPH_FILENAME}'")
    ox.save_graphml(graph, GRAPH_FILENAME)

    return graph


def get_pois(graph, force_download: bool = False):
    """Load points of interest from cache or download from OpenStreetMap.

    Returns a GeoDataFrame with cleaned POI data and nearest graph node ids.
    """
    if os.path.exists(POI_FILENAME) and not force_download:
        print(f"Loading POIs from cache: '{POI_FILENAME}'")
        return gpd.read_file(POI_FILENAME)

    print(f"Downloading POIs for '{LOCATION}' from OpenStreetMap...")
    pois_gdf = ox.features_from_place(LOCATION, tags=POI_TAGS)

    if pois_gdf.empty:
        print("Warning: no POIs found for the specified tags.")
        return pois_gdf

    print("Processing and cleaning POI data...")

    # Project to a metric CRS to compute accurate centroids, then reproject back.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois_centroids = pois_gdf.to_crs(epsg=3857).geometry.centroid.to_crs(pois_gdf.crs)

    print("Associating POIs with nearest graph nodes...")
    pois_gdf['nearest_node'] = ox.nearest_nodes(graph, pois_centroids.x, pois_centroids.y)

    # Keep only essential columns and requested tag keys.
    # Extra columns (e.g. 'fuel:Gasoline') break the GeoPackage serializer.
    columns_to_keep = ['name', 'geometry', 'nearest_node']
    for tag_key in POI_TAGS.keys():
        if tag_key in pois_gdf.columns:
            columns_to_keep.append(tag_key)

    pois_gdf = pois_gdf[columns_to_keep]

    print(f"Saving POIs to cache: '{POI_FILENAME}'")
    try:
        pois_gdf.to_file(POI_FILENAME, driver='GPKG')
    except Exception as exc:
        print(f"Warning: could not save POI cache ({exc}); execution will continue.")

    return pois_gdf
