import argparse
import json
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox

import graph_utils
from routing_engine import RoutingEngine
from llm_agent import LLMAgent
from context_engine import ContextEngine
from shapely.geometry import LineString

# Resolve project root from this file's location.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)


def _load_config() -> dict:
    config_path = os.path.join(_PROJECT_ROOT, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _get_model_name(method: str, config: dict) -> str:
    llm_cfg = config.get("llm", {})
    if method == 'hf':
        return llm_cfg.get("hf_model", "Qwen/Qwen3-4B")
    if method == 'ollama':
        return llm_cfg.get("ollama_model", "qwen:4b")
    if method == 'openai':
        return llm_cfg.get("openai_model", "gpt-3.5-turbo")
    return llm_cfg.get("hf_model", "Qwen/Qwen3-4B")


def plot_final_route(router: RoutingEngine, path: list, pois_info: list, output_path: str = "rota_final.png"):
    """Plot the optimised route on an OSMnx map and save it to disk."""
    print(f"\nGenerating route map at '{output_path}'...")
    if not path:
        print("Empty path — cannot generate map.")
        return

    poi_lons = [poi['lon'] for poi in pois_info]
    poi_lats = [poi['lat'] for poi in pois_info]

    path_lons = [router.graph.nodes[n]['x'] for n in path]
    path_lats = [router.graph.nodes[n]['y'] for n in path]

    margin = 0.5
    lon_range = max(path_lons) - min(path_lons)
    lat_range = max(path_lats) - min(path_lats)
    bbox = (
        min(path_lats) - lat_range * margin,
        max(path_lats) + lat_range * margin,
        min(path_lons) - lon_range * margin,
        max(path_lons) + lon_range * margin
    )

    _, ax = ox.plot_graph_route(
        router.graph, path,
        route_color='red',
        route_linewidth=6,
        node_size=0,
        bgcolor='white',
        edge_color='#333333',
        edge_linewidth=0.5,
        show=False,
        close=False,
        figsize=(12, 12)
    )

    if poi_lons and poi_lats:
        ax.scatter(
            poi_lons, poi_lats,
            c='blue', s=150, zorder=5,
            label='Stops (POIs)', edgecolors='black', linewidths=1.5
        )

    ax.set_title("Optimised Route with Stops", fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print("Map saved successfully.")


def geometry_to_latlon(geom):
    """Return (lat, lon) for any shapely geometry."""
    if geom is None:
        return None, None
    point = geom if geom.geom_type == "Point" else geom.representative_point()
    return point.y, point.x


def main():
    config = _load_config()
    llm_cfg = config.get("llm", {})
    out_cfg = config.get("output", {})

    default_method = llm_cfg.get("default_method", "hf")
    default_output = out_cfg.get("route_image", "rota_final.png")

    parser = argparse.ArgumentParser(
        description="VAMOS — Vehicular Agent for Multi-objective Optimization and Semantics",
        allow_abbrev=False
    )
    parser.add_argument(
        "--origin", "--origem",
        dest="origin", required=True, type=str,
        help="Departure address or coordinates (e.g. 'Av. Paulista, São Paulo' or '-23.56,-46.65')."
    )
    parser.add_argument(
        "--destination", "--destino",
        dest="destination", required=True, type=str,
        help="Destination address or coordinates."
    )
    parser.add_argument(
        "--tasks", "--tarefas",
        dest="tasks", nargs='+', default=[],
        help="List of tasks (e.g. 'stop at a pharmacy' 'buy bread')."
    )
    parser.add_argument(
        "--method",
        default=default_method,
        choices=['ollama', 'hf', 'openai'],
        help="LLM inference backend to use (default from config.json)."
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Output file path for the route map image (default from config.json)."
    )
    parser.add_argument(
        "--force-download", "--force_download",
        dest="force_download", action="store_true",
        help="Force re-download of the street graph and POIs, ignoring cache."
    )
    args = parser.parse_args()

    model_name = _get_model_name(args.method, config)

    print("--- Starting VAMOS Routing System ---")

    graph = graph_utils.get_graph(args.force_download)
    pois_gdf = graph_utils.get_pois(graph, args.force_download)

    projected_graph = ox.project_graph(graph)
    projected_pois = (
        pois_gdf.to_crs(projected_graph.graph["crs"]) if not pois_gdf.empty else pois_gdf
    )

    router = RoutingEngine(graph)
    language_model = LLMAgent(model_name=model_name, method=args.method)
    context_provider = ContextEngine()

    origin_node = router.address_to_node(args.origin)
    destination_node = router.address_to_node(args.destination)
    if not all([origin_node, destination_node]):
        print("Error: could not resolve origin/destination addresses. Check the input values.")
        return

    task_analysis = language_model.classify_tasks(args.tasks)
    tasks = sorted(task_analysis.get("tasks", []), key=lambda t: t.get('importance', 0), reverse=True)

    candidate_routes = []

    direct_path, direct_cost = router.find_shortest_path(origin_node, destination_node)

    if direct_path:
        candidate_routes.append({
            "description": "Direct Route",
            "path": direct_path,
            "cost_seconds": direct_cost,
            "tasks_completed": [],
            "pois_info": []
        })

    if tasks and direct_path:
        print("\nTasks ranked by importance:")
        for task in tasks:
            print(f"  - {task['task']} (importance: {task.get('importance', 'N/A')}) -> POI: {task['poi_tags']}")

        waypoint_nodes = []
        waypoint_task_map = {}

        for task in tasks:
            tags = task.get("poi_tags", {})
            if not tags:
                continue
            tag_key, tag_value = list(tags.items())[0]
            if tag_key not in projected_pois.columns:
                print(f"Warning: POI column '{tag_key}' not available for task '{task['task']}'. Skipping.")
                continue
            matching_pois = projected_pois[projected_pois[tag_key] == tag_value]

            if not matching_pois.empty:
                route_coords = [
                    (projected_graph.nodes[n]["x"], projected_graph.nodes[n]["y"])
                    for n in direct_path
                ]
                route_line = LineString(route_coords)

                matching_pois = matching_pois.copy()
                matching_pois["dist_to_route"] = matching_pois.geometry.apply(
                    lambda geom: geom.distance(route_line)
                )

                best_poi = matching_pois.sort_values("dist_to_route").iloc[0]
                poi_node = best_poi["nearest_node"]
                if poi_node not in waypoint_nodes:
                    waypoint_nodes.append(poi_node)
                    waypoint_task_map[poi_node] = task
            else:
                print(f"Warning: no POI found for task '{task['task']}'")

        if waypoint_nodes:
            print(f"\nOptimising visit order for {len(waypoint_nodes)} waypoint(s)...")
            multi_stop_path, multi_stop_cost, ordered_stops = router.find_optimal_route_for_pois(
                origin_node, destination_node, waypoint_nodes
            )

            if multi_stop_path:
                stops_info = [
                    pois_gdf[pois_gdf['nearest_node'] == node].iloc[0]
                    for node in ordered_stops
                    if node not in [origin_node, destination_node]
                ]
                candidate_routes.append({
                    "description": "Route with Stops (multi-task)",
                    "path": multi_stop_path,
                    "cost_seconds": multi_stop_cost,
                    "tasks_completed": [
                        waypoint_task_map[node]
                        for node in ordered_stops
                        if node in waypoint_task_map
                    ],
                    "pois_info": [
                        {
                            "name": stop.get("name", "POI"),
                            "lat": geometry_to_latlon(stop.geometry)[0],
                            "lon": geometry_to_latlon(stop.geometry)[1],
                        }
                        for stop in stops_info
                    ]
                })
            else:
                print(
                    "\nWarning: could not compute a route connecting all waypoints. "
                    "This may be due to ambiguous addresses or very distant POIs. "
                    "Only the direct route will be considered."
                )

    if len(candidate_routes) <= 1:
        final_route = candidate_routes[0] if candidate_routes else None
        justification = "Direct route selected — no tasks or no viable multi-stop route."
    else:
        print("\nCandidate routes submitted to LLM:")
        for index, route in enumerate(candidate_routes):
            print(f"  Route {index + 1}: {route['description']} — {route['cost_seconds'] / 60:.2f} min")

        user_context = context_provider.get_user_context()
        origin_coords = router.get_node_coords(origin_node)
        destination_coords = router.get_node_coords(destination_node)
        scenario_context = context_provider.get_scenario_context(origin_coords, destination_coords)

        llm_evaluation = language_model.evaluate_routes(user_context, scenario_context, candidate_routes)

        chosen_id = llm_evaluation.get("chosen_route_id", 1)
        justification = llm_evaluation.get("justification", "No justification provided.")
        final_route = candidate_routes[chosen_id - 1]

    if final_route:
        print("\n--- Agent Final Decision ---")
        print(f"Justification: {justification}")
        print("\nRecommended Route Details:")
        print(f"  Description     : {final_route['description']}")
        print(f"  Estimated time  : {final_route['cost_seconds'] / 60:.2f} minutes")
        if final_route['tasks_completed']:
            task_names = [task['task'] for task in final_route['tasks_completed']]
            print(f"  Tasks covered   : {', '.join(task_names)}")

        plot_final_route(router, final_route['path'], final_route['pois_info'], args.output)
    else:
        print("\nCould not determine a final route.")

    language_model.print_timing_report()


if __name__ == "__main__":
    main()
