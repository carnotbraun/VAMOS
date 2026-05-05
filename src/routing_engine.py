import osmnx as ox
import networkx as nx
from itertools import permutations
import pandas as pd


class RoutingEngine:
    """Computes vehicle routes over an OSMnx street graph.

    Supports single-destination shortest paths and multi-stop optimisation
    via a brute-force Travelling Salesman Problem (TSP) solver.
    """

    def __init__(self, graph):
        if graph is None:
            raise ValueError("The provided graph cannot be None.")
        self.graph = graph

    def address_to_node(self, location: str) -> int:
        """Convert a street address or 'lat, lon' string to the nearest graph node id.

        Tries to parse the input as numeric coordinates first; falls back to
        geocoding via the Nominatim API if parsing fails.
        """
        try:
            if "," in location and any(c.isdigit() for c in location):
                parts = [float(part.strip()) for part in location.split(",")]
                if len(parts) == 2:
                    lat, lon = parts
                    return ox.nearest_nodes(self.graph, lon, lat)
        except ValueError:
            pass

        try:
            print(f"Geocoding address: '{location}'...")
            lat, lon = ox.geocode(location)
            return ox.nearest_nodes(self.graph, lon, lat)
        except Exception as exc:
            print(f"Geocoding error for '{location}': {exc}")
            return None

    def get_node_coords(self, node: int) -> tuple:
        """Return (lat, lon) coordinates for a graph node."""
        return (self.graph.nodes[node]['y'], self.graph.nodes[node]['x'])

    def find_shortest_path(self, origin_node: int, destination_node: int, weight: str = 'travel_time'):
        """Find the shortest path between two nodes.

        Returns (path, cost) where path is a list of node ids and cost is in
        the unit of the chosen weight (seconds for travel_time).
        Returns (None, inf) if no path exists.
        """
        try:
            path = nx.shortest_path(
                self.graph, source=origin_node, target=destination_node, weight=weight
            )
            cost = nx.shortest_path_length(
                self.graph, source=origin_node, target=destination_node, weight=weight
            )
            return path, cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float('inf')

    def find_optimal_route_for_pois(self, origin_node: int, destination_node: int, poi_nodes: list):
        """Find the optimal visiting order for a list of waypoints.

        Solves TSP by brute force over all permutations of the given POI nodes.

        Returns (full_path, total_cost, ordered_stops) or (None, inf, None) if no
        valid route can be found.
        """
        if not poi_nodes:
            path, cost = self.find_shortest_path(origin_node, destination_node)
            return path, cost, []

        all_nodes = [origin_node] + poi_nodes + [destination_node]
        distance_matrix = self._calculate_distance_matrix(all_nodes)

        best_stop_order = None
        minimum_cost = float('inf')

        for permutation in permutations(poi_nodes):
            stop_order = [origin_node] + list(permutation) + [destination_node]
            total_cost = 0
            route_feasible = True

            for i in range(len(stop_order) - 1):
                segment_cost = distance_matrix.loc[stop_order[i], stop_order[i + 1]]
                if segment_cost == float('inf'):
                    route_feasible = False
                    break
                total_cost += segment_cost

            if route_feasible and total_cost < minimum_cost:
                minimum_cost = total_cost
                best_stop_order = stop_order

        if best_stop_order is None:
            return None, float('inf'), None

        full_path = []
        for i in range(len(best_stop_order) - 1):
            segment, _ = self.find_shortest_path(best_stop_order[i], best_stop_order[i + 1])
            if i > 0:
                segment = segment[1:]
            full_path.extend(segment)

        return full_path, minimum_cost, best_stop_order

    def _calculate_distance_matrix(self, nodes: list) -> pd.DataFrame:
        """Build a pairwise travel-time cost matrix for the given node list."""
        unique_nodes = list(set(nodes))
        matrix = pd.DataFrame(index=unique_nodes, columns=unique_nodes)
        for origin in unique_nodes:
            for destination in unique_nodes:
                if origin == destination:
                    matrix.loc[origin, destination] = 0
                else:
                    _, cost = self.find_shortest_path(origin, destination)
                    matrix.loc[origin, destination] = cost
        return matrix

    def get_route_coords(self, path: list) -> list:
        """Return a list of (lon, lat) coordinate tuples for a node path."""
        if not path:
            return []
        return [(self.graph.nodes[node]['x'], self.graph.nodes[node]['y']) for node in path]
