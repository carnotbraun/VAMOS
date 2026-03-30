import osmnx as ox
import networkx as nx
from itertools import permutations
import pandas as pd

class RoutingEngine:
    """
    Gerencia o cálculo de rotas usando um grafo OSMnx, com suporte para otimização de
    múltiplas paradas (Problema do Caixeiro Viajante - TSP).
    """
    def __init__(self, graph):
        if graph is None:
            raise ValueError("O grafo fornecido não pode ser nulo.")
        self.graph = graph

    def address_to_node(self, location: str) -> int:
            """ 
            Converte um endereço (texto) OU coordenadas ("lat, lon") para o nó mais próximo.
            """
            # 1. Tenta interpretar como coordenadas (lat, lon)
            try:
                # Verifica se parece um par de coordenadas
                if "," in location and any(c.isdigit() for c in location):
                    parts = [float(p.strip()) for p in location.split(",")]
                    if len(parts) == 2:
                        lat, lon = parts
                        # ox.nearest_nodes espera (X=lon, Y=lat)
                        return ox.nearest_nodes(self.graph, lon, lat)
            except ValueError:
                pass # Não são coordenadas, continua para geocodificação textual

            # 2. Tenta Geocodificação via API (Texto)
            try:
                print(f"Buscando coordenadas para '{location}' via API...")
                lat, lon = ox.geocode(location)
                return ox.nearest_nodes(self.graph, lon, lat)
            except Exception as e:
                print(f"Erro no Geocoding para '{location}': {e}")
                return None

    def get_node_coords(self, node: int) -> tuple:
        """ Retorna as coordenadas (lat, lon) de um nó. """
        return (self.graph.nodes[node]['y'], self.graph.nodes[node]['x'])

    def find_shortest_path(self, origin_node: int, dest_node: int, weight='travel_time'):
        """ Encontra a rota mais curta entre dois nós. """
        try:
            path = nx.shortest_path(self.graph, source=origin_node, target=dest_node, weight=weight)
            cost = nx.shortest_path_length(self.graph, source=origin_node, target=dest_node, weight=weight)
            return path, cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float('inf')

    def find_optimal_route_for_pois(self, origin_node: int, dest_node: int, poi_nodes: list):
        """
        Encontra a ordem ótima para visitar uma lista de POIs.
        Soluciona o Problema do Caixeiro Viajante (TSP) para os POIs.
        """
        if not poi_nodes:
            path, cost = self.find_shortest_path(origin_node, dest_node)
            return path, cost, []

        all_nodes = [origin_node] + poi_nodes + [dest_node]
        dist_matrix = self._calculate_distance_matrix(all_nodes)

        best_path_order = None
        min_cost = float('inf')
        
        for perm in permutations(poi_nodes):
            current_path_order = [origin_node] + list(perm) + [dest_node]
            current_cost = 0
            
            path_is_possible = True
            for i in range(len(current_path_order) - 1):
                segment_cost = dist_matrix.loc[current_path_order[i], current_path_order[i+1]]
                if segment_cost == float('inf'):
                    path_is_possible = False
                    break
                current_cost += segment_cost
            
            if path_is_possible and current_cost < min_cost:
                min_cost = current_cost
                best_path_order = current_path_order
        
        # Se nenhum caminho completo foi encontrado, retorna None para evitar o crash.
        if best_path_order is None:
            return None, float('inf'), None

        full_osm_path = []
        for i in range(len(best_path_order) - 1):
            segment, _ = self.find_shortest_path(best_path_order[i], best_path_order[i+1])
            if i > 0:
                segment = segment[1:]
            full_osm_path.extend(segment)
            
        return full_osm_path, min_cost, best_path_order

    def _calculate_distance_matrix(self, nodes: list):
        """ Cria uma matriz de distâncias (custos de tempo de viagem) entre uma lista de nós. """
        unique_nodes = list(set(nodes))
        matrix = pd.DataFrame(index=unique_nodes, columns=unique_nodes)
        for u in unique_nodes:
            for v in unique_nodes:
                if u == v:
                    matrix.loc[u, v] = 0
                else:
                    _, cost = self.find_shortest_path(u, v)
                    matrix.loc[u, v] = cost
        return matrix

    def get_route_coords(self, path: list[int]) -> list[tuple[float, float]]:
        """ Extrai as coordenadas lon/lat de um caminho. """
        if not path: return []
        return [(self.graph.nodes[node]['x'], self.graph.nodes[node]['y']) for node in path]