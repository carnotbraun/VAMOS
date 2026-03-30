import os
import random

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point

# --- CONFIGURAÇÕES ---
GRAPH_PATH = "/home/carnot.filho/move-llm/PAVe/graph_data/SP/são paulo_drive_graph.graphml"
POIS_PATH = "/home/carnot.filho/move-llm/PAVe/graph_data/SP/são paulo_pois.gpkg"
OUTPUT_FILE = "/home/carnot.filho/move-llm/PAVe/scenarios_generated.json"

# Categorias que queremos testar
TARGET_CATEGORIES = {
    "fuel": {"osm_tag": "fuel", "buffer_m": 500, "name": "Urgência - Combustível"},
    "hospital": {"osm_tag": "hospital", "buffer_m": 2000, "name": "Urgência - Hospital"},
    "supermarket": {"osm_tag": "supermarket", "buffer_m": 800, "name": "Conveniência - Mercado"},
    "park": {"osm_tag": "park", "buffer_m": 1000, "name": "Lazer - Parque"}
}

SCENARIOS_PER_CATEGORY = 6
MIN_TRIP_DISTANCE_M = 100  # Mínimo 100m de viagem
MAX_TRIP_DISTANCE_M = 2000 # Máximo 2km (cenário urbano)

def load_data():
    """Carrega grafo e POIs, mantendo duas versões do grafo:

    - G_wgs (EPSG:4326): para exportar coordenadas lat/lon e evitar geocoding no benchmark.
    - G_proj (EPSG:31983): para cálculos métricos (buffers e distâncias).
    """
    print("--- Carregando Grafo Viário (pode demorar um pouco)... ---")
    G_raw = ox.load_graphml(GRAPH_PATH)

    print("--- Carregando POIs... ---")
    gdf_pois = gpd.read_file(POIS_PATH)

    # 1) Garante uma versão WGS84 (lat/lon) para serialização de cenários
    if str(G_raw.graph.get("crs", "")).lower() != "epsg:4326":
        G_wgs = ox.project_graph(G_raw, to_crs="epsg:4326")
    else:
        G_wgs = G_raw

    # 2) Versão projetada para métricas (SP ~ UTM 23S)
    if str(G_wgs.graph.get("crs", "")).lower() == "epsg:4326":
        G_proj = ox.project_graph(G_wgs, to_crs="epsg:31983")
    else:
        G_proj = G_wgs

    if gdf_pois.crs is None:
        raise ValueError("POIs sem CRS definido. Verifique o arquivo GPKG.")
    if gdf_pois.crs.to_string() != "EPSG:31983":
        gdf_pois = gdf_pois.to_crs("epsg:31983")

    return G_proj, G_wgs, gdf_pois

def get_node_name(G, node_id):
    """Tenta recuperar o nome da rua de um nó, ou retorna coordenadas se falhar."""
    data = G.nodes[node_id]
    try:
        edges = list(G.out_edges(node_id, data=True))
        if edges:
            name = edges[0][2].get("name", "Rua Desconhecida")
            if isinstance(name, list):
                name = name[0]
            return name
    except Exception:
        pass
    return f"Coord ({data.get('y'):.6f}, {data.get('x'):.6f})"


def format_latlon(G_wgs, node_id) -> str:
    """Retorna string "lat, lon" para ser consumida por RoutingEngine.address_to_node."""
    lat = float(G_wgs.nodes[node_id]["y"])
    lon = float(G_wgs.nodes[node_id]["x"])
    # 6 casas já é suficiente e evita problemas de parsing
    return f"{lat:.6f}, {lon:.6f}"

def generate_scenarios(G_proj, G_wgs, gdf_pois):
    nodes = list(G_proj.nodes)
    generated_scenarios = []
    
    print(f"\n--- Iniciando Mineração de Cenários ({SCENARIOS_PER_CATEGORY} por categoria) ---")

    for cat_key, config in TARGET_CATEGORIES.items():
        print(f"   > Buscando cenários para: {config['name']}...")
        count = 0
        attempts = 0
        
        # Filtra POIs da categoria atual
        if cat_key == 'fuel':
            pois_subset = gdf_pois[gdf_pois['amenity'] == 'fuel']
        elif cat_key == 'hospital':
            pois_subset = gdf_pois[gdf_pois['amenity'] == 'hospital']
        elif cat_key == 'supermarket':
            pois_subset = gdf_pois[gdf_pois['shop'] == 'supermarket']
        elif cat_key == 'park':
            pois_subset = gdf_pois[gdf_pois['leisure'] == 'park']
        
        # Remove POIs sem nome
        pois_subset = pois_subset[pois_subset['name'].notna()]
        
        if pois_subset.empty:
            print(f"     [AVISO] Nenhum POI encontrado para {cat_key} no arquivo GPKG.")
            continue

        while count < SCENARIOS_PER_CATEGORY and attempts < 2000:
            attempts += 1
            
            # 1. Sorteia Origem e Destino
            u, v = random.sample(nodes, 2)
            
            # 2. Verifica Distância Euclidiana Rápida (para não perder tempo com rota)
            pt_u = Point(G_proj.nodes[u]["x"], G_proj.nodes[u]["y"])
            pt_v = Point(G_proj.nodes[v]["x"], G_proj.nodes[v]["y"])
            dist = pt_u.distance(pt_v)
            
            if dist < MIN_TRIP_DISTANCE_M or dist > MAX_TRIP_DISTANCE_M:
                continue
                
            # 3. Calcula Rota (Shortest Path)
            try:
                route = nx.shortest_path(G_proj, u, v, weight="length")
            except nx.NetworkXNoPath:
                continue
                
            # 4. Cria Geometria da Rota (LineString) e Buffer
            route_coords = [(G_proj.nodes[n]["x"], G_proj.nodes[n]["y"]) for n in route]
            route_line = LineString(route_coords)
            route_buffer = route_line.buffer(config['buffer_m']) # Buffer em metros
            
            # 5. Busca Espacial: Existe POI relevante DENTRO do buffer?
            # Usando interseção espacial vetorizada
            possible_pois = pois_subset[pois_subset.geometry.intersects(route_buffer)]
            
            if not possible_pois.empty:
                # Seleciona o POI mais próximo da rota (distância geométrica até a LineString)
                possible_pois = possible_pois.copy()
                possible_pois["dist_to_route"] = possible_pois.geometry.apply(lambda geom: geom.distance(route_line))
                poi = possible_pois.sort_values("dist_to_route").iloc[0]

                # Exporta origem/destino como coordenadas lat/lon para evitar geocoding no benchmark
                origin_str = format_latlon(G_wgs, u)
                dest_str = format_latlon(G_wgs, v)

                # (Opcional) nomes humanos para depuração
                origin_name = get_node_name(G_proj, u)
                dest_name = get_node_name(G_proj, v)

                scenario = {
                    "scenario_name": f"{config['name']} - Exemplo {count+1}",
                    # NOTE: agora origin/destination são strings "lat, lon" (RoutingEngine aceita direto)
                    "origin": origin_str,
                    "destination": dest_str,
                    "origin_node": int(u),
                    "destination_node": int(v),
                    "origin_street_hint": origin_name,
                    "destination_street_hint": dest_name,
                    "target_poi": str(poi["name"]),
                    "tasks": [f"I need to go to a {cat_key}"],
                    "category": cat_key,
                    "expected_choice": 2,
                    "recalculation_expected": True,
                    # NOVO: ajuda o benchmark a medir completeness sem depender da escolha (rota 1 vs 2)
                    "expected_action_type": "ADD_WAYPOINT",
                }
                
                generated_scenarios.append(scenario)
                count += 1
                print(
                    f"     [SUCESSO] Rota encontrada! Passa perto de '{poi['name']}' "
                    f"({origin_name} -> {dest_name}) | dist_to_route={float(poi['dist_to_route']):.1f}m"
                )

    return generated_scenarios

def save_to_python_format(scenarios):
    """Imprime no formato que você pode copiar e colar no benchmark_real_data.py"""
    print("\n\n" + "="*50)
    print("COPIE E COLE ISSO NO SEU ARQUIVO 'benchmark_real_data.py'")
    print("="*50)
    print("SCENARIOS = [")
    for sc in scenarios:
        # Limpeza para string python válida
        print(f"    {{")
        print(f"        'scenario_name': \"{sc['scenario_name']}\",")
        print(f"        'origin': \"{sc['origin']}\",")
        print(f"        'destination': \"{sc['destination']}\",")
        print(f"        'tasks': {sc['tasks']},")
        print(f"        'expected_choice': {sc['expected_choice']},")
        print(f"        'recalculation_expected': {sc['recalculation_expected']},")
        if 'expected_action_type' in sc:
            print(f"        'expected_action_type': '{sc['expected_action_type']}',")
        print(f"        'note': \"POI Alvo detectado na mineração: {sc['target_poi']}\"")
        print(f"    }},")
    print("]")

if __name__ == "__main__":
    G_proj, G_wgs, gdf = load_data()
    scenarios = generate_scenarios(G_proj, G_wgs, gdf)
    save_to_python_format(scenarios)