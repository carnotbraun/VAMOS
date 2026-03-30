"""
Script para gerar mapas de rotas com POIs para São Paulo, Belém e Salvador.
Usa dados pré-carregados da pasta graph_data/
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Circle
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os

# Caminho base para os dados
BASE_PATH = "VAMOS/graph_data"

# Configurações das cidades
CITIES = {
    'SP': {
        'name': 'São Paulo',
        'folder': 'SP',
        'graph_file': 'são paulo_drive_graph.graphml',
        'pois_file': 'são paulo_pois.gpkg',
        'poi_type': 'fuel',  # Posto de gasolina
        'color': '#1f77b4'
    },
    'BL': {
        'name': 'Belém',
        'folder': 'BL',
        'graph_file': 'belém_drive_graph.graphml',
        'pois_file': 'belém_pois.gpkg',
        'poi_type': 'hospital',  # Hospital
        'color': '#ff7f0e'
    },
    'SA': {
        'name': 'Salvador',
        'folder': 'SA',
        'graph_file': 'salvador_drive_graph.graphml',
        'pois_file': 'salvador_pois.gpkg',
        'poi_type': 'restaurant',  # Restaurante
        'color': '#2ca02c'
    }
}

# Tamanho fixo para todas as figuras
FIG_WIDTH = 10
FIG_HEIGHT = 10

def load_city_data(city_config):
    """Carrega o grafo e POIs da cidade a partir dos arquivos."""
    print(f"\nCarregando dados de {city_config['name']}...")
    
    # Caminhos dos arquivos
    graph_path = os.path.join(BASE_PATH, city_config['folder'], city_config['graph_file'])
    pois_path = os.path.join(BASE_PATH, city_config['folder'], city_config['pois_file'])
    
    try:
        # Carrega o grafo
        print(f"  Carregando grafo de {graph_path}...")
        G = ox.load_graphml(graph_path)
        print(f"  ✓ Grafo carregado: {len(G.nodes)} nós, {len(G.edges)} arestas")
        
        # Carrega os POIs
        print(f"  Carregando POIs de {pois_path}...")
        pois_gdf = gpd.read_file(pois_path)
        print(f"  ✓ POIs carregados: {len(pois_gdf)} registros")
        
        return G, pois_gdf
    except Exception as e:
        print(f"  ✗ Erro ao carregar dados: {e}")
        return None, None

def find_suitable_origin_destination(G, min_distance=2000, max_distance=8000):
    """Encontra nós de origem e destino com uma distância razoável."""
    print(f"  Buscando nós para rota (distância entre {min_distance}-{max_distance}m)...")
    
    nodes = list(G.nodes())
    
    # Tenta encontrar um par de nós com distância adequada
    for _ in range(100):  # Tenta até 100 vezes
        origin = np.random.choice(nodes)
        destination = np.random.choice(nodes)
        
        if origin == destination:
            continue
        
        try:
            # Calcula a distância euclidiana aproximada
            ox_node = G.nodes[origin]['x']
            oy_node = G.nodes[origin]['y']
            dx_node = G.nodes[destination]['x']
            dy_node = G.nodes[destination]['y']
            
            # Distância aproximada em graus
            dist_deg = np.sqrt((dx_node - ox_node)**2 + (dy_node - oy_node)**2)
            # Converte para metros (aproximadamente 111km por grau)
            dist_m = dist_deg * 111000
            
            if min_distance <= dist_m <= max_distance:
                # Verifica se existe caminho
                if nx.has_path(G, origin, destination):
                    print(f"  ✓ Nós encontrados: origem={origin}, destino={destination}")
                    print(f"    Distância aproximada: {dist_m:.0f}m")
                    return origin, destination
        except:
            continue
    
    # Se não encontrou, retorna os dois primeiros nós conectados
    print(f"  ⚠ Usando nós padrão")
    for u, v in G.edges():
        if nx.has_path(G, u, v):
            return u, v
    
    return None, None

def calculate_routes(G, origin_node, dest_node, pois_gdf, poi_type, max_detour_percent=40):
    """Calcula rotas: direta e com desvio para POI.
    
    Args:
        max_detour_percent: Percentual máximo de desvio aceitável (padrão: 40%)
    """
    routes = {}
    
    # Rota direta
    try:
        direct_route = nx.shortest_path(G, origin_node, dest_node, weight='length')
        direct_dist = nx.shortest_path_length(G, origin_node, dest_node, weight='length')
        routes['direct'] = direct_route
        routes['direct_dist'] = direct_dist
        print(f"  ✓ Rota direta calculada: {len(direct_route)} nós, {direct_dist:.0f}m")
    except Exception as e:
        print(f"  ✗ Erro ao calcular rota direta: {e}")
        return None
    
    # Define o limite de desvio aceitável
    max_acceptable_dist = direct_dist * (1 + max_detour_percent / 100)
    
    # Filtra POIs do tipo desejado
    poi_candidates = []
    if pois_gdf is not None and not pois_gdf.empty:
        print(f"  Buscando POIs do tipo '{poi_type}'...")
        
        # Filtra POIs por tipo (amenity)
        if 'amenity' in pois_gdf.columns:
            filtered_pois = pois_gdf[pois_gdf['amenity'] == poi_type]
            print(f"    Encontrados {len(filtered_pois)} POIs do tipo '{poi_type}'")
        else:
            filtered_pois = pois_gdf.head(50)
            print(f"    Usando os primeiros 50 POIs (coluna 'amenity' não encontrada)")
        
        if len(filtered_pois) == 0:
            print(f"    ⚠ Nenhum POI do tipo '{poi_type}' encontrado. Tentando outros tipos...")
            filtered_pois = pois_gdf.head(50)
        
        # Converte POIs para nós do grafo e calcula distâncias
        print(f"    Avaliando POIs quanto à viabilidade da rota...")
        for idx, poi in filtered_pois.head(50).iterrows():
            try:
                if poi.geometry.geom_type == 'Point':
                    poi_lat = poi.geometry.y
                    poi_lon = poi.geometry.x
                else:
                    centroid = poi.geometry.centroid
                    poi_lat = centroid.y
                    poi_lon = centroid.x
                
                # Encontra o nó mais próximo
                poi_node = ox.distance.nearest_nodes(G, poi_lon, poi_lat)
                
                # Calcula distâncias
                dist_to_poi = nx.shortest_path_length(G, origin_node, poi_node, weight='length')
                dist_from_poi = nx.shortest_path_length(G, poi_node, dest_node, weight='length')
                total_dist = dist_to_poi + dist_from_poi
                extra_dist = total_dist - direct_dist
                detour_percent = (extra_dist / direct_dist) * 100
                
                # Só considera POIs com desvio aceitável
                if total_dist <= max_acceptable_dist and detour_percent >= 5:  # Mínimo de 5% para ser visível
                    poi_candidates.append({
                        'node': poi_node,
                        'total_dist': total_dist,
                        'detour_percent': detour_percent,
                        'extra_dist': extra_dist
                    })
            except Exception as e:
                continue
        
        print(f"    → {len(poi_candidates)} POIs viáveis (desvio < {max_detour_percent}%)")
    
    # Escolhe o melhor POI (menor desvio que ainda seja visível)
    if poi_candidates:
        # Ordena por distância total e pega o melhor
        poi_candidates.sort(key=lambda x: x['total_dist'])
        best_candidate = poi_candidates[0]
        best_poi = best_candidate['node']
        
        try:
            route_to_poi = nx.shortest_path(G, origin_node, best_poi, weight='length')
            route_from_poi = nx.shortest_path(G, best_poi, dest_node, weight='length')
            routes['with_poi'] = route_to_poi + route_from_poi[1:]
            routes['poi_node'] = best_poi
            print(f"  ✓ Rota com POI calculada: {len(routes['with_poi'])} nós")
            print(f"    Distância com POI: {best_candidate['total_dist']:.0f}m")
            print(f"    Desvio: +{best_candidate['extra_dist']:.0f}m ({best_candidate['detour_percent']:.1f}%)")
        except Exception as e:
            print(f"  ⚠ Erro ao calcular rota com POI: {e}")
    else:
        print(f"  ⚠ Nenhum POI viável encontrado (desvio < {max_detour_percent}%)")
    
    return routes

def plot_city_map(city_key, city_config, G, routes, output_file=None):
    """Plota o mapa da cidade com as rotas."""
    print(f"\n  Gerando figura para {city_config['name']}...")
    
    # Cria a figura com tamanho fixo
    fig, ax = ox.plot_graph(
        G, 
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        node_size=0, 
        edge_linewidth=0.8, 
        edge_color='#cccccc',
        bgcolor='white',
        show=False, 
        close=False
    )
    
    # Extrai coordenadas dos nós
    def get_route_coords(route):
        lons = [G.nodes[node]['x'] for node in route]
        lats = [G.nodes[node]['y'] for node in route]
        return lons, lats
    
    # Plota rota com POI primeiro (se disponível) - fica por baixo
    if 'with_poi' in routes:
        lons, lats = get_route_coords(routes['with_poi'])
        ax.plot(lons, lats, color=city_config['color'], linewidth=5, 
                alpha=0.7, label='Rota com POI', zorder=2,
                solid_capstyle='round')
        
        # Marca o POI
        if 'poi_node' in routes:
            poi_node = routes['poi_node']
            poi_lon = G.nodes[poi_node]['x']
            poi_lat = G.nodes[poi_node]['y']
            ax.plot(poi_lon, poi_lat, marker='*', color='red', 
                   markersize=25, zorder=5, markeredgecolor='darkred',
                   markeredgewidth=2, label=f'POI ({city_config["poi_type"]})')
    
    # Plota rota direta por cima (sempre visível)
    if 'direct' in routes:
        lons, lats = get_route_coords(routes['direct'])
        ax.plot(lons, lats, color='#2E86AB', linewidth=3.5, alpha=0.9, 
                linestyle='--', label='Rota Direta', zorder=3,
                dash_capstyle='round')
    
    # Marca origem e destino
    origin_lon = G.nodes[routes['direct'][0]]['x']
    origin_lat = G.nodes[routes['direct'][0]]['y']
    dest_lon = G.nodes[routes['direct'][-1]]['x']
    dest_lat = G.nodes[routes['direct'][-1]]['y']
    
    ax.plot(origin_lon, origin_lat, marker='o', color='green', 
           markersize=12, zorder=4, markeredgecolor='darkgreen',
           markeredgewidth=1.5, label='Origem')
    ax.plot(dest_lon, dest_lat, marker='s', color='red', 
           markersize=12, zorder=4, markeredgecolor='darkred',
           markeredgewidth=1.5, label='Destino')
    
    # Ajusta os limites do plot para centralizar nas rotas
    all_lons = []
    all_lats = []
    for route_key in ['direct', 'with_poi']:
        if route_key in routes:
            route_lons, route_lats = get_route_coords(routes[route_key])
            all_lons.extend(route_lons)
            all_lats.extend(route_lats)
    
    if all_lons and all_lats:
        # Calcula o centro e a extensão
        center_lon = (min(all_lons) + max(all_lons)) / 2
        center_lat = (min(all_lats) + max(all_lats)) / 2
        
        # Calcula a extensão necessária
        lon_range = max(all_lons) - min(all_lons)
        lat_range = max(all_lats) - min(all_lats)
        
        # Usa a maior extensão para manter proporção quadrada
        max_range = max(lon_range, lat_range)
        
        # Adiciona margem de 20%
        margin = max_range * 0.2
        final_range = max_range + margin
        
        # Define limites simétricos ao redor do centro
        ax.set_xlim(center_lon - final_range/2, center_lon + final_range/2)
        ax.set_ylim(center_lat - final_range/2, center_lat + final_range/2)
    
    # Força aspect ratio igual para todos os mapas
    ax.set_aspect('equal', adjustable='box')
    
    # Configurações do plot
    ax.set_title(f'{city_config["name"]}', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Remove os ticks dos eixos para deixar mais limpo
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.tick_params(labelsize=8)
    
    # Garante que a figura seja salva com o tamanho correto
    plt.tight_layout()
    
    # Salva a figura com tamanho fixo
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   pad_inches=0.1)
        print(f"  ✓ Figura salva: {output_file}")
    
    plt.close(fig)
    return fig

def generate_all_maps():
    """Função principal para gerar todos os mapas."""
    print("="*60)
    print("GERADOR DE MAPAS DE ROTAS COM POIs")
    print("Cidades: São Paulo, Belém, Salvador")
    print("="*60)
    
    results = {}
    
    for city_key, city_config in CITIES.items():
        print(f"\n{'='*60}")
        print(f"Processando: {city_config['name']}")
        print(f"{'='*60}")
        
        # 1. Carrega os dados da cidade
        G, pois_gdf = load_city_data(city_config)
        if G is None:
            print(f"✗ Falha ao processar {city_config['name']}")
            continue
        
        # 2. Tenta encontrar uma boa combinação de origem/destino com POI viável
        routes = None
        max_attempts = 20
        
        for attempt in range(max_attempts):
            # Encontra nós de origem e destino adequados
            origin_node, dest_node = find_suitable_origin_destination(G)
            if origin_node is None or dest_node is None:
                continue
            
            # Calcula as rotas
            routes = calculate_routes(G, origin_node, dest_node, pois_gdf, city_config['poi_type'])
            
            # Se encontrou rota com POI, sucesso!
            if routes and 'with_poi' in routes:
                print(f"\n  ✓ Combinação válida encontrada na tentativa {attempt + 1}")
                break
            
            # Se não encontrou POI viável, tenta novamente
            if attempt < max_attempts - 1:
                print(f"    Tentando nova combinação... ({attempt + 2}/{max_attempts})")
        
        if routes is None or 'with_poi' not in routes:
            print(f"✗ Não foi possível encontrar rota com POI viável após {max_attempts} tentativas")
            if routes:
                print(f"  → Gerando apenas rota direta")
            else:
                print(f"✗ Falha ao processar {city_config['name']}")
                continue
        
        # 4. Gera o mapa
        output_file = f"mapa_{city_key.lower()}.png"
        plot_city_map(city_key, city_config, G, routes, output_file)
        
        results[city_key] = {
            'graph': G,
            'routes': routes,
            'pois': pois_gdf,
            'output_file': output_file
        }
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO DA GERAÇÃO")
    print("="*60)
    for city_key, result in results.items():
        city_name = CITIES[city_key]['name']
        print(f"\n{city_name}:")
        print(f"  ✓ Arquivo: {result['output_file']}")
        if 'with_poi' in result['routes']:
            print(f"  ✓ Rota com POI gerada")
        else:
            print(f"  ⚠ Apenas rota direta (POI não disponível)")
    
    print("\n" + "="*60)
    print("Processo concluído!")
    print("="*60)

if __name__ == "__main__":
    generate_all_maps()
