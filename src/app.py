import argparse
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import json

import graph_utils
from routing_engine import RoutingEngine
from llm_agent import LLMAgent
from context_engine import ContextEngine
from shapely.geometry import LineString

# --- CONFIGURAÇÕES ---
# Modelos por método
#HF_MODEL_NAME = "solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ"  # Modelo HuggingFace teste
#HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Modelo HuggingFace maior
#HF_MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct" # Menor modelo 
#HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Modelo medio
#HF_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct" #modelo semelhante de tamanho

HF_MODEL_NAME = "Qwen/Qwen3-4B" # Modelo baseline
#HF_MODEL_NAME = "Qwen/Qwen3-0.6B"
#HF_MODEL_NAME = "Qwen/Qwen3-1.7"  
#HF_MODEL_NAME = "Qwen/Qwen3-8B"  # Modelo maior Qwen-3


OLLAMA_MODEL_NAME = "llama3"  # Modelo Ollama local
OPENAI_MODEL_NAME = "gpt-3.5-turbo"  # Modelo OpenAI API
#OPENAI_MODEL_NAME = "gpt-4-turbo-preview"  # GPT-4 (mais caro)
#OPENAI_MODEL_NAME = "gpt-4o"  # GPT-4 Optimized

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def plot_final_route(router: RoutingEngine, path: list, pois_info: list, output_filename="rota_final.png"):
    """
    Plota a rota final usando as funcionalidades nativas e mais robustas do OSMnx.
    """
    print(f"\nGerando mapa da rota em '{output_filename}'...")
    if not path:
        print("Caminho vazio, não é possível gerar o mapa.")
        return

    # Extrai as coordenadas dos POIs para plotagem
    poi_lons = [poi['lon'] for poi in pois_info]
    poi_lats = [poi['lat'] for poi in pois_info]

    # Calcula o bounding box do caminho para zoom
    path_lons = [router.graph.nodes[n]['x'] for n in path]
    path_lats = [router.graph.nodes[n]['y'] for n in path]
    
    # Adiciona margem de 10% ao redor do caminho
    margin = 0.5
    lon_range = max(path_lons) - min(path_lons)
    lat_range = max(path_lats) - min(path_lats)
    bbox = (
        min(path_lats) - lat_range * margin,
        max(path_lats) + lat_range * margin,
        min(path_lons) - lon_range * margin,
        max(path_lons) + lon_range * margin
    )

    # Plota o grafo e a rota com cores visíveis
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
    
    # Adiciona os POIs visitados ao mapa como pontos azuis
    if poi_lons and poi_lats:
        ax.scatter(poi_lons, poi_lats, c='blue', s=150, zorder=5, label='Paradas (POIs)', edgecolors='black', linewidths=1.5)
    
    ax.set_title("Rota Final Otimizada com Paradas", fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print("Mapa salvo com sucesso.")
    
def geometry_to_latlon(geom):
    """Retorna (lat, lon) para qualquer geometria shapely."""
    if geom is None:
        return None, None
    if geom.geom_type == "Point":
        p = geom
    else:
        p = geom.representative_point()
    return p.y, p.x

def main():
    parser = argparse.ArgumentParser(
        description="Agente VAMOS SBRC 2026",
        allow_abbrev=False  # Evita ambiguidade com números negativos
    )
    parser.add_argument("--method", default="ollama", choices=['ollama', 'hf', 'openai'], help="Método do LLM a ser utilizado.")
    parser.add_argument("--origem", required=True, type=str, help="Endereço de partida (ex: 'Rua Reitor Lafayete, Campinas' ou '-1.370769,-48.442457').")
    parser.add_argument("--destino", required=True, type=str, help="Endereço de destino (ex: 'Shopping Iguatemi, Campinas' ou '-1.360059,-48.432432').")
    parser.add_argument("--tarefas", nargs='+', default=[], help="Lista de tarefas (ex: 'passar na farmácia' 'comprar pão').")
    parser.add_argument("--force_download", action="store_true", help="Força o download do grafo e POIs, ignorando o cache.")
    args = parser.parse_args()

    print("--- Iniciando Sistema de Roteamento... ---")

    graph = graph_utils.get_graph(args.force_download)
    pois_gdf = graph_utils.get_pois(graph, args.force_download)
    # --- PROJEÇÃO PARA CÁLCULO DE DISTÂNCIAS (em metros) ---
    G_proj = ox.project_graph(graph)
    pois_proj = pois_gdf.to_crs(G_proj.graph["crs"]) if not pois_gdf.empty else pois_gdf

    router = RoutingEngine(graph)
    
    # Seleciona o modelo baseado no método
    if args.method == 'hf':
        model_name = HF_MODEL_NAME
    elif args.method == 'ollama':
        model_name = OLLAMA_MODEL_NAME
    elif args.method == 'openai':
        model_name = OPENAI_MODEL_NAME
    else:
        model_name = OLLAMA_MODEL_NAME  # fallback
    
    llm = LLMAgent(model_name=model_name, method=args.method) 
    context_provider = ContextEngine()

    origem_node = router.address_to_node(args.origem)
    destino_node = router.address_to_node(args.destino) 
    if not all([origem_node, destino_node]):
        print("Erro: Não foi possível encontrar os nós de origem/destino. Verifique os endereços.")
        return

    task_analysis = llm.classify_tasks(args.tarefas)
    tasks = sorted(task_analysis.get("tasks", []), key=lambda x: x.get('importance', 0), reverse=True)
    
    candidate_routes = []
    
    direct_path, direct_cost = router.find_shortest_path(origem_node, destino_node)
    route_direct_nodes = direct_path  # alias para o snippet de POI (rota direta)

    if direct_path:
        candidate_routes.append({
            "description": "Rota Direta", "path": direct_path, "cost_seconds": direct_cost,
            "tasks_completed": [], "pois_info": []
        })

    if tasks:
        print("\nTarefas classificadas por importância:")
        for t in tasks: print(f"- {t['task']} (Importance: {t.get('importance', 'N/A')}) -> POI: {t['poi_tags']}")
        
        poi_nodes_to_visit, task_poi_map = [], {}
        for task in tasks:
            tags = task.get("poi_tags", {})
            if not tags: continue
            key, value = list(tags.items())[0]
            relevant_pois = pois_proj[pois_proj[key] == value]

            if not relevant_pois.empty:
                # 1) Construir LineString da rota direta (em coordenadas projetadas)
                route_coords = [
                    (G_proj.nodes[n]["x"], G_proj.nodes[n]["y"])
                    for n in route_direct_nodes
                ]
                route_line = LineString(route_coords)

                # 2) Calcular distância de cada POI à rota
                relevant_pois = relevant_pois.copy()
                relevant_pois["dist_to_route"] = relevant_pois.geometry.apply(
                    lambda geom: geom.distance(route_line)
                )

                # 3) Selecionar o POI mais próximo da rota
                best_poi = relevant_pois.sort_values("dist_to_route").iloc[0]
                poi_node = best_poi["nearest_node"]
                if poi_node not in poi_nodes_to_visit:
                    poi_nodes_to_visit.append(poi_node)
                    task_poi_map[poi_node] = task # salvamos o dict inteiro da tarefa
            else: print(f"Aviso: Nenhum POI encontrado para a tarefa '{task['task']}'")
        
        if poi_nodes_to_visit:
            print(f"\nOtimizando a ordem de visita para {len(poi_nodes_to_visit)} POI(s)...")
            multi_stop_path, multi_stop_cost, ordered_stops = router.find_optimal_route_for_pois(origem_node, destino_node, poi_nodes_to_visit)
        
            if multi_stop_path:
                stops_info = [pois_gdf[pois_gdf['nearest_node'] == node].iloc[0] for node in ordered_stops if node not in [origem_node, destino_node]]
                candidate_routes.append({
                    "description": "Rota com Paradas (Multi-tarefa)", "path": multi_stop_path, "cost_seconds": multi_stop_cost,
                    "tasks_completed": [task_poi_map[node] for node in ordered_stops if node in task_poi_map],
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
                print("\nAVISO: Não foi possível calcular uma rota que conecte todos os pontos. Isso pode ser devido a endereços ambíguos ou POIs muito distantes. Apenas a rota direta será considerada.")

    if len(candidate_routes) <= 1:
        final_route = candidate_routes[0] if candidate_routes else None
        justification = "Rota direta selecionada pois não havia tarefas ou a rota com paradas não pôde ser calculada."
    else:
        print("\nRotas candidatas apresentadas ao LLM:")
        for i, r in enumerate(candidate_routes): print(f"  Rota {i+1}: {r['description']}, Tempo: {r['cost_seconds']/60:.2f} min")
        user_context = context_provider.get_user_context()
        origem_coords = router.get_node_coords(origem_node)
        destino_coords = router.get_node_coords(destino_node)
        scenario_context = context_provider.get_scenario_context(origem_coords, destino_coords)
        #print(f"\nContexto em tempo real: Clima: {scenario_context.get('weather')}, Tráfego: {scenario_context.get('traffic_conditions')}")
        llm_evaluation = llm.evaluate_routes(user_context, scenario_context, candidate_routes)

        chosen_id = llm_evaluation.get("chosen_route_id", 1)
        justification = llm_evaluation.get("justification", "Nenhuma justificativa fornecida.")
        final_route = candidate_routes[chosen_id - 1]

    if final_route:
        print("\n--- Decisão Final do Agente ---")
        print(f"Justificativa: {justification}")
        print("\nDetalhes da Rota Recomendada:")
        print(f"  - Descrição: {final_route['description']}")
        print(f"  - Tempo estimado de viagem: {final_route['cost_seconds'] / 60:.2f} minutos")
        if final_route['tasks_completed']:
            task_texts = [task['task'] for task in final_route['tasks_completed']]
            print("  - Tarefas a serem completadas:", ", ".join(task_texts))
            
        plot_final_route(router, final_route['path'], final_route['pois_info'])
    else:
        print("\nNão foi possível determinar uma rota final.")
    
    # Exibe relatório de timing do LLM
    llm.print_timing_report()


if __name__ == "__main__":
    main()