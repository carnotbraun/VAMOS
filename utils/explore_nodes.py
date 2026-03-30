import geopandas as gpd
import osmnx as ox
import pandas as pd
import random
import os

# Nomes dos arquivos de cache, para manter a consistência com graph_utils.py
GRAPH_FILENAME = "PAVe/são paulo_drive_graph.graphml"
POIS_FILENAME = "PAVe/são paulo_pois.gpkg"

def find_poi_type(row):
    """Função auxiliar para encontrar a categoria de um POI."""
    if pd.notna(row.get('amenity')):
        return row['amenity']
    if pd.notna(row.get('shop')):
        return row['shop']
    if pd.notna(row.get('leisure')):
        return row['leisure']
    return "desconhecido"

def main():
    print("--- Explorador de Cenário ---")

    # --- 1. Explorando Pontos de Interesse (POIs) ---
    pois_path = os.path.join(POIS_FILENAME)
    print(f"\n--- 1. Verificando Pontos de Interesse do arquivo '{pois_path}' ---")
    if not os.path.exists(pois_path):
        print(f"Arquivo de POIs '{POIS_FILENAME}' não encontrado.")
        print("Execute o 'app.py' com endereços de Campinas para baixar os dados corretos.")
    else:
        pois_gdf = gpd.read_file(pois_path)
        named_pois = pois_gdf.dropna(subset=['name']).copy()
        
        if not named_pois.empty:
            print(f"Encontrados {len(named_pois)} POIs com nome. Aqui estão alguns exemplos:")
            named_pois['type'] = named_pois.apply(find_poi_type, axis=1)
            for _, poi in named_pois.sample(min(15, len(named_pois))).iterrows():
                print(f"  - {poi['name']} (Tipo: {poi['type']})")
        else:
            print("Nenhum POI com nome encontrado no arquivo de cache.")

    # --- 2. Explorando Ruas e Avenidas ---
    graph_path = os.path.join(GRAPH_FILENAME)
    print(f"\n--- 2. Verificando Ruas do arquivo '{graph_path}' ---")
    if not os.path.exists(graph_path):
        print(f"Arquivo de Grafo '{GRAPH_FILENAME}' não encontrado.")
        print("Execute o 'app.py' com endereços de Campinas para baixar os dados corretos.")
    else:
        print("Carregando grafo (isso pode levar um momento)...")
        graph = ox.load_graphml(graph_path)
        edges_gdf = ox.graph_to_gdfs(graph, nodes=False)
        
        # Filtra as ruas que têm nome.
        named_streets_series = edges_gdf.dropna(subset=['name'])['name']
        
        # Cria uma lista vazia para "achatar" os nomes.
        all_street_names = []
        # Itera sobre a série de nomes.
        for item in named_streets_series:
            if isinstance(item, str):
                # Se for uma string, apenas adiciona à lista.
                all_street_names.append(item)
            elif isinstance(item, list):
                # Se for uma lista, adiciona cada item da lista.
                all_street_names.extend(item)
        
        # Agora, com a lista achatada, podemos pegar os nomes únicos.
        street_names = list(set(all_street_names))
        
        if len(street_names) > 0:
            print(f"Encontrados {len(street_names)} nomes de ruas únicos. Aqui estão alguns exemplos:")
            sample_streets = random.sample(street_names, min(15, len(street_names)))
            for street in sample_streets:
                print(f"  - {street}")
        else:
            print("Nenhum nome de rua encontrado no arquivo de cache.")

if __name__ == "__main__":
    main()
