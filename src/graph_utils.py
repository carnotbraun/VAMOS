import os
import osmnx as ox
import geopandas as gpd
import warnings

# --- CONFIGURAÇÕES ---
PATH = '../graph_data/SP/'
os.makedirs(PATH, exist_ok=True)
LOCATION = "São Paulo, Brazil" # Você pode mudar isso dinamicamente ou via config
GRAPH_FILENAME = f"{PATH}{LOCATION.split(',')[0].lower()}_drive_graph.graphml"
POI_FILENAME = f"{PATH}{LOCATION.split(',')[0].lower()}_pois.gpkg"

# Tags OSM para extrair e cachear.
POI_TAGS = {
    "amenity": ["hospital", "pharmacy", "fuel", "restaurant", "cafe", "bar"],
    "shop": ["supermarket", "bakery", "mall"],
    "leisure": ["park", "stadium"]
}

def get_graph(force_download=False):
    """
    Carrega o grafo da cidade a partir de um arquivo de cache local.
    Se o arquivo não existir ou force_download for True, baixa do OSM e salva.
    """
    if os.path.exists(GRAPH_FILENAME) and not force_download:
        print(f"Carregando grafo do cache: '{GRAPH_FILENAME}'")
        return ox.load_graphml(GRAPH_FILENAME)
    
    print(f"Baixando o grafo de ruas para '{LOCATION}' do OpenStreetMap...")
    graph = ox.graph_from_place(LOCATION, network_type='drive')
    
    print("Adicionando informações de velocidade e tempo de viagem...")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    
    print(f"Salvando o grafo em cache: '{GRAPH_FILENAME}'")
    ox.save_graphml(graph, GRAPH_FILENAME)
    
    return graph

def get_pois(graph, force_download=False):
    """
    Carrega os POIs de um cache ou baixa do OSM.
    Retorna um GeoDataFrame limpo e salvo em cache.
    """
    if os.path.exists(POI_FILENAME) and not force_download:
        print(f"Carregando POIs do cache: '{POI_FILENAME}'")
        return gpd.read_file(POI_FILENAME)

    print(f"Baixando POIs para '{LOCATION}' do OpenStreetMap...")
    # Baixa todas as tags solicitadas
    pois_gdf = ox.features_from_place(LOCATION, tags=POI_TAGS)
    
    if pois_gdf.empty:
        print("Aviso: Nenhum POI encontrado para as tags especificadas.")
        return pois_gdf

    print("Processando e limpando dados dos POIs...")
    
    # 1. Correção do Aviso de Geometria (Centroid)
    # Projeta temporariamente para metros (EPSG:3857) para calcular o centroide preciso,
    # depois volta para o CRS original (lat/lon) para alinhar com o grafo.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois_centroids = pois_gdf.to_crs(epsg=3857).geometry.centroid.to_crs(pois_gdf.crs)
    
    # 2. Associa ao nó mais próximo
    print("Associando POIs aos nós mais próximos do grafo...")
    pois_gdf['nearest_node'] = ox.nearest_nodes(graph, pois_centroids.x, pois_centroids.y)
    
    # 3. FILTRAGEM DE COLUNAS (A Correção do Erro)
    # Mantém apenas colunas essenciais e as tags que pedimos. 
    # Descarta colunas "estranhas" como 'fuel:Gasoline' que quebram o GeoPackage.
    cols_to_keep = ['name', 'geometry', 'nearest_node']
    
    # Adiciona as chaves das tags (amenity, shop, leisure) se existirem no download
    for tag_key in POI_TAGS.keys():
        if tag_key in pois_gdf.columns:
            cols_to_keep.append(tag_key)
            
    # Filtra o DataFrame
    pois_gdf = pois_gdf[cols_to_keep]
    
    print(f"Salvando POIs em cache: '{POI_FILENAME}'")
    try:
        pois_gdf.to_file(POI_FILENAME, driver='GPKG')
    except Exception as e:
        print(f"Aviso: Não foi possível salvar o cache de POIs ({e}), mas a execução continuará.")
    
    return pois_gdf