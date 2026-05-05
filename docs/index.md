# VAMOS — Documentação Técnica

Este documento descreve a estrutura interna do artefato VAMOS: seus módulos, classes, APIs públicas e opções de configuração.

---

## Estrutura de Diretórios

```
VAMOS/
├── config.json             # Arquivo de configuração central (modelos, caminhos, benchmark)
├── requirements.txt        # Dependências Python
├── README.md               # Instalação, teste mínimo e guia de reprodução
├── VAMOS.png               # Diagrama de arquitetura
│
├── src/                    # Módulos centrais da aplicação
│   ├── app.py              # Ponto de entrada — CLI e orquestração do pipeline
│   ├── graph_utils.py      # Download e cache do grafo viário e dos POIs
│   ├── routing_engine.py   # Cálculo de rotas e otimização multi-parada (TSP)
│   ├── llm_agent.py        # Interface com o modelo de linguagem (HuggingFace / Ollama / OpenAI)
│   └── context_engine.py   # Provedor de contexto do usuário e do cenário
│
├── utils/                  # Utilitários de experimento
│   ├── bench.py            # Executor do benchmark (reproduz as Tabelas 2 e 3 do artigo)
│   ├── scenario_generator.py  # Ferramenta de mineração espacial que gerou a lista SCENARIOS
│   └── explore_nodes.py    # Explorador interativo de nós do grafo (depuração)
│
├── mapas/                  # Utilitários de visualização
│   └── generate_city_maps.py
│
└── docs/                   # Esta documentação
    └── index.md
```

---

## Configuração (`config.json`)

Todos os parâmetros ajustáveis estão em `config.json` na raiz do projeto. Os módulos carregam este arquivo em tempo de importação — nenhum arquivo de código-fonte precisa ser editado para alterar modelos, localização ou parâmetros do benchmark.

| Chave | Tipo | Padrão | Descrição |
|---|---|---|---|
| `llm.default_method` | string | `"hf"` | Backend padrão quando `--method` é omitido (`hf`, `ollama`, `openai`) |
| `llm.hf_model` | string | `"Qwen/Qwen3-4B"` | Identificador do modelo no HuggingFace Hub |
| `llm.ollama_model` | string | `"qwen:4b"` | Tag do modelo no Ollama (deve corresponder a `ollama list`) |
| `llm.openai_model` | string | `"gpt-3.5-turbo"` | Nome do modelo na API OpenAI |
| `routing.location` | string | `"São Paulo, Brazil"` | Nome da cidade passado ao OSMnx para download do grafo e POIs |
| `routing.graph_cache_dir` | string | `"graph_data"` | Diretório raiz para os arquivos de cache `.graphml` e `.gpkg` |
| `output.route_image` | string | `"rota_final.png"` | Caminho padrão de saída para o mapa da rota gerado |
| `benchmark.llm_method` | string | `"hf"` | Método padrão utilizado pelo `bench.py` |
| `benchmark.runs_per_scenario` | int | `3` | Repetições por cenário para confiabilidade estatística |
| `benchmark.timeout_seconds` | int | `900` | Timeout por execução de subprocesso (segundos); aumente em hardware mais lento |
| `benchmark.log_directory` | string | `"benchmark_logs"` | Diretório para os arquivos de log individuais por cenário |

**Para mudar a cidade**, atualize `routing.location` em `config.json`. O cache do grafo e dos POIs será armazenado em `graph_data/<nome-da-cidade>/`. Delete o diretório de cache ou use `--force-download` para forçar um novo download.

---

## Referência dos Módulos

### `src/app.py` — Ponto de Entrada da Aplicação

Orquestra o pipeline completo do VAMOS: carrega os dados do grafo, inicializa o modelo de linguagem, resolve os endereços, classifica as tarefas, busca os POIs, gera as rotas candidatas e chama o modelo para selecionar a melhor opção.

**Argumentos de linha de comando**

| Argumento | Aliases | Obrigatório | Descrição |
|---|---|---|---|
| `--origin` | `--origem` | Sim | Endereço de partida ou string `"lat,lon"` |
| `--destination` | `--destino` | Sim | Endereço de destino ou string `"lat,lon"` |
| `--tasks` | `--tarefas` | Não | Lista de tarefas em linguagem natural |
| `--method` | — | Não | Backend de inferência (`hf`, `ollama`, `openai`); padrão lido de `config.json` |
| `--output` | — | Não | Caminho de saída do mapa da rota; padrão lido de `config.json` |
| `--force-download` | `--force_download` | Não | Ignora o cache local e força novo download do grafo e dos POIs |

**Funções principais**

`plot_final_route(router, path, pois_info, output_path)` — Renderiza a rota sobre um mapa OSMnx e salva como arquivo PNG.

`geometry_to_latlon(geom)` — Converte qualquer geometria Shapely em tupla `(lat, lon)`, usando o ponto representativo para polígonos.

`main()` — Ponto de entrada da CLI; interpreta os argumentos, executa o pipeline completo e imprime o relatório de temporização.

---

### `src/graph_utils.py` — Gerenciamento do Grafo e dos POIs

Realiza o download, cache e carregamento do grafo viário OSMnx e dos pontos de interesse (POIs) do OpenStreetMap para a cidade configurada. Os caminhos de cache são calculados como caminhos absolutos a partir da localização do próprio módulo, garantindo resolução correta independentemente do diretório de trabalho.

**Variáveis de módulo** (derivadas de `config.json`)

| Variável | Descrição |
|---|---|
| `LOCATION` | Nome da cidade passado às consultas OSMnx |
| `GRAPH_CACHE_DIR` | Caminho absoluto do diretório de cache |
| `GRAPH_FILENAME` | Caminho absoluto do arquivo `.graphml` de cache |
| `POI_FILENAME` | Caminho absoluto do arquivo `.gpkg` de cache dos POIs |
| `POI_TAGS` | Dicionário de categorias de tags OSM a serem baixadas |

**Funções**

`get_graph(force_download=False) → networkx.MultiDiGraph`  
Carrega o grafo viário do cache ou faz o download do OpenStreetMap. Adiciona velocidades e tempos de viagem às arestas quando faz o download.

`get_pois(graph, force_download=False) → geopandas.GeoDataFrame`  
Carrega os pontos de interesse do cache ou faz o download do OpenStreetMap. Associa cada POI ao nó mais próximo do grafo via `ox.nearest_nodes`.

---

### `src/routing_engine.py` — Cálculo de Rotas

Encapsula os algoritmos de caminho mínimo do NetworkX e implementa um solucionador de TSP por força bruta para rotas com múltiplas paradas.

**Classe `RoutingEngine`**

`__init__(graph)` — Aceita um MultiDiGraph do NetworkX (retornado por `graph_utils.get_graph()`).

`address_to_node(location: str) → int`  
Converte um endereço textual ou string `"lat, lon"` no nó mais próximo do grafo. Tenta a análise numérica primeiro; recorre à API de geocodificação Nominatim se necessário.

`get_node_coords(node: int) → tuple`  
Retorna `(lat, lon)` para um nó do grafo.

`find_shortest_path(origin_node, destination_node, weight='travel_time') → (list, float)`  
Retorna `(lista_de_nós, custo)`. Retorna `(None, inf)` quando não existe caminho.

`find_optimal_route_for_pois(origin_node, destination_node, poi_nodes: list) → (list, float, list)`  
Resolve o Problema do Caixeiro Viajante sobre `poi_nodes` por permutação exaustiva. Retorna `(caminho_completo, custo_total, ordem_das_paradas)`.

`get_route_coords(path: list) → list`  
Converte um caminho de IDs de nós em uma lista de tuplas `(lon, lat)` para visualização.

---

### `src/llm_agent.py` — Interface com o Modelo de Linguagem

Fornece uma interface unificada para três backends de inferência. Gerencia a geração de saída estruturada via biblioteca `outlines` (HF) e o parsing de JSON para todos os backends.

**Esquemas Pydantic** (definem a saída estruturada)

| Classe | Campos | Propósito |
|---|---|---|
| `Task` | `task`, `importance`, `poi_tags` | Uma tarefa do usuário classificada |
| `ModelOutput` | `tasks: List[Task]` | Resposta completa da classificação de tarefas |
| `RequiredAction` | `type` (`ADD_WAYPOINT` ou `NONE`), `description` | Ação a executar na rota |
| `EvaluatedRoute` | `chosen_route_id`, `justification`, `required_action` | Resposta da seleção de rota |

**Função auxiliar**

`_serialize_response(content) → dict`  
Normaliza respostas do modelo independentemente do tipo: aceita modelos Pydantic (HF/outlines), strings JSON (Ollama/OpenAI) ou dicionários Python.

**Classe `LLMAgent`**

`__init__(model_name, method='hf', api_key=None)`  
Inicializa o backend escolhido. Para `hf`, carrega o modelo na GPU (com fallback via accelerate). Para `openai`, lê a chave da API do argumento ou da variável de ambiente `OPENAI_API_KEY`.

`classify_tasks(tasks: list) → dict`  
Envia a lista de tarefas ao modelo e retorna um dicionário com a chave `"tasks"` contendo os objetos de tarefa classificados. Retorna `{}` em caso de falha.

`evaluate_routes(user_context, scenario_context, routes) → dict`  
Pede ao modelo que escolha a melhor rota. Retorna um dicionário com `chosen_route_id`, `justification` e `required_action`. Retorna `{"error": ...}` em caso de falha.

`print_timing_report()`  
Imprime estatísticas de contagem de chamadas e latência por operação.

---

### `src/context_engine.py` — Provedor de Contexto

Retorna dados de preferência do usuário e informações do cenário em tempo real utilizados pelo modelo de linguagem para personalizar a seleção de rotas.

**Classe `ContextEngine`**

`get_user_context() → dict`  
Retorna preferências e regras de evitação do usuário.

`get_scenario_context(origin=None, destination=None) → dict`  
Retorna hora atual, dia da semana, condições de tráfego e clima.

> Na implementação atual, os valores são estáticos (adequados para benchmark e demonstração). Para um sistema em produção, substitua por chamadas a APIs de tráfego e clima em tempo real.

---

### `utils/bench.py` — Executor do Benchmark

Executa todos os cenários da lista `SCENARIOS`, captura o JSON de decisão do modelo a partir da saída de cada subprocesso e produz um relatório de resultados. Todos os valores padrão são lidos de `config.json`.

**Argumentos de linha de comando**

| Argumento | Padrão | Descrição |
|---|---|---|
| `--method` | `config.benchmark.llm_method` | Backend de LLM para todas as execuções |
| `--runs` | `config.benchmark.runs_per_scenario` | Repetições por cenário |
| `--timeout` | `config.benchmark.timeout_seconds` | Timeout por execução (segundos) |
| `--log-dir` | `config.benchmark.log_directory` | Diretório para logs individuais por cenário |

**Funções principais**

`run_benchmark(llm_method, runs_per_scenario, timeout_seconds, log_directory)`  
Laço principal: itera sobre rodadas e cenários, chama `src/app.py` como subprocesso, classifica o resultado e gera o relatório no bloco `finally`.

`_extract_llm_json(log_content) → dict | None`  
Extrai o JSON de avaliação de rota da saída do subprocesso localizando o marcador `--- LLM Response (Route Evaluation JSON) ---`.

`_classify_result(output, scenario, llm_data, duration) → (label, precision, completeness)`  
Mapeia os dados extraídos do modelo para PASS / FAIL / TIMEOUT / GEO_ERR / NO_LLM.

`_generate_report(results, output_dir)`  
Grava `benchmark_summary_report.txt` e `benchmark_raw_data.csv` na raiz do projeto.

---

### `utils/scenario_generator.py` — Mineração Espacial de Cenários

Ferramenta offline utilizada para gerar a lista `SCENARIOS` presente em `bench.py`. Requer os arquivos de grafo e POIs pré-baixados (os caminhos absolutos devem ser configurados diretamente no script). Não faz parte do fluxo padrão de reprodução.

---

## Dependências

Todas as dependências Python estão declaradas em `requirements.txt`. Bibliotecas principais:

| Biblioteca | Função |
|---|---|
| `osmnx` | Download do grafo viário, geocodificação e visualização |
| `networkx` | Algoritmos de caminho mínimo |
| `geopandas` / `shapely` | Operações espaciais sobre geometrias de POIs |
| `transformers` | Carregamento de modelos HuggingFace |
| `outlines` | Geração estruturada (restrita a esquema) com modelos de linguagem |
| `ollama` | Cliente Python para o daemon Ollama |
| `openai` | Cliente da API OpenAI ChatCompletion |
| `pydantic` | Validação de dados e esquemas de saída estruturada |
| `matplotlib` | Renderização do mapa de rotas |

Dependências de sistema: `git`, `curl` (para instalação do Ollama, caso utilize esse backend).
