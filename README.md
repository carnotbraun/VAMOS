# VAMOS - Vehicular Agent for Multi-objective Optimization and Semantics

## Visão Geral

**VAMOS** (Vehicular Agent for Multi-objective Optimization and Semantics) é um sistema avançado de roteamento inteligente que integra Modelos de Linguagem de Grande Escala (LLMs) para planejamento contextual de rotas urbanas. O sistema combina processamento de linguagem natural, otimização de grafos e dados geoespaciais do OpenStreetMap para gerar rotas personalizadas que consideram múltiplas tarefas, preferências do usuário e contextos dinâmicos.

### Características Principais

- **Planejamento Multi-Tarefa**: Processa múltiplas tarefas em linguagem natural e as incorpora ao planejamento de rota
- **Classificação Inteligente**: Utiliza LLMs para classificar tarefas por importância e extrair tags OSM relevantes automaticamente
- **Otimização TSP**: Resolve o Problema do Caixeiro Viajante para determinar a ordem ótima de visita aos POIs (Points of Interest)
- **Suporte Multi-Modelo**: Compatível com Ollama (local), HuggingFace e OpenAI
- **Análise Contextual**: Considera horário, tráfego, clima e preferências do usuário
- **Visualização Geográfica**: Gera mapas de alta qualidade das rotas otimizadas

## Arquitetura do Sistema

O sistema é composto por quatro componentes principais:

```
┌─────────────────┐
│   LLM Agent     │  → Classificação de tarefas e extração de POI tags
└────────┬────────┘
         │
┌────────▼────────┐
│ Context Engine  │  → Contexto do usuário e cenário (tempo, tráfego)
└────────┬────────┘
         │
┌────────▼────────┐
│ Routing Engine  │  → Cálculo de rotas e otimização TSP
└────────┬────────┘
         │
┌────────▼────────┐
│  Graph Utils    │  → Gerenciamento de grafos OSM e POIs
└─────────────────┘
```

### Componentes

#### 1. **LLM Agent** ([llm_agent.py](src/llm_agent.py))
Responsável pela interação com modelos de linguagem:
- Classifica tarefas em linguagem natural
- Extrai tags OSM relevantes para cada tarefa
- Avalia rotas candidatas considerando contexto
- Suporta três backends: Ollama, HuggingFace, OpenAI

#### 2. **Context Engine** ([context_engine.py](src/context_engine.py))
Fornece informações contextuais para a tomada de decisão:
- Preferências do usuário (rotas mais seguras, evitar áreas, etc.)
- Contexto temporal (hora do dia, dia da semana)
- Condições dinâmicas (tráfego, clima)

#### 3. **Routing Engine** ([routing_engine.py](src/routing_engine.py))
Motor de cálculo de rotas:
- Conversão de endereços para nós do grafo
- Cálculo de caminhos mais curtos (Dijkstra)
- Otimização de múltiplas paradas (TSP brute-force)
- Suporte a pesos customizados (tempo de viagem, distância)

#### 4. **Graph Utils** ([graph_utils.py](src/graph_utils.py))
Gerenciamento de dados geoespaciais:
- Download e cache de grafos OSM
- Extração e cache de POIs
- Projeção de coordenadas para cálculos precisos

## Instalação

### Requisitos
- Python 3.8+
- CUDA (opcional, para HuggingFace com GPU)
- Ollama (opcional, para execução local)

### Configuração

1. Clone o repositório:
```bash
git clone <repository_url>
cd VAMOS
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. (Opcional) Configure o Ollama para execução local:
```bash
ollama pull llama3
```

4. (Opcional) Configure a API OpenAI:
```bash
export OPENAI_API_KEY="sua-chave-api"
```

## Uso

### Linha de Comando

O sistema é executado via linha de comando com os seguintes parâmetros:

```bash
python src/app.py \
  --method [ollama|hf|openai] \
  --origem "Endereço de origem" \
  --destino "Endereço de destino" \
  --tarefas "tarefa 1" "tarefa 2" "tarefa N"
```

### Exemplos Práticos

#### Exemplo 1: Rota com compras
```bash
python src/app.py \
  --method ollama \
  --origem "Rua Reitor Lafayete, Campinas" \
  --destino "Shopping Iguatemi, Campinas" \
  --tarefas "comprar remédios" "comprar pão"
```

#### Exemplo 2: Rota de emergência
```bash
python src/app.py \
  --method openai \
  --origem "-22.90,47.06" \
  --destino "-22.88,47.04" \
  --tarefas "preciso abastecer urgente" "ir ao hospital"
```

#### Exemplo 3: Rota de lazer
```bash
python src/app.py \
  --method hf \
  --origem "Centro, Campinas" \
  --destino "Parque Portugal, Campinas" \
  --tarefas "visitar um parque" "tomar café"
```

### Parâmetros

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `--method` | String | Não | Backend do LLM: `ollama`, `hf` ou `openai` (padrão: `ollama`) |
| `--origem` | String | Sim | Endereço ou coordenadas (lat,lon) de origem |
| `--destino` | String | Sim | Endereço ou coordenadas (lat,lon) de destino |
| `--tarefas` | Lista | Não | Lista de tarefas em linguagem natural |
| `--force_download` | Flag | Não | Força re-download de grafos e POIs (ignora cache) |

## Fluxo de Execução

1. **Carregamento de Dados**
   - Carrega ou baixa o grafo OSM da região
   - Carrega ou extrai POIs relevantes
   - Projeta dados para cálculos métricos precisos

2. **Processamento de Tarefas**
   - LLM classifica cada tarefa por importância (1-10)
   - Extrai tags OSM relevantes para cada tarefa
   - Ordena tarefas por prioridade

3. **Geração de Rotas Candidatas**
   - **Rota Direta**: Caminho mais curto sem paradas
   - **Rota Multi-tarefa**: Incorpora POIs das tarefas
     - Identifica POIs relevantes próximos à rota
     - Otimiza ordem de visita (TSP)
     - Calcula rota completa

4. **Avaliação e Decisão**
   - LLM avalia rotas considerando:
     - Tempo total de viagem
     - Número de tarefas completadas
     - Contexto (tráfego, horário, preferências)
   - Seleciona rota ótima e justifica a decisão

5. **Visualização**
   - Gera mapa de alta resolução (300 DPI)
   - Destaca rota em vermelho
   - Marca POIs visitados em azul
   - Salva como `rota_final.png`

## Modelos Suportados

### HuggingFace
- **Qwen3-4B** (baseline)
- Qwen3-0.6B, Qwen3-1.7B, Qwen3-8B
- LFM2.5-1.2B-Instruct
- Mistral-7B-Instruct
- Phi-3.5-mini-instruct
- Meta-Llama-3.1-8B-Instruct

### Ollama
- llama3
- mistral
- phi3

### OpenAI
- gpt-3.5-turbo
- gpt-4-turbo-preview
- gpt-4o

## Estrutura do Projeto

```
VAMOS/
├── src/
│   ├── app.py              # Aplicação principal
│   ├── llm_agent.py        # Interface com LLMs
│   ├── context_engine.py   # Gerenciamento de contexto
│   ├── routing_engine.py   # Motor de roteamento
│   └── graph_utils.py      # Utilitários de grafos
├── utils/
│   ├── bench.py            # Script de benchmarking
│   ├── explore_nodes.py    # Exploração de nós
│   └── scenario_generator.py # Gerador de cenários
├── notebook/
│   └── plots.ipynb         # Análises e visualizações
├── benchmark_logs/         # Logs de experimentos
├── graph_data/             # Cache de grafos OSM
├── mapas/                  # Mapas gerados
├── requirements.txt        # Dependências Python
└── README.md              # Este arquivo
```

## Detalhes Técnicos

### Otimização TSP
O sistema resolve o TSP (Traveling Salesman Problem) para múltiplas paradas usando:
- Enumeração exaustiva via permutações (para n ≤ 10 POIs)
- Matriz de distâncias pré-calculada
- Validação de conectividade entre todos os nós

### Seleção de POIs
POIs são selecionados por proximidade à rota direta:
1. Projeta grafo e POIs para sistema métrico
2. Constrói LineString da rota direta
3. Calcula distância de cada POI à linha
4. Seleciona POI mais próximo para cada tarefa

### Structured Output
O LLM é configurado para gerar outputs estruturados via Pydantic:
```python
class Task(BaseModel):
    task: str
    importance: int  # 1-10
    poi_tags: Dict   # OSM tags
```

Isso garante outputs parseáveis e previne alucinações.

## Limitações Conhecidas

- **Escalabilidade TSP**: Tempo exponencial para n > 10 POIs
- **Geocodificação**: Dependente de APIs externas (pode falhar offline)
- **Contexto Dinâmico**: Dados de tráfego e clima são simulados (mock)
- **Cache Local**: Requer re-download para regiões não cachadas

## Trabalhos Futuros

- [ ] Implementar heurísticas TSP (Christofides, 2-opt)
- [ ] Integração com APIs de tráfego real (Google, TomTom)
- [ ] Suporte a múltiplos modos de transporte
- [ ] Interface web/móvel
- [ ] Aprendizado por reforço para preferências

## Citação

Se você utilizar este sistema em sua pesquisa, por favor cite:

```bibtex
@article{vamos2026,
  title={VAMOS: Versatile Agent for Multimodal Optimization of Streets},
  author={[Seu Nome]},
  journal={[Nome da Conferência/Journal]},
  year={2026}
}
```

## Licença

[Adicione informações de licença]

## Contato

[Adicione informações de contato]

## Agradecimentos

Este projeto utiliza:
- [OSMnx](https://github.com/gboeing/osmnx) para dados do OpenStreetMap
- [Outlines](https://github.com/outlines-dev/outlines) para structured generation
- [Ollama](https://ollama.ai/) para execução local de LLMs