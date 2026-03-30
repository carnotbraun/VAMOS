# VAMOS: Agente Híbrido para Planejamento de Rotas Veiculares Cientes de Contexto Semântico

**Resumo do Artigo:** Sistemas de navegação tradicionais priorizam a eficiência métrica, como tempo ou distância, mas falham frequentemente na interpretação de intenções humanas complexas e dependentes do contexto. Embora os Grandes Modelos de Linguagem (LLMs) demonstrem potencial para preencher essa lacuna semântica, sua integração direta em Sistemas de Transporte Inteligentes (ITS) enfrenta barreiras críticas de escalabilidade, latência e dependência de conectividade. Para superar esses desafios, este trabalho apresenta o VAMOS (Vehicular Agent for Multi-objective Optimization and Semantics), um agente híbrido desenhado para operar eficientemente embarcado. O VAMOS desacopla o raciocínio semântico da otimização espacial, combinando Pequenos Modelos de Linguagem (SLMs) locais para a interpretação de intenções com algoritmos de grafos clássicos para a execução de rotas. A avaliação experimental em três cenários urbanos demonstra que o VAMOS atinge acurácia e completude superiores a 91% utilizando modelos compactos. Além disso, os resultados evidenciam um trade-off favorável: embora modelos massivos apresentem um ganho marginal de qualidade (~3%), o VAMOS oferece uma redução significativa no overhead computacional e de comunicação, validando a viabilidade de assistentes de navegação semanticamente conscientes.

![](framework.png)

# Estrutura do readme.md

A documentação está estruturada para orientar o processo de avaliação, contendo:
1. Selos pretendidos.
2. Informações básicas de hardware e software.
3. Dependências do sistema.
4. Preocupações de segurança.
5. Instruções de instalação.
6. Teste mínimo de funcionalidade.
7. Guias de reprodução dos experimentos/reivindicações do artigo.
8. Licença.

A arquitetura do projeto reflete-se na seguinte estrutura de diretórios:
- `/src`: Módulos centrais (App, Motor de Roteamento, Agente LLM e Geo-Utils).
- `/utils`: Ferramentas de benchmarking e geradores de cenários.
- `/mapas`: Saídas visuais das rotas geradas (e utilitários).
- `/graph_data`: (Gerado na execução) Cache e persistência das malhas viárias.

# Selos Considerados

Os selos considerados para avaliação são: **Artefatos Disponíveis (SeloD)**, **Artefatos Funcionais (SeloF)**, **Artefatos Sustentáveis (SeloS)** e **Experimentos Reprodutíveis (SeloR)**.

# Informações básicas

O sistema foi concebido para execução local (borda/embarcado), visando minimizar a latência e a dependência de rede.
- **Sistema Operacional:** Linux (Ubuntu 20.04/22.04 LTS) ou MacOs.
- **Hardware Mínimo:** Processador multi-core, 16 GB de RAM, 10 GB de armazenamento.
- **Hardware Recomendado:** Placa gráfica (GPU) com arquitetura NVIDIA (suporte a CUDA) e um mínimo de 8GB de VRAM para a execução eficiente do modelo SLM (`Qwen3-4B`) sem offload para disco.
- **Ambiente de Execução:** Python 3.10.

# Dependências

Para a execução deste artefato, requer-se a instalação dos seguintes componentes:
- **Sistema:** `git`, `curl`.
- **Bibliotecas Python principais (ver `requirements.txt`):** `osmnx` (grafos), `networkx` (algoritmos), `geopandas`/`pandas`, `matplotlib`, `outlines`, `ollama` e `transformers`.

*(Opcional)*: Chaves de API para OpenAI podem ser passadas via variável de ambiente.

# Preocupações com segurança

A execução do artefato não apresenta riscos para o hardware do avaliador. O sistema efetua downloads pontuais das redes viárias através da API pública do OpenStreetMap, necessitando de acesso à internet na primeira execução de um cenário urbano inédito. Caso o avaliador utilize *tokens* de API pagos (ex: OpenAI) para testes avulsos, adverte-se para que não os adicione permanentemente ao código fonte para prevenir exposições acidentais em *commits*.

# Instalação

O processo de instalação configura o ambiente virtual e prepara o motor de inferência local.

1. Clone o repositório:
```bash
git clone https://github.com/carnotbraun/VAMOS
cd VAMOS
```

2. Configure o ambiente virtual e as dependências:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

3. Instale o serviço Ollama e transfira o modelo base do artigo (Qwen):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  
ollama pull qwen:4b
```

# Teste mínimo

O teste mínimo assegura que as bibliotecas espaciais, as projeções geográficas e a comunicação local com o modelo de linguagem estão devidamente configuradas.

**Procedimento:**
No terminal, com o ambiente ativado:
```bash
python src/app.py \
  --method ollama \
  --origem "Avenida Paulista, São Paulo" \
  --destino "Parque Ibirapuera, São Paulo" \
  --tarefas "preciso de combustivel urgente"
```

**Resultado Esperado:**
O sistema fará o download do polígono viário do OpenStreetMap (pode demorar 1-2 minutos na primeira vez, ficando posteriormente em cache). O log exibirá o Agente identificando "combustível" como prioridade máxima (Importância 10), o motor de rotas resolverá o Problema do Caixeiro Viajante (TSP) para inserir um posto de abastecimento, e uma imagem `rota_final.png` será gerada destacando a trajetória ótima na raiz do projeto.

# Experimentos

Esta secção descreve os passos para a obtenção dos resultados cruciais apresentados no artigo.

## Reivindicação #1: Alta Acurácia e Completude na Execução Local
O artigo argumenta (Tabela 2) que o modelo compacto `Qwen3-4B` processando localmente alcança mais de 91% de eficácia ao interpretar intenções humanas perante cenários topológicos e decidir sobre desvios semânticos (Urgência vs. Conveniência).

1. **Configuração:** Garanta que possui o Ollama ativo e as dependências instaladas, ou que vai fazer o uso do modelo extraido do HF, considere os que já estão comentados no código principal.
2. **Execução:** O ficheiro `bench.py` processará automaticamente dezenas de cenários espacialmente verificados. 
```bash
python utils/bench.py
```
3. **Recursos e Duração:** Dependendo da capacidade da GPU do avaliador, o teste completo consome entre 30 a 50 minutos. Serão utilizados ~4GB de RAM para instanciar o grafo da metrópole e ~5GB de VRAM para a inferência local da LLM.
4. **Resultado Esperado:** O script gera o log completo e o ficheiro sumário `benchmark_summary_report.txt`. Neste último, nas secções de "Performance Cognitiva", os valores consolidados de *Precision* (escolha da rota certa) e *Completeness* (intenção adequada) corroborarão o padrão superior a 90% atestado no artigo, considerando as margens naturais de não-determinismo probabilístico dos modelos autoregressivos.

## Reivindicação #2: Redução de Overhead de Comunicação (Estabilidade)
O artigo defende (Tabela 3) que a adoção de SLMs locais oferece um *overhead* fim-a-fim estável comparativamente a instâncias em nuvem que dependem das instabilidades e latências associadas à rede móvel veicular.

1. **Execução:** Esta métrica é extraída como subproduto da execução da Reivindicação #1. Não há ação extra requerida.
2. **Resultado Esperado:** Analisando os tempos na tabela de "DETALHE POR CATEGORIA" dentro de `benchmark_summary_report.txt`, observar-se-ão os valores de `Avg_Time`. Os tempos reportados atestarão estabilidade absoluta por iteração, operando os processos de Geração, Enriquecimento e Avaliação puramente *offline*, blindando o sistema veicular contra anomalias na conectividade de rede (Jitter/Loss).

# LICENSE

Este projeto é distribuído sob a licença MIT. Para mais detalhes e permissões de replicação, consulte o ficheiro `LICENSE` na raiz do repositório.