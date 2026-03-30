import subprocess
import re
import json
from datetime import datetime
import sys
import os
import time
import pandas as pd # Necessário para a análise por categoria
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# --- CONFIGURAÇÕES ---
CONTEXT_ENGINE_PATH = "src/context_engine.py" 
LOG_DIRECTORY = "benchmark_logs" 
RUNS_PER_SCENARIO = 3

# Método padrão para o benchmark. Importante:
# - 'hf' atualmente retorna objetos Pydantic via Outlines no llm_agent.py, e o app.py tenta fazer json.loads
#   direto, o que pode resultar em NO_LLM por falha de parse.
# - 'ollama' retorna JSON string e tende a ser o caminho mais estável para avaliação.
# - 'openai' usa a API da OpenAI (requer OPENAI_API_KEY definida no ambiente)
DEFAULT_LLM_METHOD = "hf"

# Arquivo para salvar todo o output do terminal
FULL_LOG_FILE = f"benchmark_full_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# --- CLASSE PARA SALVAR O TERMINAL NO TXT ---
class DualLogger:
    """Duplica o output: imprime na tela e salva no arquivo ao mesmo tempo."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Garante que grave em tempo real

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- LISTA DE CENÁRIOS ---
SCENARIOS = [
    {
        'scenario_name': "Urgência - Combustível - Exemplo 1",
        'origin': "-23.622872, -46.621864",
        'destination': "-23.620355, -46.622481",
        'tasks': ['I need to go to a fuel'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Ipiranga"
    },
    {
        'scenario_name': "Urgência - Combustível - Exemplo 2",
        'origin': "-23.545907, -46.393073",
        'destination': "-23.563572, -46.394605",
        'tasks': ['I need to go to a fuel'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: BR"
    },
    {
        'scenario_name': "Urgência - Combustível - Exemplo 3",
        'origin': "-23.602212, -46.553596",
        'destination': "-23.596540, -46.540776",
        'tasks': ['I need to go to a fuel'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Shell"
    },
    {
        'scenario_name': "Urgência - Combustível - Exemplo 4",
        'origin': "-23.581839, -46.583463",
        'destination': "-23.573511, -46.571406",
        'tasks': ['I need to go to a fuel'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: BR"
    },
    {
        'scenario_name': "Urgência - Combustível - Exemplo 5",
        'origin': "-23.611788, -46.700331",
        'destination': "-23.607708, -46.683156",
        'tasks': ['I need to go to a fuel'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Vibe"
    },
    {
        'scenario_name': "Urgência - Combustível - Exemplo 6",
        'origin': "-23.474240, -46.670398",
        'destination': "-23.470588, -46.654329",
        'tasks': ['I need to go to a fuel'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Setee"
    },
    {
        'scenario_name': "Urgência - Hospital - Exemplo 1",
        'origin': "-23.490132, -46.600877",
        'destination': "-23.492685, -46.582251",
        'tasks': ['I need to go to a hospital'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Hospital Presidente"
    },
    {
        'scenario_name': "Urgência - Hospital - Exemplo 2",
        'origin': "-23.565719, -46.744279",
        'destination': "-23.566416, -46.752697",
        'tasks': ['I need to go to a hospital'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Hospital Universitário"
    },
    {
        'scenario_name': "Urgência - Hospital - Exemplo 3",
        'origin': "-23.539192, -46.698872",
        'destination': "-23.549677, -46.706607",
        'tasks': ['I need to go to a hospital'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Hospital e Maternidade Metropolitano"
    },
    {
        'scenario_name': "Urgência - Hospital - Exemplo 4",
        'origin': "-23.658714, -46.654622",
        'destination': "-23.664674, -46.645434",
        'tasks': ['I need to go to a hospital'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Hospital Municipal Vila Santa Catarina Dr. Gilson de C. Marques de Carvalho"
    },
    {
        'scenario_name': "Urgência - Hospital - Exemplo 5",
        'origin': "-23.754212, -46.679021",
        'destination': "-23.768073, -46.686845",
        'tasks': ['I need to go to a hospital'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Pronto Socorro Municipal Dona Maria Antonieta Ferreira de Barros"
    },
    {
        'scenario_name': "Urgência - Hospital - Exemplo 6",
        'origin': "-23.505987, -46.411997",
        'destination': "-23.504042, -46.422313",
        'tasks': ['I need to go to a hospital'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Hospital Tide Setubal"
    },
    {
        'scenario_name': "Conveniência - Mercado - Exemplo 1",
        'origin': "-23.690454, -46.783143",
        'destination': "-23.680629, -46.788937",
        'tasks': ['I need to go to a supermarket'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Supermercado Ricoy"
    },
    {
        'scenario_name': "Conveniência - Mercado - Exemplo 2",
        'origin': "-23.500053, -46.575261",
        'destination': "-23.490045, -46.577613",
        'tasks': ['I need to go to a supermarket'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Hokim"
    },
    {
        'scenario_name': "Conveniência - Mercado - Exemplo 3",
        'origin': "-23.587876, -46.730672",
        'destination': "-23.581851, -46.715257",
        'tasks': ['I need to go to a supermarket'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Empório Morato"
    },
    {
        'scenario_name': "Conveniência - Mercado - Exemplo 4",
        'origin': "-23.511786, -46.700826",
        'destination': "-23.507543, -46.709314",
        'tasks': ['I need to go to a supermarket'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Sonda Supermercado"
    },
    {
        'scenario_name': "Conveniência - Mercado - Exemplo 5",
        'origin': "-23.486258, -46.690488",
        'destination': "-23.475079, -46.690265",
        'tasks': ['I need to go to a supermarket'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Sonda"
    },
    {
        'scenario_name': "Conveniência - Mercado - Exemplo 6",
        'origin': "-23.489953, -46.709941",
        'destination': "-23.501840, -46.718452",
        'tasks': ['I need to go to a supermarket'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Pastorinho"
    },
    {
        'scenario_name': "Lazer - Parque - Exemplo 1",
        'origin': "-23.583829, -46.520412",
        'destination': "-23.572917, -46.525485",
        'tasks': ['I need to go to a park'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Praça Comendador José Gonzalez"
    },
    {
        'scenario_name': "Lazer - Parque - Exemplo 2",
        'origin': "-23.511623, -46.759952",
        'destination': "-23.506818, -46.747002",
        'tasks': ['I need to go to a park'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Praça Guerino Ricciotti"
    },
    {
        'scenario_name': "Lazer - Parque - Exemplo 3",
        'origin': "-23.782440, -46.683925",
        'destination': "-23.765631, -46.680630",
        'tasks': ['I need to go to a park'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Praça das Pedras"
    },
    {
        'scenario_name': "Lazer - Parque - Exemplo 4",
        'origin': "-23.679055, -46.704245",
        'destination': "-23.668706, -46.693364",
        'tasks': ['I need to go to a park'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Praça Contendas"
    },
    {
        'scenario_name': "Lazer - Parque - Exemplo 5",
        'origin': "-23.599506, -46.615200",
        'destination': "-23.602851, -46.630887",
        'tasks': ['I need to go to a park'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Praça Amim Ghorayeb"
    },
    {
        'scenario_name': "Lazer - Parque - Exemplo 6",
        'origin': "-23.551637, -46.676458",
        'destination': "-23.540619, -46.677005",
        'tasks': ['I need to go to a park'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Praça Rubens do Amaral"
    },
]

CONTEXT_TEMPLATE_STR = """from datetime import datetime

class ContextEngine:
    \"\"\"
    MOCK AUTOMATIZADO PARA BENCHMARK (DADOS REAIS DE SP)
    \"\"\"
    def get_user_context(self) -> dict:
        return {{
            "preferences": ["avoid downtown during rush hour", "prefers safer routes at night"],
            "avoidance_rules": {rules_json} 
        }}

    def get_scenario_context(self, origin=None, dest=None) -> dict:
        return {{
            "current_time": datetime.now().strftime("%H:%M"),
            "day_of_week": "Tuesday",
            "traffic_conditions": "moderate",
            "weather": "cloudy"
        }}
"""

def update_context_file(rule: dict = None):
    """Atualiza context_engine.py com regras de 'avoidance'."""
    rules_list = [rule] if rule else []
    content = CONTEXT_TEMPLATE_STR.format(rules_json=json.dumps(rules_list))
    with open(CONTEXT_ENGINE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

def extract_llm_json(log_content: str) -> dict:
    """Extrai o JSON de decisão da LLM de forma robusta."""
    marker = "--- LLM Response (Route Evaluation JSON) ---"
    start_index = log_content.find(marker)
    if start_index == -1: return None
    
    json_start = log_content.find("{", start_index)
    if json_start == -1: return None
    
    brace_count = 0
    json_end = -1
    for i in range(json_start, len(log_content)):
        char = log_content[i]
        if char == '{': brace_count += 1
        elif char == '}': brace_count -= 1
        if brace_count == 0:
            json_end = i + 1
            break
            
    if json_end != -1:
        json_str = log_content[json_start:json_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def generate_report(results_data):
    """Gera um relatório final agregado e salva em arquivo."""
    df = pd.DataFrame(results_data)
    
    if not df.empty:
        df['Categoria'] = df['Scenario'].apply(lambda x: x.split(' - Exemplo')[0])
    
    valid_runs = df[~df['Result'].isin(['TIMEOUT', 'GEO_ERR', 'NO_LLM', 'ERROR'])]
    
    report_file = "benchmark_summary_report.txt"
    with open(report_file, "w", encoding='utf-8') as f:
        f.write("=== RELATÓRIO FINAL DETALHADO ===\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tentativas: {len(df)}\n")
        f.write(f"Execuções Válidas (Sem Timeout/Erro): {len(valid_runs)}\n")
        f.write("="*60 + "\n\n")

        # 1. Performance Cognitiva
        if not valid_runs.empty:
            acc = valid_runs['Precision'].mean() * 100
            comp = valid_runs['Completeness'].mean() * 100
            f.write("--- 1. PERFORMANCE COGNITIVA (Acurácia Real) ---\n")
            f.write(f"Precision (Escolha Rota):   {acc:.2f}%\n")
            f.write(f"Completeness (Intenção):    {comp:.2f}%\n")
        else:
            f.write("--- 1. PERFORMANCE COGNITIVA ---\n")
            f.write("Nenhum dado válido para cálculo.\n")

        # 2. Estabilidade do Sistema
        f.write("\n--- 2. ESTABILIDADE ---\n")
        timeouts = len(df[df['Result'] == 'TIMEOUT'])
        errors = len(df[df['Result'].isin(['GEO_ERR', 'NO_LLM', 'ERROR'])])
        f.write(f"Timeouts: {timeouts} ({timeouts/len(df)*100:.1f}%)\n")
        f.write(f"Outros Erros: {errors}\n\n")

        # 3. Análise por Categoria
        f.write("--- 3. DETALHE POR CATEGORIA ---\n")
        
        if not valid_runs.empty:
            # Agrupa apenas os válidos para ter a porcentagem de acerto real
            valid_grouped = valid_runs.groupby('Categoria').agg(
                Precision=('Precision', 'mean'),
                Completeness=('Completeness', 'mean'),
                Avg_Time=('Duration', 'mean'),
                Count=('Run', 'count')
            )
            
            # Formatação
            valid_grouped['Precision'] = (valid_grouped['Precision'] * 100).map("{:.1f}%".format)
            valid_grouped['Completeness'] = (valid_grouped['Completeness'] * 100).map("{:.1f}%".format)
            valid_grouped['Avg_Time'] = valid_grouped['Avg_Time'].map("{:.2f}s".format)
            
            f.write(valid_grouped.to_string())
        else:
            f.write("Não há dados válidos suficientes para agrupar por categoria.")
            
        f.write("\n\n")

        # 4. Dados Brutos de Erros
        f.write("--- 4. EXECUÇÕES COM PROBLEMAS TÉCNICOS ---\n")
        errors_df = df[df['Result'] != 'PASS']
        if not errors_df.empty:
            # Seleciona colunas relevantes para o log de erro
            cols = ['Scenario', 'Run', 'Result', 'Duration']
            f.write(errors_df[cols].to_string(index=False))
        else:
            f.write("Nenhuma falha registrada.")
            
    print(f"\nRelatório de análise salvo em: {report_file}")
    df.to_csv("benchmark_raw_data.csv", index=False)
    print("Dados brutos (CSV) salvos em: benchmark_raw_data.csv")

def run_benchmark():
    # ATIVA O LOGGER DUPLO
    sys.stdout = DualLogger(FULL_LOG_FILE)

    print(f"--- INICIANDO BENCHMARK ---")
    print(f"Log Completo: {FULL_LOG_FILE}")
    print(f"Método LLM: {DEFAULT_LLM_METHOD}")
    if DEFAULT_LLM_METHOD == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"✓ OPENAI_API_KEY configurada (***{api_key[-4:]})")
        else:
            print("⚠ AVISO: OPENAI_API_KEY não encontrada no ambiente!")
    print("-" * 60)

    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    python_exec = sys.executable
    
    # 1. Backup do Contexto Original
    original_context = ""
    if os.path.exists(CONTEXT_ENGINE_PATH):
        with open(CONTEXT_ENGINE_PATH, 'r', encoding='utf-8') as f:
            original_context = f.read()

    # LISTA PARA ARMAZENAR DADOS DETALHADOS
    results_data = []

    try:
        for run_idx in range(1, RUNS_PER_SCENARIO + 1):
            print(f"\n>>> RODADA DE EXECUÇÃO {run_idx}/{RUNS_PER_SCENARIO}")
            
            for i, scen in enumerate(SCENARIOS):
                scen_name = scen['scenario_name']
                print(f"   ({i+1}/{len(SCENARIOS)}) {scen_name}...", end=" ", flush=True)
                
                update_context_file(scen.get("avoid_rule"))
                
                method = scen.get("method", DEFAULT_LLM_METHOD)
                # Monta comando - origem/destino sem espaços após vírgula
                # Usa formato --origem=valor para evitar que números negativos sejam interpretados como flags
                origin = scen["origin"].replace(", ", ",")
                destination = scen["destination"].replace(", ", ",")
                cmd = [python_exec, "src/app.py",
                       f"--origem={origin}",
                       f"--destino={destination}",
                       f"--method={method}"]
                if scen.get("tasks"):
                    cmd.append("--tarefas")
                    cmd.extend(scen["tasks"])

                start_time = time.time()
                try:
                    process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=500)
                    output = process.stdout + "\n" + process.stderr
                except subprocess.TimeoutExpired:
                    print("[TIMEOUT]")
                    output = "TIMEOUT"
                except Exception as e:
                    print(f"[ERRO: {e}]")
                    output = str(e)
                
                duration = time.time() - start_time

                # Log Individual
                safe_name = scen_name.replace(" ", "_").replace("/", "-")
                log_file = os.path.join(LOG_DIRECTORY, f"run{run_idx}_{safe_name}.log")
                with open(log_file, "w", encoding='utf-8') as f:
                    f.write(f"CMD: {' '.join(cmd)}\n\n{output}")

                # Análise
                llm_data = extract_llm_json(output)
                
                precision_pass = False
                completeness_pass = False
                result_str = "ERROR"

                if llm_data:
                    chosen_id = llm_data.get("chosen_route_id")
                    req_action = llm_data.get("required_action", {}) if isinstance(llm_data, dict) else {}
                    action_type = req_action.get("type", "NONE")

                    # Precision: escolha de rota (1/2) como planejado
                    if chosen_id == scen.get("expected_choice"):
                        precision_pass = True

                    # Completeness: para o PAVe atual, a intenção é atendida quando o LLM escolhe
                    # a rota esperada (ex.: rota com paradas quando há tarefa).
                    # Se no futuro o LLM preencher required_action.type, também aceitamos isso.
                    expected_action = scen.get(
                        "expected_action_type",
                        "ADD_WAYPOINT" if scen.get("recalculation_expected", False) else "NONE",
                    )

                    if expected_action == "ADD_WAYPOINT":
                        completeness_pass = (chosen_id == scen.get("expected_choice")) or (action_type == "ADD_WAYPOINT")
                    else:
                        completeness_pass = (chosen_id == scen.get("expected_choice")) or (action_type == "NONE")

                    result_str = "PASS" if (precision_pass and completeness_pass) else "FAIL"
                    print(f"[{result_str}] ({duration:.1f}s)")
                else:
                    if "TIMEOUT" in output: result_str = "TIMEOUT"
                    elif "Geocoding Error" in output: result_str = "GEO_ERR"
                    else: result_str = "NO_LLM"
                    print(f"[{result_str}] ({duration:.1f}s)")

                # REGISTRA OS DADOS NA LISTA
                results_data.append({
                    "Scenario": scen_name,
                    "Run": run_idx,
                    "Duration": duration,
                    "Precision": 1 if precision_pass else 0,
                    "Completeness": 1 if completeness_pass else 0,
                    "Result": result_str,
                    "Chosen_ID": llm_data.get("chosen_route_id") if llm_data else None,
                    "Action": llm_data.get("required_action", {}).get("type") if llm_data else None,
                    "Expected_Action": scen.get(
                        "expected_action_type",
                        "ADD_WAYPOINT" if scen.get("recalculation_expected", False) else "NONE",
                    ),
                    "Method": scen.get("method", DEFAULT_LLM_METHOD),
                })

    finally:
        if original_context:
            with open(CONTEXT_ENGINE_PATH, 'w', encoding='utf-8') as f:
                f.write(original_context)
            print("\nContexto restaurado.")
            
        # Gera relatório final mesmo se cancelar no meio
        generate_report(results_data)

if __name__ == "__main__":
    run_benchmark()