"""Benchmark runner for VAMOS reproducibility evaluation.

Executes each scenario from the SCENARIOS list multiple times, captures the LLM
decision JSON from the subprocess output, and produces a summary report.

Usage:
    python utils/bench.py [--method hf|ollama|openai] [--runs N] [--timeout T]

All configuration defaults come from config.json at the project root.
"""

import argparse
import subprocess
import json
from datetime import datetime
import sys
import os
import time
import pandas as pd

# Resolve paths relative to the project root regardless of where the script is called from.
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'config.json')
_CONTEXT_ENGINE_PATH = os.path.join(_PROJECT_ROOT, 'src', 'context_engine.py')
_APP_PATH = os.path.join(_PROJECT_ROOT, 'src', 'app.py')


def _load_config() -> dict:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


_CONFIG = _load_config()
_BENCH_CFG = _CONFIG.get("benchmark", {})

DEFAULT_LLM_METHOD = _BENCH_CFG.get("llm_method", "hf")
DEFAULT_RUNS = _BENCH_CFG.get("runs_per_scenario", 3)
DEFAULT_TIMEOUT = _BENCH_CFG.get("timeout_seconds", 900)
DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, _BENCH_CFG.get("log_directory", "benchmark_logs"))


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

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
        'note': "POI Alvo detectado na mineração: Hospital Municipal Vila Santa Catarina"
    },
    {
        'scenario_name': "Urgência - Hospital - Exemplo 5",
        'origin': "-23.754212, -46.679021",
        'destination': "-23.768073, -46.686845",
        'tasks': ['I need to go to a hospital'],
        'expected_choice': 2,
        'recalculation_expected': True,
        'expected_action_type': 'ADD_WAYPOINT',
        'note': "POI Alvo detectado na mineração: Pronto Socorro Municipal"
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

# ---------------------------------------------------------------------------
# Context engine mock template (written to disk before each subprocess call)
# ---------------------------------------------------------------------------

_CONTEXT_TEMPLATE = """from datetime import datetime

class ContextEngine:
    \"\"\"Automated benchmark mock using static São Paulo context data.\"\"\"

    def get_user_context(self) -> dict:
        return {{
            "preferences": ["avoid downtown during rush hour", "prefers safer routes at night"],
            "avoidance_rules": {rules_json}
        }}

    def get_scenario_context(self, origin=None, destination=None) -> dict:
        return {{
            "current_time": datetime.now().strftime("%H:%M"),
            "day_of_week": "Tuesday",
            "traffic_conditions": "moderate",
            "weather": "cloudy"
        }}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DualLogger:
    """Write every output line to both the terminal and a log file simultaneously."""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding='utf-8')

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def _update_context_file(avoidance_rule: dict = None):
    """Overwrite context_engine.py with a benchmark mock containing the given rule."""
    rules = [avoidance_rule] if avoidance_rule else []
    content = _CONTEXT_TEMPLATE.format(rules_json=json.dumps(rules))
    with open(_CONTEXT_ENGINE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)


def _extract_llm_json(log_content: str) -> dict:
    """Extract the route-evaluation JSON block from subprocess output.

    Looks for the marker line printed by LLMAgent.evaluate_routes(), then
    parses the first complete JSON object that follows it.
    """
    marker = "--- LLM Response (Route Evaluation JSON) ---"
    marker_pos = log_content.find(marker)
    if marker_pos == -1:
        return None

    json_start = log_content.find("{", marker_pos)
    if json_start == -1:
        return None

    brace_depth = 0
    json_end = -1
    for pos in range(json_start, len(log_content)):
        char = log_content[pos]
        if char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
        if brace_depth == 0:
            json_end = pos + 1
            break

    if json_end == -1:
        return None

    try:
        return json.loads(log_content[json_start:json_end])
    except json.JSONDecodeError as exc:
        print(f"  [WARN] JSON parse error in LLM response: {exc}")
        return None


def _classify_result(output: str, scenario: dict, llm_data: dict, duration: float):
    """Determine whether a scenario run passed and return a result record."""
    precision_pass = False
    completeness_pass = False
    result_label = "ERROR"

    if llm_data:
        chosen_id = llm_data.get("chosen_route_id")
        required_action = llm_data.get("required_action", {}) if isinstance(llm_data, dict) else {}
        action_type = required_action.get("type", "NONE")

        if chosen_id == scenario.get("expected_choice"):
            precision_pass = True

        expected_action = scenario.get(
            "expected_action_type",
            "ADD_WAYPOINT" if scenario.get("recalculation_expected", False) else "NONE",
        )

        if expected_action == "ADD_WAYPOINT":
            completeness_pass = (chosen_id == scenario.get("expected_choice")) or (action_type == "ADD_WAYPOINT")
        else:
            completeness_pass = (chosen_id == scenario.get("expected_choice")) or (action_type == "NONE")

        result_label = "PASS" if (precision_pass and completeness_pass) else "FAIL"
    else:
        if "TIMEOUT" in output:
            result_label = "TIMEOUT"
        elif "Geocoding error" in output or "Geocoding Error" in output:
            result_label = "GEO_ERR"
        else:
            result_label = "NO_LLM"

    return result_label, precision_pass, completeness_pass


def _generate_report(results: list, output_dir: str):
    """Write a detailed text report and a raw CSV file from the collected results."""
    data_frame = pd.DataFrame(results)

    if not data_frame.empty:
        data_frame['Category'] = data_frame['Scenario'].apply(lambda x: x.split(' - Exemplo')[0])

    valid_runs = data_frame[~data_frame['Result'].isin(['TIMEOUT', 'GEO_ERR', 'NO_LLM', 'ERROR'])]

    report_path = os.path.join(output_dir, "benchmark_summary_report.txt")
    csv_path = os.path.join(output_dir, "benchmark_raw_data.csv")

    with open(report_path, "w", encoding='utf-8') as report_file:
        report_file.write("=== DETAILED BENCHMARK REPORT ===\n")
        report_file.write(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write(f"Total runs : {len(data_frame)}\n")
        report_file.write(f"Valid runs  : {len(valid_runs)} (excludes TIMEOUT/errors)\n")
        report_file.write("=" * 60 + "\n\n")

        # 1. Cognitive performance
        report_file.write("--- 1. COGNITIVE PERFORMANCE ---\n")
        if not valid_runs.empty:
            precision = valid_runs['Precision'].mean() * 100
            completeness = valid_runs['Completeness'].mean() * 100
            report_file.write(f"Precision   (route choice) : {precision:.2f}%\n")
            report_file.write(f"Completeness (intent match): {completeness:.2f}%\n")
        else:
            report_file.write("No valid data available for calculation.\n")

        # 2. Stability
        report_file.write("\n--- 2. STABILITY ---\n")
        timeout_count = len(data_frame[data_frame['Result'] == 'TIMEOUT'])
        error_count = len(data_frame[data_frame['Result'].isin(['GEO_ERR', 'NO_LLM', 'ERROR'])])
        report_file.write(f"Timeouts    : {timeout_count} ({timeout_count / len(data_frame) * 100:.1f}%)\n")
        report_file.write(f"Other errors: {error_count}\n\n")

        # 3. Per-category breakdown
        report_file.write("--- 3. DETAIL BY CATEGORY ---\n")
        if not valid_runs.empty:
            grouped = valid_runs.groupby('Category').agg(
                Precision=('Precision', 'mean'),
                Completeness=('Completeness', 'mean'),
                Avg_Time=('Duration', 'mean'),
                Count=('Run', 'count')
            )
            grouped['Precision'] = (grouped['Precision'] * 100).map("{:.1f}%".format)
            grouped['Completeness'] = (grouped['Completeness'] * 100).map("{:.1f}%".format)
            grouped['Avg_Time'] = grouped['Avg_Time'].map("{:.2f}s".format)
            report_file.write(grouped.to_string())
        else:
            report_file.write("Insufficient valid data to group by category.\n")

        report_file.write("\n\n")

        # 4. Failed runs
        report_file.write("--- 4. FAILED RUNS ---\n")
        failed_runs = data_frame[data_frame['Result'] != 'PASS']
        if not failed_runs.empty:
            cols = ['Scenario', 'Run', 'Result', 'Duration', 'Error_Hint']
            available = [c for c in cols if c in failed_runs.columns]
            report_file.write(failed_runs[available].to_string(index=False))
        else:
            report_file.write("No failures recorded.\n")

    print(f"\nSummary report saved to: {report_path}")
    data_frame.to_csv(csv_path, index=False)
    print(f"Raw CSV data saved to  : {csv_path}")


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(llm_method: str, runs_per_scenario: int, timeout_seconds: int, log_directory: str):
    full_log_path = os.path.join(
        _PROJECT_ROOT,
        f"benchmark_full_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    sys.stdout = DualLogger(full_log_path)

    print("--- STARTING VAMOS BENCHMARK ---")
    print(f"Full log  : {full_log_path}")
    print(f"Method    : {llm_method}")
    print(f"Runs/scenario: {runs_per_scenario}")
    print(f"Timeout   : {timeout_seconds}s per run")
    print(f"Scenarios : {len(SCENARIOS)}")
    print(f"Total runs: {len(SCENARIOS) * runs_per_scenario}")

    if llm_method == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"OPENAI_API_KEY found (***{api_key[-4:]})")
        else:
            print("WARNING: OPENAI_API_KEY not set in environment!")
    print("-" * 60)

    os.makedirs(log_directory, exist_ok=True)
    python_executable = sys.executable

    # Back up the original context_engine.py to restore it after the run.
    original_context = ""
    if os.path.exists(_CONTEXT_ENGINE_PATH):
        with open(_CONTEXT_ENGINE_PATH, 'r', encoding='utf-8') as f:
            original_context = f.read()

    results = []

    try:
        for run_index in range(1, runs_per_scenario + 1):
            print(f"\n>>> RUN {run_index}/{runs_per_scenario}")

            for scenario_index, scenario in enumerate(SCENARIOS):
                scenario_name = scenario['scenario_name']
                print(
                    f"  ({scenario_index + 1}/{len(SCENARIOS)}) {scenario_name}...",
                    end=" ", flush=True
                )

                _update_context_file(scenario.get("avoid_rule"))

                effective_method = scenario.get("method", llm_method)
                origin = scenario["origin"].replace(", ", ",")
                destination = scenario["destination"].replace(", ", ",")

                command = [
                    python_executable, _APP_PATH,
                    f"--origin={origin}",
                    f"--destination={destination}",
                    f"--method={effective_method}",
                ]
                if scenario.get("tasks"):
                    command.append("--tasks")
                    command.extend(scenario["tasks"])

                start_time = time.time()
                error_hint = ""
                try:
                    process = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        timeout=timeout_seconds,
                        cwd=_PROJECT_ROOT,
                    )
                    output = process.stdout + "\n" + process.stderr
                    if process.returncode != 0:
                        error_hint = f"exit_code={process.returncode}"
                except subprocess.TimeoutExpired:
                    print(f"[TIMEOUT after {timeout_seconds}s]")
                    output = "TIMEOUT"
                    error_hint = f"exceeded {timeout_seconds}s"
                except Exception as exc:
                    print(f"[ERROR: {exc}]")
                    output = str(exc)
                    error_hint = str(exc)

                duration = time.time() - start_time

                # Save per-scenario log.
                safe_name = scenario_name.replace(" ", "_").replace("/", "-")
                log_path = os.path.join(log_directory, f"run{run_index}_{safe_name}.log")
                with open(log_path, "w", encoding='utf-8') as log_file:
                    log_file.write(f"CMD: {' '.join(command)}\n\n{output}")

                llm_data = _extract_llm_json(output)
                result_label, precision_pass, completeness_pass = _classify_result(
                    output, scenario, llm_data, duration
                )

                print(f"[{result_label}] ({duration:.1f}s)")

                results.append({
                    "Scenario": scenario_name,
                    "Run": run_index,
                    "Duration": duration,
                    "Precision": 1 if precision_pass else 0,
                    "Completeness": 1 if completeness_pass else 0,
                    "Result": result_label,
                    "Chosen_ID": llm_data.get("chosen_route_id") if llm_data else None,
                    "Action": (
                        llm_data.get("required_action", {}).get("type")
                        if llm_data else None
                    ),
                    "Expected_Action": scenario.get(
                        "expected_action_type",
                        "ADD_WAYPOINT" if scenario.get("recalculation_expected", False) else "NONE",
                    ),
                    "Method": effective_method,
                    "Error_Hint": error_hint,
                })

    finally:
        if original_context:
            with open(_CONTEXT_ENGINE_PATH, 'w', encoding='utf-8') as f:
                f.write(original_context)
            print("\nContext engine restored to original.")

        _generate_report(results, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="VAMOS benchmark — reproduces the experimental results from the paper."
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_LLM_METHOD,
        choices=['hf', 'ollama', 'openai'],
        help=f"LLM backend to use (default: '{DEFAULT_LLM_METHOD}' from config.json)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of runs per scenario (default: {DEFAULT_RUNS} from config.json)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds per scenario run (default: {DEFAULT_TIMEOUT}s from config.json)."
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for per-scenario log files."
    )
    args = parser.parse_args()

    run_benchmark(
        llm_method=args.method,
        runs_per_scenario=args.runs,
        timeout_seconds=args.timeout,
        log_directory=args.log_dir,
    )


if __name__ == "__main__":
    main()
