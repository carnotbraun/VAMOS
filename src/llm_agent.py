import ollama
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import Generator, from_transformers
from pydantic import BaseModel
from typing import Literal, Dict, List, Annotated
import torch
from accelerate import disk_offload
import openai

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Task(BaseModel):
    task: str
    importance: Annotated[int, "1 (low) to 10 (high)"]
    poi_tags: Dict

class ModelOutput(BaseModel):
    tasks: List[Task]

class RequireAction(BaseModel):
    type: Literal["ADD_WAYPOINT", "NONE"]
    description: str

class EvalRoute(BaseModel):
    chosen_route_id: int
    justification: str
    required_action: RequireAction

class LLMAgent:
    """
    Interage com um modelo de linguagem local via Ollama para classificar tarefas,
    extrair tags OSM e avaliar rotas com regras complexas.
    """
    def __init__(self, model_name: str, method: str = 'ollama', api_key: str = None):
        self.model_name = model_name
        self.method = method
        self.api_key = api_key # Aqui você inseriria sua chave de API do OpenAI se for usar esse método
        print(f"LLM Agent initialized with model: {self.model_name} and the method: {self.method}")
        
        # Dicionário para armazenar medições de tempo
        self.timing_metrics = {
            'classify_tasks': [],
            'extract_poi_tags': [],
            'evaluate_routes': []
        }

        if method == 'hf':
            self.create_hf_model()
        elif method == 'openai':
            self.create_openai_client()

    def create_hf_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=False)
            self.model.to('cuda')
        except Exception as e:
            msg = str(e)
            print(f"\nError creating HF model (basic load): {msg}")
            try:
                print("Attempting fallback load using accelerate-friendly parameters (may offload to ./offload)...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    device_map='auto',
                    offload_folder='./offload'
                )
                # ensure model modules are accessible on CPU when needed
                self.model.to('cuda')
            except Exception as e2:
                # Detect common meta-tensor error and provide guidance
                full_err = f"{e}\nFallback error: {e2}"
                print(f"\nFailed to load HF model using fallback: {full_err}")
                if 'meta' in full_err or 'Tensor.item() cannot be called on meta tensors' in full_err:
                    print("\nDetected 'meta' tensor / lazy-initialization problem. Common remedies:")
                    print(" - Run with CPython (not PyPy). Some HF packages and accelerate features require CPython.")
                    print(" - Install/upgrade 'accelerate' and configure with 'accelerate config' if you want device_map=auto/offload.")
                    print(" - Avoid disk/offload layers: try a smaller model or increase available RAM so the model can be loaded fully on CPU.")
                raise RuntimeError("Could not initialize HF model. See above logs for hints.") from e2

        
        self.generator = Generator(
            from_transformers(
                model = self.model,
                tokenizer_or_processor = self.tokenizer
            ),
            output_type = ModelOutput,
        )
        self.eval_generator = Generator(
            from_transformers(
                model = self.model,
                tokenizer_or_processor = self.tokenizer
            ),
            output_type = EvalRoute
        )

    def create_openai_client(self):
        """Inicializa o cliente OpenAI"""
        try:
            if self.api_key:
                api_key = self.api_key
            else:
                # Tenta usar a variável de ambiente
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("API key do OpenAI não fornecida. Use o parâmetro api_key ou defina OPENAI_API_KEY como variável de ambiente.")
            
            self.client = openai.OpenAI(api_key=api_key)
            print("Cliente OpenAI inicializado com sucesso")
        except Exception as e:
            print(f"Erro ao inicializar cliente OpenAI: {e}")
            raise

    def call_hf_model(self, prompt: str, eval: bool = False):
        messages = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if eval:
            response = self.eval_generator(messages, max_new_tokens=2024)
            return response
        response = self.generator(messages, max_new_tokens = 2024)
        return response

    def call_openai_model(self, prompt: str, eval: bool = False):
        """Chama a API do OpenAI ChatGPT"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=2024,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Erro ao chamar API do OpenAI: {e}")
            raise

    def call_llm(self, prompt: str, eval: bool = False):
        if self.method == 'ollama':
            response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], format='json')
            content = response['message']['content']
            return content
        elif self.method == 'hf': # hugginface
            return self.call_hf_model(prompt = prompt, eval = eval)
        elif self.method == 'openai': # OpenAI ChatGPT
            return self.call_openai_model(prompt = prompt, eval = eval)
        else:
            raise ValueError(f"Método '{self.method}' não suportado. Use 'ollama', 'hf' ou 'openai'.")
    
    def classify_tasks(self, tasks: list[str]) -> list[dict]:
        if not tasks: return []
        prompt = self._build_classify_prompt(tasks)
        print("\n--- Sending prompt to LLM for task classification ---")
        
        # Inicia medição de tempo
        start_time = time.time()
        try:
            content = self.call_llm(prompt = prompt)
            elapsed_time = time.time() - start_time
            self.timing_metrics['classify_tasks'].append(elapsed_time)
            print(f"--- LLM Response (Task Classification JSON) ---\n{content}\n{type(content)}")
            print(f"⏱️  Response time: {elapsed_time:.3f}s")
            return json.loads(content)
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.timing_metrics['classify_tasks'].append(elapsed_time)
            print(f"\nError during task classification: {e}")
            return []

    def extract_poi_tags_from_tasks(self, tasks: list[str]) -> dict:
        if not tasks: return {}
        prompt = self._build_tags_prompt(tasks)
        print("\n--- Sending prompt to LLM for OSM tag extraction ---")
        
        # Inicia medição de tempo
        start_time = time.time()
        try:
            content = self.call_llm(prompt = prompt)
            elapsed_time = time.time() - start_time
            self.timing_metrics['extract_poi_tags'].append(elapsed_time)
            print(f"--- LLM Response (OSM Tags JSON) ---\n{content}")
            print(f"⏱️  Response time: {elapsed_time:.3f}s")
            return json.loads(content)
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.timing_metrics['extract_poi_tags'].append(elapsed_time)
            print(f"\nError during OSM tag extraction: {e}")
            return {}

    def evaluate_routes(self, user_context: dict, scenario_context: dict, routes: list) -> dict:
        prompt = self._build_eval_prompt(user_context, scenario_context, routes)
        print("\n--- Sending prompt to LLM for route evaluation ---")
        
        # Inicia medição de tempo
        start_time = time.time()
        try:
            content = self.call_llm(prompt = prompt, eval = True)
            elapsed_time = time.time() - start_time
            self.timing_metrics['evaluate_routes'].append(elapsed_time)
            print(f"--- LLM Response (Route Evaluation JSON) ---\n{content}")
            print(f"⏱️  Response time: {elapsed_time:.3f}s")
            return json.loads(content)
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.timing_metrics['evaluate_routes'].append(elapsed_time)
            print(f"\nError communicating with Ollama: {e}")
            return {"error": str(e)}

    def _build_classify_prompt(self, tasks: list[str]) -> str:
        return f"""
        You are a personal logistics analysis expert. Your job is to analyze a list of user tasks, 
        identify the necessary Points of Interest (POIs) using OpenStreetMap (OSM) tags, and rank the 
        importance of each task on a scale from 1 (low) to 10 (high).

        Instructions:
        1.  **URGENCY**: Tasks that involve emergencies (health, safety, vehicle) must have an importance of 10.
            Ex: "I need to go to the hospital", "my car is out of gas".
        2.  **NECESSITY**: Essential but non-emergency tasks (buying food, medicine) must have an importance between 6 and 9.
        3.  **CONVENIENCE**: Leisure or non-essential tasks (going to the park, café) must have an importance between 1 and 5.
        4.  **OSM TAGS**: Use the most common and specific tags. For "buy food", use `shop:supermarket`. For "medicine", `amenity:pharmacy`.

        Exples of Tasks and their classifications:
        - Task: ["need gas urgently and also i want to stop by the park"]
            JSON: {{"tasks": [
                {{"task": "need gas urgently", "importance": 10, "poi_tags": {{"amenity": "fuel"}}}},
                {{"task": "i want to stop by the park", "importance": 2, "poi_tags": {{"leisure": "park"}}}}
            ]}}
        - Task: ["buy some bread and go to the pharmacy"]
            JSON: {{"tasks": [
                {{"task": "buy some bread", "importance": 7, "poi_tags": {{"shop": "bakery"}}}},
                {{"task": "go to the pharmacy", "importance": 8, "poi_tags": {{"amenity": "pharmacy"}}}}
            ]}}
        Your task:
        Analyze the following list of user tasks and return ONLY a valid JSON object structured as follows:
        Tasks: {json.dumps(tasks)}
        """

    def _build_tags_prompt(self, tasks: list[str]) -> str:
        return f"""
        You are an expert in OpenStreetMap (OSM) data. Your task is to analyze a list of user tasks and convert them into a JSON object of OSM tags suitable for a search query.
        Examples:
        - ["go to the pharmacy"] -> {{"amenity": "pharmacy"}}
        - ["pick up groceries", "go to the supermarket"] -> {{"shop": "supermarket"}}
        - ["take the kids to the park"] -> {{"leisure": "park"}}
        - ["go to the hospital"] -> {{"amenity": "hospital"}}
        - ["I need to buy some food"] -> {{"shop": "supermarket"}}
        Analyze the following user tasks and provide the corresponding JSON object of OSM tags.
        User Tasks: {json.dumps(tasks)}
        Respond ONLY with the final JSON object.
        """

    def _build_eval_prompt(self, user_context, scenario_context, routes) -> str:
        formatted_routes = []
        for i, route in enumerate(routes):
            route_info = {
                "route_id": i + 1, "description": route.get("description", "Direct Route"),
                "total_estimated_minutes": round(route['cost_seconds'] / 60, 2),
                "tasks_completed": route.get('tasks_completed', [])
            }
            formatted_routes.append(route_info)

        return f"""
        You are an intelligent vehicle navigation assistant. Your task is to choose the best route from the given options and provide a justification.
        
        You must follow these rules strictly:
        **Decision Hierarchy (VERY IMPORTANT):**
        1.  **HIGHEST PRIORITY - URGENT TASKS**: If a route completes a task with **`"importance": 10`**, you **MUST** choose it, unless the detour time is extremely excessive (e.g., more than double the direct route time). This is a critical life-or-safety task.
        2.  **SECOND PRIORITY - NECESSARY TASKS**: If a route completes a task with importance 6-9, you should strongly prefer it over the direct route, considering traffic and weather context. A reasonable detour is acceptable.
        3.  **LOWEST PRIORITY - CONVENIENCE TASKS**: For tasks with importance 1-5, choose the route with the detour only if the extra time is minimal and traffic is good. Otherwise, prefer efficiency.
        4.  **EFFICIENCY**: If no tasks are involved, or if the detours are not justified by the rules above, choose the fastest route.

        **User Context:**
        {json.dumps(user_context, indent=2)}

        **Scenario Context:**
        {json.dumps(scenario_context, indent=2)}

        **Route Options:**
        {json.dumps(formatted_routes, indent=2)}
        
        **Your Response:**
        Analyze all data following the decision hierarchy strictly. Respond ONLY with a single, valid JSON object containing "chosen_route_id" and "justification".

        Example of a valid response for a non-urgent task:
        {{
            "chosen_route_id": 2,
            "justification": "Route 2 was chosen because it completes a necessary task (importance 8) with a reasonable detour time of 15 minutes, which is acceptable given the current light traffic conditions."
        }}

        Now, provide your analysis for the current routes.
        """

    def get_timing_report(self) -> dict:
        """Retorna um relatório de medições de tempo para todas as chamadas de LLM."""
        report = {
            'method': self.method,
            'model': self.model_name,
            'metrics': {}
        }
        
        for task_name, timings in self.timing_metrics.items():
            if timings:
                report['metrics'][task_name] = {
                    'count': len(timings),
                    'total_seconds': sum(timings),
                    'avg_seconds': sum(timings) / len(timings),
                    'min_seconds': min(timings),
                    'max_seconds': max(timings),
                }
            else:
                report['metrics'][task_name] = {
                    'count': 0,
                    'total_seconds': 0,
                    'avg_seconds': 0,
                    'min_seconds': 0,
                    'max_seconds': 0,
                }
        
        return report
    
    def print_timing_report(self):
        """Imprime um relatório formatado de medições de tempo."""
        report = self.get_timing_report()
        
        print("\n" + "="*80)
        print("⏱️  LLM TIMING REPORT")
        print("="*80)
        print(f"Method: {report['method'].upper()}")
        print(f"Model: {report['model']}")
        print("-"*80)
        
        for task_name, metrics in report['metrics'].items():
            if metrics['count'] > 0:
                print(f"\n📊 {task_name.upper().replace('_', ' ')}:")
                print(f"   Calls: {metrics['count']}")
                print(f"   Total time: {metrics['total_seconds']:.3f}s")
                print(f"   Average: {metrics['avg_seconds']:.3f}s")
                print(f"   Min: {metrics['min_seconds']:.3f}s")
                print(f"   Max: {metrics['max_seconds']:.3f}s")
        
        # Total de todas as chamadas
        total_time = sum(
            sum(timings) for timings in self.timing_metrics.values()
        )
        total_calls = sum(
            len(timings) for timings in self.timing_metrics.values()
        )
        
        if total_calls > 0:
            print(f"\n📈 SUMMARY:")
            print(f"   Total API calls: {total_calls}")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average per call: {total_time/total_calls:.3f}s")
        
        print("="*80 + "\n")
