import ollama
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import Generator, from_transformers
from pydantic import BaseModel
from typing import Literal, Dict, List, Annotated
import torch
import openai


class Task(BaseModel):
    task: str
    importance: Annotated[int, "1 (low) to 10 (high)"]
    poi_tags: Dict


class ModelOutput(BaseModel):
    tasks: List[Task]


class RequiredAction(BaseModel):
    type: Literal["ADD_WAYPOINT", "NONE"]
    description: str


class EvaluatedRoute(BaseModel):
    chosen_route_id: int
    justification: str
    required_action: RequiredAction


def _serialize_response(content) -> dict:
    """Convert an LLM response to a plain dict regardless of its type.

    Handles three cases:
    - Pydantic model (returned by the HF/outlines pipeline)
    - JSON string (returned by Ollama and OpenAI)
    - Plain dict (already parsed)
    """
    if isinstance(content, dict):
        return content
    if hasattr(content, 'model_dump'):
        return content.model_dump()
    if isinstance(content, str):
        return json.loads(content)
    raise TypeError(f"Unexpected LLM response type: {type(content)}")


class LLMAgent:
    """Routes user tasks through a language model to classify intent and evaluate routes.

    Supports three inference backends:
    - 'hf'     : local HuggingFace model loaded via transformers + outlines
    - 'ollama' : locally served model via the Ollama daemon
    - 'openai' : OpenAI ChatCompletion API
    """

    def __init__(self, model_name: str, method: str = 'hf', api_key: str = None):
        self.model_name = model_name
        self.method = method
        self.api_key = api_key
        print(f"LLM Agent initialized — model: {self.model_name}, method: {self.method}")

        self.timing_metrics = {
            'classify_tasks': [],
            'evaluate_routes': []
        }

        if method == 'hf':
            self._init_hf_model()
        elif method == 'openai':
            self._init_openai_client()

    def _init_hf_model(self):
        """Load the HuggingFace model and set up outlines generators."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=False)
            self.model.to('cuda')
        except Exception as primary_error:
            print(f"\nCould not load HF model with basic settings: {primary_error}")
            try:
                print("Retrying with accelerate offload (may use ./offload directory)...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    device_map='auto',
                    offload_folder='./offload'
                )
                self.model.to('cuda')
            except Exception as fallback_error:
                combined = f"{primary_error}\nFallback error: {fallback_error}"
                print(f"\nFailed to load HF model: {combined}")
                if 'meta' in combined or 'Tensor.item() cannot be called on meta tensors' in combined:
                    print(
                        "\nDetected meta-tensor / lazy-init issue. Possible fixes:\n"
                        " - Use CPython (not PyPy).\n"
                        " - Install/upgrade 'accelerate' and run 'accelerate config'.\n"
                        " - Try a smaller model or add RAM so the model fits in memory."
                    )
                raise RuntimeError("Could not initialize HF model. See logs above.") from fallback_error

        self.task_generator = Generator(
            from_transformers(model=self.model, tokenizer_or_processor=self.tokenizer),
            output_type=ModelOutput,
        )
        self.route_generator = Generator(
            from_transformers(model=self.model, tokenizer_or_processor=self.tokenizer),
            output_type=EvaluatedRoute,
        )

    def _init_openai_client(self):
        """Initialise the OpenAI client using the provided key or OPENAI_API_KEY env var."""
        try:
            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Pass it via the api_key parameter "
                    "or set the OPENAI_API_KEY environment variable."
                )
            self.client = openai.OpenAI(api_key=api_key)
            print("OpenAI client initialised successfully.")
        except Exception as exc:
            print(f"Error initialising OpenAI client: {exc}")
            raise

    # ------------------------------------------------------------------
    # Low-level model calls
    # ------------------------------------------------------------------

    def _call_hf(self, prompt: str, evaluate_route: bool = False):
        messages = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        generator = self.route_generator if evaluate_route else self.task_generator
        return generator(messages, max_new_tokens=2024)

    def _call_openai(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=2024,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _call_llm(self, prompt: str, evaluate_route: bool = False):
        """Dispatch the prompt to the configured backend and return the raw response."""
        if self.method == 'ollama':
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            return response['message']['content']
        if self.method == 'hf':
            return self._call_hf(prompt=prompt, evaluate_route=evaluate_route)
        if self.method == 'openai':
            return self._call_openai(prompt=prompt)
        raise ValueError(f"Unsupported method '{self.method}'. Choose 'ollama', 'hf', or 'openai'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_tasks(self, tasks: list) -> dict:
        """Classify user tasks by importance and map each to OSM POI tags.

        Returns a dict with a 'tasks' key containing a list of classified task objects.
        Returns an empty dict on failure.
        """
        if not tasks:
            return {}
        prompt = self._build_classify_prompt(tasks)
        print("\n--- Sending prompt to LLM for task classification ---")

        start = time.time()
        try:
            raw = self._call_llm(prompt=prompt)
            elapsed = time.time() - start
            self.timing_metrics['classify_tasks'].append(elapsed)
            result = _serialize_response(raw)
            print(f"--- LLM Response (Task Classification JSON) ---\n{json.dumps(result)}")
            print(f"Response time: {elapsed:.3f}s")
            return result
        except Exception as exc:
            elapsed = time.time() - start
            self.timing_metrics['classify_tasks'].append(elapsed)
            print(f"\nError during task classification: {exc}")
            return {}

    def evaluate_routes(self, user_context: dict, scenario_context: dict, routes: list) -> dict:
        """Ask the LLM to pick the best route given user context and scenario data.

        Returns a dict with 'chosen_route_id' and 'justification' keys.
        Returns an error dict on failure.
        """
        prompt = self._build_eval_prompt(user_context, scenario_context, routes)
        print("\n--- Sending prompt to LLM for route evaluation ---")

        start = time.time()
        try:
            raw = self._call_llm(prompt=prompt, evaluate_route=True)
            elapsed = time.time() - start
            self.timing_metrics['evaluate_routes'].append(elapsed)
            result = _serialize_response(raw)
            print(f"--- LLM Response (Route Evaluation JSON) ---\n{json.dumps(result)}")
            print(f"Response time: {elapsed:.3f}s")
            return result
        except Exception as exc:
            elapsed = time.time() - start
            self.timing_metrics['evaluate_routes'].append(elapsed)
            print(f"\nError during route evaluation: {exc}")
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_classify_prompt(self, tasks: list) -> str:
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

        Examples of tasks and their classifications:
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

        Analyze the following list of user tasks and return ONLY a valid JSON object structured as shown above.
        Tasks: {json.dumps(tasks)}
        """

    def _build_eval_prompt(self, user_context: dict, scenario_context: dict, routes: list) -> str:
        formatted_routes = []
        for index, route in enumerate(routes):
            formatted_routes.append({
                "route_id": index + 1,
                "description": route.get("description", "Direct Route"),
                "total_estimated_minutes": round(route['cost_seconds'] / 60, 2),
                "tasks_completed": route.get('tasks_completed', [])
            })

        return f"""
        You are an intelligent vehicle navigation assistant. Choose the best route from the options below.

        Decision hierarchy (follow strictly):
        1. **URGENT tasks (importance 10)**: Always choose the route that completes them, unless the detour
           more than doubles the direct route time. This involves life or safety.
        2. **NECESSARY tasks (importance 6-9)**: Strongly prefer routes that complete them; a reasonable
           detour is acceptable given traffic and weather.
        3. **CONVENIENCE tasks (importance 1-5)**: Choose the detour only if extra time is minimal and
           traffic is good; otherwise prefer efficiency.
        4. **No tasks**: Choose the fastest route.

        User context:
        {json.dumps(user_context, indent=2)}

        Scenario context:
        {json.dumps(scenario_context, indent=2)}

        Route options:
        {json.dumps(formatted_routes, indent=2)}

        Respond ONLY with a single valid JSON object. Example:
        {{
            "chosen_route_id": 2,
            "justification": "Route 2 completes an urgent task with an acceptable 8-minute detour.",
            "required_action": {{"type": "ADD_WAYPOINT", "description": "Stop at the fuel station."}}
        }}

        Now provide your analysis.
        """

    # ------------------------------------------------------------------
    # Timing report
    # ------------------------------------------------------------------

    def get_timing_report(self) -> dict:
        """Return a summary of per-call timing metrics for all LLM operations."""
        report = {'method': self.method, 'model': self.model_name, 'metrics': {}}
        for operation, timings in self.timing_metrics.items():
            if timings:
                report['metrics'][operation] = {
                    'count': len(timings),
                    'total_seconds': sum(timings),
                    'avg_seconds': sum(timings) / len(timings),
                    'min_seconds': min(timings),
                    'max_seconds': max(timings),
                }
            else:
                report['metrics'][operation] = {
                    'count': 0, 'total_seconds': 0,
                    'avg_seconds': 0, 'min_seconds': 0, 'max_seconds': 0,
                }
        return report

    def print_timing_report(self):
        """Print a formatted timing report to stdout."""
        report = self.get_timing_report()
        print("\n" + "=" * 60)
        print("LLM TIMING REPORT")
        print("=" * 60)
        print(f"Method : {report['method'].upper()}")
        print(f"Model  : {report['model']}")
        print("-" * 60)
        for operation, metrics in report['metrics'].items():
            if metrics['count'] > 0:
                print(f"\n{operation.upper().replace('_', ' ')}:")
                print(f"  Calls  : {metrics['count']}")
                print(f"  Total  : {metrics['total_seconds']:.3f}s")
                print(f"  Avg    : {metrics['avg_seconds']:.3f}s")
                print(f"  Min    : {metrics['min_seconds']:.3f}s")
                print(f"  Max    : {metrics['max_seconds']:.3f}s")
        total_time = sum(sum(t) for t in self.timing_metrics.values())
        total_calls = sum(len(t) for t in self.timing_metrics.values())
        if total_calls > 0:
            print(f"\nSUMMARY:")
            print(f"  Total calls : {total_calls}")
            print(f"  Total time  : {total_time:.3f}s")
            print(f"  Avg/call    : {total_time / total_calls:.3f}s")
        print("=" * 60 + "\n")
