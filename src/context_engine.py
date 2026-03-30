from datetime import datetime

class ContextEngine:
    """
    Contexto do usuário e do cenário para personalizar as rotas. O contexto do usuário inclui preferências e regras de evitação, enquanto o contexto do cenário inclui informações em tempo real como condições de tráfego e clima.
    """
    def get_user_context(self) -> dict:
        return {
            "preferences": ["avoid downtown during rush hour", "prefers safer routes at night"],
            "avoidance_rules": [] 
        }

    def get_scenario_context(self, origin=None, dest=None) -> dict:
        return {
            "current_time": datetime.now().strftime("%H:%M"),
            "day_of_week": "Tuesday",
            "traffic_conditions": "moderate",
            "weather": "cloudy"
        }
