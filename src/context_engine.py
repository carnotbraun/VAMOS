from datetime import datetime

class ContextEngine:
    """
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
