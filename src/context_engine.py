from datetime import datetime


class ContextEngine:
    """Provides user preferences and real-time scenario context for route personalisation.

    In a production deployment, these values would be fetched from live data sources
    (traffic APIs, user profile services, weather APIs). The current implementation
    returns static defaults suitable for benchmarking and demonstration purposes.
    """

    def get_user_context(self) -> dict:
        """Return user preference data used by the LLM to personalise route selection."""
        return {
            "preferences": [
                "avoid downtown during rush hour",
                "prefers safer routes at night"
            ],
            "avoidance_rules": []
        }

    def get_scenario_context(self, origin: tuple = None, destination: tuple = None) -> dict:
        """Return current scenario data (time, traffic, weather) for the given trip.

        Args:
            origin: (lat, lon) of the departure point (unused in default implementation).
            destination: (lat, lon) of the arrival point (unused in default implementation).
        """
        return {
            "current_time": datetime.now().strftime("%H:%M"),
            "day_of_week": datetime.now().strftime("%A"),
            "traffic_conditions": "moderate",
            "weather": "cloudy"
        }
