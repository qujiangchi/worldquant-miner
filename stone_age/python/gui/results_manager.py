import json
import os
from datetime import datetime
from typing import Dict, Any

class ResultsManager:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_result(self, alpha_data: Dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alpha_result_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(alpha_data, f, indent=2)
        
        return filepath

    def load_result(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)

    def list_results(self) -> list:
        return [f for f in os.listdir(self.results_dir) if f.endswith('.json')]

    def get_latest_result(self) -> Dict[str, Any]:
        results = self.list_results()
        if not results:
            return None
        
        latest = max(results, key=lambda x: os.path.getctime(os.path.join(self.results_dir, x)))
        return self.load_result(latest) 