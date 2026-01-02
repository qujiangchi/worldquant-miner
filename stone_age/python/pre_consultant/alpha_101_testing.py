import argparse
import requests
import json
import os
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict
import time
import logging
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Configure logger
logger = logging.getLogger(__name__)

# Alpha 101 formulas
ALPHA_101_FORMULAS = [
    # Original formulas adapted for WorldQuant Brain syntax
    "rank(ts_argmax(power(where(returns < 0, ts_std_dev(returns, 20), close), 2), 5)) - 0.5",
    "-1 * correlation(rank(ts_delta(log(volume), 2)), rank((close - open) / open), 6)",
    "-1 * correlation(rank(open), rank(volume), 10)",
    "-1 * ts_rank(rank(low), 9)",
    "rank(open - (ts_mean(vwap, 10))) * (-1 * abs(rank(close - vwap)))",
    "-1 * correlation(open, volume, 10)",
    "where(ts_mean(volume, 20) < volume, -1 * ts_rank(abs(ts_delta(close, 7)), 60) * sign(ts_delta(close, 7)), -1)",
    "-1 * rank((ts_sum(open, 5) * ts_sum(returns, 5)) - ts_delay((ts_sum(open, 5) * ts_sum(returns, 5)), 10))",
    "where(0 < ts_min(ts_delta(close, 1), 5), ts_delta(close, 1), where(ts_max(ts_delta(close, 1), 5) < 0, ts_delta(close, 1), -1 * ts_delta(close, 1)))",
    "rank(where(0 < ts_min(ts_delta(close, 1), 4), ts_delta(close, 1), where(ts_max(ts_delta(close, 1), 4) < 0, ts_delta(close, 1), -1 * ts_delta(close, 1))))",
    "(rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(ts_delta(volume, 3))",
    "sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))",
    "-1 * rank(covariance(rank(close), rank(volume), 5))",
    "-1 * rank(ts_delta(returns, 3)) * correlation(open, volume, 10)",
    "-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)",
    "-1 * rank(covariance(rank(high), rank(volume), 5))",
    "-1 * rank(ts_rank(close, 10)) * rank(ts_delta(ts_delta(close, 1), 1)) * rank(ts_rank(volume / ts_mean(volume, 20), 5))",
    "-1 * rank(ts_std_dev(abs(close - open), 5) + (close - open) + correlation(close, open, 10))",
    "-1 * sign(ts_delta(close, 7) + ts_delta(close, 7)) * (1 + rank(1 + ts_sum(returns, 250)))",
    "-1 * rank(open - ts_delay(high, 1)) * rank(open - ts_delay(close, 1)) * rank(open - ts_delay(low, 1))",
    
    # Next batch (21-40)
    "where((ts_mean(close, 8) / 8 + ts_std_dev(close, 8)) < (ts_mean(close, 2) / 2), -1, where((ts_mean(close, 2) / 2) < ((ts_mean(close, 8) / 8) - ts_std_dev(close, 8)), 1, where(1 < (volume / ts_mean(volume, 20)), 1, -1)))",
    "-1 * (ts_delta(correlation(high, volume, 5), 5) * rank(ts_std_dev(close, 20)))",
    "where((ts_mean(high, 20) / 20) < high, -1 * ts_delta(high, 2), 0)",
    "where((ts_delta(ts_mean(close, 100) / 100, 100) / ts_delay(close, 100)) < 0.05, -1 * (close - ts_min(close, 100)), -1 * ts_delta(close, 3))",
    "rank(((-1 * returns) * ts_mean(volume, 20) * vwap * (high - close)))",
    "-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    "where(0.5 < rank((ts_sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0)), -1, 1)",
    "scale(((correlation(ts_mean(volume, 20), low, 5) + ((high + low) / 2)) - close))",
    "(min(product(rank(rank(scale(log(ts_sum(ts_min(rank(rank(-1 * rank(ts_delta(close - 1, 5)))), 2), 1)))), 1), 5) + ts_rank(ts_delay(-1 * returns, 6), 5))",
    "(1.0 - rank(((sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2))) + sign(ts_delay(close, 2) - ts_delay(close, 3))))) * ts_sum(volume, 5) / ts_sum(volume, 20)",
    "(rank(rank(rank(decay_linear(-1 * rank(rank(ts_delta(close, 10))), 10)))) + rank(-1 * ts_delta(close, 3))) + sign(scale(correlation(ts_mean(volume, 20), low, 12)))",
    "scale((ts_mean(close, 7) / 7 - close)) + (20 * scale(correlation(vwap, ts_delay(close, 5), 230)))",
    "rank(-1 * (1 - (open / close)))",
    "rank((1 - rank(ts_std_dev(returns, 2) / ts_std_dev(returns, 5))) + (1 - rank(ts_delta(close, 1))))",
    "(ts_rank(volume, 32) * (1 - ts_rank((close + high - low), 16))) * (1 - ts_rank(returns, 32))",
    "(((2.21 * rank(correlation((close - open), ts_delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(ts_rank(ts_delay(-1 * returns, 6), 5)))) + rank(abs(correlation(vwap, ts_mean(volume, 20), 6))) + (0.6 * rank((ts_mean(close, 200) / 200 - open) * (close - open)))",
    "rank(correlation(ts_delay(open - close, 1), close, 200)) + rank(open - close)",
    "-1 * rank(ts_rank(close, 10)) * rank(close / open)",
    "-1 * rank(ts_delta(close, 7) * (1 - rank(decay_linear(volume / ts_mean(volume, 20), 9))))) * (1 + rank(ts_sum(returns, 250)))",
    "-1 * rank(ts_std_dev(high, 10)) * correlation(high, volume, 10)",

    # Next batch (41-60)
    "power((high * low), 0.5) - vwap",
    "rank((vwap - close)) / rank((vwap + close))",
    "ts_rank((volume / ts_mean(volume, 20)), 20) * ts_rank((-1 * ts_delta(close, 7)), 8)",
    "-1 * correlation(high, rank(volume), 5)",
    "-1 * ((rank(ts_mean(ts_delay(close, 5), 20) / 20) * correlation(close, volume, 2)) * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))",
    "where(0.25 < (ts_delta(ts_delay(close, 20), 10) / 10 - ts_delta(ts_delay(close, 10), 10) / 10), -1, where((ts_delta(ts_delay(close, 20), 10) / 10 - ts_delta(ts_delay(close, 10), 10) / 10) < 0, 1, -1 * ts_delta(close, 1)))",
    "((rank(1 / close) * volume / ts_mean(volume, 20)) * ((high * rank(high - close)) / (ts_mean(high, 5) / 5))) - rank(ts_delta(vwap, 5))",
    "(correlation(ts_delta(close, 1), ts_delta(ts_delay(close, 1), 1), 250) * ts_delta(close, 1)) / close",
    "where((ts_delta(ts_delay(close, 20), 10) / 10 - ts_delta(ts_delay(close, 10), 10) / 10) < -0.1, 1, -1 * ts_delta(close, 1))",
    "-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)",

    # Batch (61-80)
    "where((ts_delta(ts_delay(close, 20), 10) / 10 - ts_delta(ts_delay(close, 10), 10) / 10) < -0.05, 1, -1 * ts_delta(close, 1))",
    "-1 * ((ts_min(low, 5) - ts_delay(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220)) * ts_rank(volume, 5)",
    "-1 * ts_delta((((close - low) - (high - close)) / (close - low)), 9)",
    "-1 * ((low - close) * power(open, 5)) / ((low - high) * power(close, 5))",
    "-1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)",
    "-1 * (rank((ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3))) * rank(returns))",
    "-1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))",
    "-1 * ts_rank(decay_linear(correlation(vwap, volume, 4), 8), 6)",
    "-1 * ts_rank(decay_linear(correlation(((vwap * 0.728317) + (vwap * (1 - 0.728317))), volume, 4), 16), 8)",
    "-1 * (2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume)) - scale(rank(ts_argmax(close, 10))))",
    "where(rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, ts_mean(volume, 180), 18)), 1, -1)",
    "where(rank(correlation(vwap, ts_sum(ts_mean(volume, 20), 22), 10)) < rank(((rank(open) + rank(open)) < (rank((high + low) / 2) + rank(high)))), 1, -1)",
    "-1 * (rank(decay_linear(ts_delta(close, 2), 8)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), ts_sum(ts_mean(volume, 180), 37), 14), 12)))",
    "-1 * where(rank(correlation(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), ts_sum(ts_mean(volume, 120), 13), 17)) < rank(ts_delta(((high + low) / 2 * 0.178404) + (vwap * (1 - 0.178404)), 4)), 1, -1)",
    "-1 * where(rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(ts_mean(volume, 60), 9), 6)) < rank(open - ts_min(open, 14)), 1, -1)",
    "-1 * (rank(decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(decay_linear(((low * 0.96633) + (low * (1 - 0.96633)) - vwap) / (open - (high + low) / 2), 11), 7))",
    "-1 * power(rank(high - ts_min(high, 2)), rank(correlation(vwap, ts_mean(volume, 120), 6)))",
    "-1 * where(ts_rank(correlation(rank(high), rank(ts_mean(volume, 15)), 9), 14) < rank(ts_delta((close * 0.518371) + (low * (1 - 0.518371)), 1)), 1, -1)",
    "-1 * power(rank(ts_max(ts_delta(vwap, 3), 5)), ts_rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), ts_mean(volume, 20), 5), 9))",
    "-1 * power(rank(ts_delta(vwap, 1)), ts_rank(correlation(close, ts_mean(volume, 50), 18), 18))",

    # Batch (81-101)
    "max(ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), ts_rank(decay_linear(rank((low + open) - (vwap + vwap)), 16), 4))",
    "rank(decay_linear(correlation((high + low) / 2, ts_mean(volume, 40), 9), 10)) / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))",
    "max(rank(decay_linear(ts_delta(vwap, 5), 3)), ts_rank(decay_linear(((ts_delta(((open * 0.147155) + (low * (1 - 0.147155))), 2) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3), 17)) * -1",
    "-1 * where(rank(correlation(close, ts_sum(ts_mean(volume, 30), 37), 15)) < rank(correlation(rank((high * 0.0261661) + (vwap * (1 - 0.0261661))), rank(volume), 11)), 1, -1)",
    "where(rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(ts_mean(volume, 50)), 12)), 1, -1)",
    "-1 * max(rank(decay_linear(ts_delta(vwap, 1), 12)), ts_rank(decay_linear(ts_rank(correlation(low, ts_mean(volume, 81), 8), 20), 17), 19))",
    "min(rank(decay_linear(((high + low) / 2 + high - (vwap + high)), 20)), rank(decay_linear(correlation((high + low) / 2, ts_mean(volume, 40), 3), 6)))",
    "power(rank(correlation(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 20), ts_sum(ts_mean(volume, 40), 20), 7)), rank(correlation(rank(vwap), rank(volume), 6)))",
    "where(rank(ts_delta(((close * 0.60733) + (open * (1 - 0.60733))), 1)) < rank(correlation(ts_rank(vwap, 4), ts_rank(ts_mean(volume, 150), 9), 15)), 1, -1)",
    "-1 * power(rank(sign(ts_delta(((open * 0.868128) + (high * (1 - 0.868128))), 4))), ts_rank(correlation(high, ts_mean(volume, 10), 5), 6))",
    "-1 * where(rank(log(product(rank(correlation(vwap, ts_sum(ts_mean(volume, 10), 50), 8)), 15))) < rank(correlation(rank(vwap), rank(volume), 5)), 1, -1)",
    "-1 * min(rank(decay_linear(ts_delta(open, 1), 15)), ts_rank(decay_linear(correlation(volume, ((open * 0.634196) + (open * (1 - 0.634196))), 17), 7), 13))",
    "((rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume))) / ((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))",
    "power(ts_rank((vwap - ts_max(vwap, 15)), 21), ts_delta(close, 5))",
    "power(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), ts_mean(volume, 30), 10)), rank(correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 7), 7)))",
    "-1 * where(ts_rank(correlation(close, ts_sum(ts_mean(volume, 20), 15), 6), 20) < rank((open + close) - (vwap + open)), 1, -1)",
    "-1 * max(rank(decay_linear(ts_delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 2), 3)), ts_rank(decay_linear(abs(correlation(ts_mean(volume, 81), close, 13)), 5), 14))",
    "min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8)), ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))",
    "(ts_rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), ts_mean(volume, 10), 7), 6), 4) - ts_rank(decay_linear(ts_delta(vwap, 3), 10), 15))",
    "-1 * power(rank(close - ts_max(close, 5)), ts_rank(correlation(ts_mean(volume, 40), low, 5), 3))",
    "-1 * (ts_rank(decay_linear(decay_linear(correlation(close, volume, 10), 16), 4), 5) - rank(decay_linear(correlation(vwap, ts_mean(volume, 30), 4), 3)))",
    "min(ts_rank(decay_linear(((high + low) / 2 + close < (low + open)), 15), 19), ts_rank(decay_linear(correlation(rank(low), rank(ts_mean(volume, 30)), 8), 7), 7))",
    "(ts_rank(decay_linear(correlation(vwap, ts_mean(volume, 81), 17), 20), 8) / rank(decay_linear(ts_delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 3), 16)))",
    "-1 * power(rank(vwap - ts_min(vwap, 12)), ts_rank(correlation(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3))",
    "where(rank(open - ts_min(open, 12)) < ts_rank(rank(correlation(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)), 12), 1, -1)",
    "-1 * max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 13), 14), 13))",
    "-1 * (rank(decay_linear(ts_delta(((low * 0.721001) + (vwap * (1 - 0.721001))), 3), 20)) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 8), ts_rank(ts_mean(volume, 60), 17), 5), 19), 7))",
    "(rank(decay_linear(correlation(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(ts_mean(volume, 15)), 21), 9), 8)))",
    "-1 * where(rank(correlation(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)) < rank(correlation(low, volume, 6)), 1, -1)",
    "-1 * (1.5 * scale(rank(((close - low) - (high - close)) / (high - low) * volume)) - scale(rank(ts_argmin(close, 30))))",
    "divide(subtract(close, open), add(subtract(high, low), 0.001))"
]

class RetryQueue:
    def __init__(self, generator, max_retries=3, retry_delay=60):
        self.queue = Queue()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.generator = generator
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
    
    def add(self, alpha: str, retry_count: int = 0):
        self.queue.put((alpha, retry_count))
    
    def _process_queue(self):
        while True:
            if not self.queue.empty():
                alpha, retry_count = self.queue.get()
                if retry_count >= self.max_retries:
                    logging.error(f"Max retries exceeded for alpha: {alpha}")
                    continue
                    
                try:
                    result = self.generator._test_alpha_impl(alpha)
                    if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                        logging.info(f"Simulation limit exceeded, requeueing alpha: {alpha}")
                        time.sleep(self.retry_delay)
                        self.add(alpha, retry_count + 1)
                    else:
                        self.generator.results.append({
                            "alpha": alpha,
                            "result": result
                        })
                except Exception as e:
                    logging.error(f"Error processing alpha: {str(e)}")
                    
            time.sleep(1)

class Alpha101Tester:
    def __init__(self, credentials_path: str):
        self.sess = requests.Session()
        self.setup_auth(credentials_path)
        self.results = []
        self.pending_results = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.retry_queue = RetryQueue(self)
    
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        with open(credentials_path) as f:
            credentials = json.load(f)
        
        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)
        
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        if response.status_code != 201:
            raise Exception(f"Authentication failed: {response.text}")
        logger.info("Successfully authenticated with WorldQuant Brain")

    def _test_alpha_impl(self, expression: str) -> Dict:
        """Submit alpha for testing."""
        url = "https://api.worldquantbrain.com/simulations"
        data = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "decay": 0,
                "neutralization": "INDUSTRY",
                "truncation": 0.08,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False
            },
            "regular": expression
        }
        
        try:
            logger.info(f"Submitting alpha with expression: {expression}")
            response = self.sess.post(url, json=data)
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 201:
                sim_id = response.headers.get('location', '').split('/')[-1]
                if sim_id:
                    logger.info(f"Simulation created with ID: {sim_id}")
                    return {"status": "success", "result": {"id": sim_id}}
                else:
                    return {"status": "error", "message": "No simulation ID in response headers"}
            elif response.status_code == 400:
                error_data = response.json()
                logger.error(f"API error details: {error_data}")
                return {"status": "error", "message": error_data.get("error", {}).get("message", "Unknown error")}
            elif response.status_code == 429:
                return {"status": "error", "message": "SIMULATION_LIMIT_EXCEEDED"}
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error testing alpha {expression}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def monitor_simulation(self, sim_id: str, alpha: str) -> Dict:
        """Monitor simulation progress and get final results."""
        while True:
            try:
                # Check simulation progress
                sim_response = self.sess.get(f"https://api.worldquantbrain.com/simulations/{sim_id}")
                if not sim_response.text.strip():
                    logger.info("Simulation still initializing, waiting...")
                    time.sleep(10)
                    continue

                sim_data = sim_response.json()
                
                # If still in progress
                if "progress" in sim_data:
                    progress = sim_data.get("progress", 0)
                    logger.info(f"Simulation progress: {progress:.2%}")
                    time.sleep(10)
                    continue
                
                # If completed, check alpha details
                if sim_data.get("status") == "COMPLETE" and sim_data.get("alpha"):
                    alpha_id = sim_data["alpha"]
                    logger.info(f"Simulation complete, checking alpha {alpha_id}")
                    
                    alpha_response = self.sess.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}")
                    if alpha_response.status_code == 200:
                        alpha_data = alpha_response.json()
                        
                        # Combine simulation and alpha data
                        result = {
                            "simulation": sim_data,
                            "alpha": alpha_data,
                            "expression": alpha
                        }
                        
                        # Log if promising
                        if self.is_promising_alpha(alpha_data):
                            self.log_hopeful_alpha(alpha, result)
                            logger.info(f"Logged hopeful alpha: {alpha_id}")
                        
                        return result
                        
                elif sim_data.get("status") in ["FAILED", "ERROR"]:
                    logger.error(f"Simulation {sim_data.get('status')}: {sim_data}")
                    return None
                
            except Exception as e:
                logger.error(f"Error monitoring simulation {sim_id}: {str(e)}")
                time.sleep(10)
                continue

    def is_promising_alpha(self, data: Dict) -> bool:
        """Check if alpha meets performance criteria."""
        metrics = data.get("metrics", {})
        return (
            metrics.get("fitness", 0) > 0.5 and
            metrics.get("sharpe", 0) > 0.5 and
            metrics.get("turnover", 0) > 0.01
        )

    def log_hopeful_alpha(self, expression: str, data: Dict) -> None:
        """Log promising alphas to a JSON file."""
        log_file = 'hopeful_alphas.json'
        
        # Load existing data
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {log_file}, starting fresh")
        
        # Add new alpha
        metrics = data["alpha"].get("metrics", {})
        entry = {
            "expression": expression,
            "timestamp": int(time.time()),
            "simulation_id": data["simulation"]["id"],
            "alpha_id": data["simulation"]["alpha"],
            "fitness": metrics.get("fitness"),
            "sharpe": metrics.get("sharpe"),
            "turnover": metrics.get("turnover"),
            "returns": metrics.get("returns"),
            "grade": data["alpha"].get("grade")
        }
        
        existing_data.append(entry)
        
        # Save updated data
        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Logged promising alpha to {log_file}")

    def generate_alpha_ideas(self, data_fields: List[Dict], operators: List[Dict]) -> List[str]:
        """Generate alpha ideas by combining and modifying Alpha 101 formulas."""
        try:
            # Organize operators by category
            operator_by_category = {}
            for op in operators:
                category = op['category']
                if category not in operator_by_category:
                    operator_by_category[category] = []
                operator_by_category[category].append({
                    'name': op['name'],
                    'definition': op['definition'],
                    'description': op['description']
                })

            # Fetch existing alphas
            submitted_alphas = self.fetch_submitted_alphas()
            existing_expressions = extract_expressions(submitted_alphas)
            existing_expr_list = "\n".join([f"- {expr['expression']}" for expr in existing_expressions])

            # Get previously tested expressions from results
            tested_expressions = [result["alpha"] for result in self.results]
            all_expressions = existing_expr_list + "\n".join([f"- {expr}" for expr in tested_expressions])

            # Check if prompt might be too long and clear results if needed
            if len(tested_expressions) > 2000:
                logger.warning("Expression list too long, clearing previous results to avoid token limit")
                self.results = []
                all_expressions = existing_expr_list

            print("Preparing prompt...")
            # Select random formulas from Alpha 101 as examples
            sample_formulas = random.sample(ALPHA_101_FORMULAS, min(5, len(ALPHA_101_FORMULAS)))
            formula_examples = "\n".join(sample_formulas)

            prompt = f"""Generate 5 unique alpha factor expressions by combining and modifying these example formulas from the Alpha 101 paper. Return ONLY the expressions, one per line, with no comments or explanations.

IMPORTANT: Do not generate any expressions similar to these previously submitted or tested expressions:
{all_expressions}

Example Alpha 101 Formulas to draw inspiration from:
{formula_examples}

Available Data Fields:
{[field['id'] for field in data_fields]}

Available Operators by Category:
Time Series: {[op['name'] for op in operator_by_category.get('Time Series', [])]}
Cross Sectional: {[op['name'] for op in operator_by_category.get('Cross Sectional', [])]}
Arithmetic: {[op['name'] for op in operator_by_category.get('Arithmetic', [])]}

Requirements:
1. Use similar patterns to the Alpha 101 formulas
2. Combine multiple operators like rank, correlation, ts_mean
3. Use conditional logic with where() function when needed
4. Include proper normalization using rank or zscore

Generate variations that maintain the core logic but use different:
- Time windows (e.g., 5, 10, 20 days)
- Data field combinations
- Operator combinations"""

            headers = {
                'Authorization': f'Bearer sk-JsVxgGzJRBSrVHmguOFVexBGDsR7VcN0GDMYm32G2EbzMuf6',
                'Content-Type': 'application/json'
            }

            data = {
                'model': 'moonshot-v1-8k',
                'messages': [
                    {
                        "role": "system", 
                        "content": "You are a quantitative analyst expert in implementing Alpha 101 strategies."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                'temperature': 0.4  # Slightly higher for more variation
            }

            print("Sending request to Moonshot API...")
            response = requests.post(
                'https://api.moonshot.cn/v1/chat/completions',
                headers=headers,
                json=data
            )

            print(f"Moonshot API response status: {response.status_code}")
            print(f"Moonshot API response: {response.text[:500]}...")

            if response.status_code != 200:
                raise Exception(f"Moonshot API request failed: {response.text}")

            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            
            # Extract expressions
            alpha_ideas = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    alpha_ideas.append(line)
            
            return alpha_ideas

        except Exception as e:
            logger.error(f"Error generating alpha ideas: {str(e)}")
            return []

    def get_data_fields(self) -> List[Dict]:
        """Fetch available data fields from WorldQuant Brain."""
        params = {
            'dataset.id': 'fundamental6',
            'delay': 1,
            'instrumentType': 'EQUITY',
            'limit': 20,
            'offset': 0,
            'region': 'USA',
            'universe': 'TOP3000'
        }
        
        try:
            print("Requesting data fields...")
            response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
            print(f"Data fields response status: {response.status_code}")
            print(f"Data fields response: {response.text[:500]}...")
            
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Failed to fetch data fields: {e}")
            return []

    def get_operators(self) -> List[Dict]:
        """Fetch available operators from WorldQuant Brain."""
        try:
            print("Requesting operators...")
            response = self.sess.get('https://api.worldquantbrain.com/operators')
            print(f"Operators response status: {response.status_code}")
            print(f"Operators response: {response.text[:500]}...")
            
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                return data
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Failed to fetch operators: {e}")
            return []

    def fetch_submitted_alphas(self):
        """Fetch submitted alphas from the WorldQuant Brain API with retry logic"""
        url = "https://api.worldquantbrain.com/users/self/alphas"
        params = {
            "limit": 100,
            "offset": 0,
            "status!=": "UNSUBMITTED%1FIS-FAIL",
            "order": "-dateCreated",
            "hidden": "false"
        }
        
        max_retries = 3
        retry_delay = 60  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.sess.get(url, params=params)
                if response.status_code == 429:  # Too Many Requests
                    wait_time = int(response.headers.get('Retry-After', retry_delay))
                    logger.info(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()["results"]
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch submitted alphas after {max_retries} attempts: {e}")
                    return []
        
        return []

def extract_expressions(alphas):
    """Extract expressions from submitted alphas"""
    expressions = []
    for alpha in alphas:
        if alpha.get("regular") and alpha["regular"].get("code"):
            expressions.append({
                "expression": alpha["regular"]["code"],
                "performance": {
                    "sharpe": alpha["is"].get("sharpe", 0),
                    "fitness": alpha["is"].get("fitness", 0)
                }
            })
    return expressions

def main():
    parser = argparse.ArgumentParser(description='Test Alpha 101 formulas using WorldQuant Brain API')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    parser.add_argument('--batch-size', type=int, default=5,
                      help='Number of alphas to generate and test per batch (default: 5)')
    parser.add_argument('--sleep-time', type=int, default=10,
                      help='Sleep time between tests in seconds (default: 10)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alpha_101_testing.log')
        ]
    )
    
    if not os.path.exists(args.credentials):
        logger.error(f"Credentials file not found: {args.credentials}")
        return 1

    try:
        tester = Alpha101Tester(args.credentials)
        batch_number = 1
        total_successful = 0
        retry_queue = []  # Store alphas that need retry
        
        logger.info("Starting Alpha 101 testing with idea generation")
        
        while True:
            logger.info(f"Processing batch {batch_number}")
            try:
                # First try alphas from retry queue
                if retry_queue:
                    logger.info(f"Processing {len(retry_queue)} alphas from retry queue")
                    alpha = retry_queue.pop(0)
                else:
                    # Generate new ideas if retry queue is empty
                    data_fields = tester.get_data_fields()
                    operators = tester.get_operators()
                    logger.info("Generating new alpha ideas...")
                    alpha_ideas = tester.generate_alpha_ideas(data_fields, operators)
                    
                    if not alpha_ideas:
                        logger.warning("No new ideas generated, waiting before retry...")
                        time.sleep(300)
                        continue
                    
                    logger.info(f"Generated {len(alpha_ideas)} new ideas")
                    alpha = alpha_ideas[0]  # Take one alpha at a time
                    retry_queue.extend(alpha_ideas[1:])  # Queue remaining alphas
                
                # Test the alpha
                logger.info(f"Testing alpha: {alpha}")
                result = tester._test_alpha_impl(alpha)
                
                if result.get("status") == "error":
                    if "SIMULATION_LIMIT_EXCEEDED" in str(result.get("message")):
                        logger.warning("Simulation limit exceeded, queueing for retry")
                        retry_queue.append(alpha)  # Add back to queue
                        logger.info(f"Waiting {args.sleep_time * 2} seconds before retry...")
                        time.sleep(args.sleep_time * 2)  # Wait longer when hitting limits
                        continue
                    else:
                        logger.error(f"Simulation error for alpha: {result.get('message')}")
                        continue
                
                sim_id = result.get("result", {}).get("id")
                if not sim_id:
                    logger.error("No simulation ID received")
                    continue
                
                # Use monitor_simulation to wait for completion
                final_result = tester.monitor_simulation(sim_id, alpha)
                if final_result:
                    logger.info("Simulation completed successfully")
                    total_successful += 1
                
                logger.info(f"Waiting {args.sleep_time} seconds before next test...")
                time.sleep(args.sleep_time)
                
                if not retry_queue and not alpha_ideas:
                    batch_number += 1
                    logger.info(f"Batch {batch_number} complete. Total successful: {total_successful}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_number}: {str(e)}")
                logger.info("Sleeping for 5 minutes before retrying...")
                time.sleep(300)
                continue
        
    except KeyboardInterrupt:
        logger.info("\nStopping alpha testing...")
        if retry_queue:
            logger.info(f"{len(retry_queue)} alphas remaining in retry queue")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 