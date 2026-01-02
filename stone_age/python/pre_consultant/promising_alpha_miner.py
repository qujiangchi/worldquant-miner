import json
import re
import time
import argparse
import logging
import os
from itertools import product
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Tuple
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpha_mining.log'),
        logging.StreamHandler()
    ]
)

class RateLimiter:
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = []
        self.lock = threading.Lock()
        
    def can_make_request(self) -> bool:
        now = datetime.now()
        with self.lock:
            # Remove old requests
            self.requests = [t for t in self.requests 
                           if t > now - timedelta(seconds=self.time_window)]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
            
    def wait_for_slot(self):
        while not self.can_make_request():
            sleep(1)

class PromisingAlphaMiner:
    def __init__(self, credentials_path: str):
        self.sess = requests.Session()
        self.setup_auth(credentials_path)
        self.rate_limiter = RateLimiter(max_requests=8, time_window=60)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logging.info(f"Loading credentials from {credentials_path}")
        with open(credentials_path) as f:
            credentials = json.load(f)
        
        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)
        
        logging.info("Authenticating with WorldQuant Brain...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        if response.status_code != 201:
            raise Exception(f"Authentication failed: {response.text}")

    def parse_expression(self, expression: str) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Parse expression to find numeric parameters and their positions."""
        numbers = []
        positions = []
        for match in re.finditer(r'\d+', expression):
            numbers.append(int(match.group()))
            positions.append((match.start(), match.end()))
        return positions, numbers

    def generate_parameter_combinations(self, num_params: int, base_values: List[int]) -> List[List[int]]:
        """Generate parameter combinations using smart ranges around base values."""
        combinations = []
        
        for base_value in base_values:
            # Define range based on base value
            if base_value <= 10:
                # For small values, test more granularly
                values = [max(1, int(base_value * factor)) for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
            else:
                # For larger values, test wider range
                lower_bound = max(1, int(base_value * 0.5))  # -50%
                upper_bound = int(base_value * 1.5)  # +50%
                step = max(1, int((upper_bound - lower_bound) / 4))  # 4 steps between bounds
                values = list(range(lower_bound, upper_bound + 1, step))
                
            if base_value not in values:
                values.append(base_value)  # Always include original value
                values.sort()
                
            logging.info(f"Testing parameter variations for {base_value}: {values}")
            combinations.append(values)
        
        # Generate all combinations of parameter values
        return list(product(*combinations))

    def create_expression_variant(self, base_expression: str, positions: List[Tuple[int, int]], params: List[int]) -> str:
        """Create a new expression with the given parameters."""
        result = base_expression
        offset = 0
        for (start, end), new_value in zip(positions, params):
            new_str = str(new_value)
            start += offset
            end += offset
            result = result[:start] + new_str + result[end:]
            offset += len(new_str) - (end - start)
        return result

    def test_alpha(self, expression: str, max_retries: int = 3) -> Dict:
        """Test an alpha expression using WorldQuant Brain simulation."""
        logging.info(f"Testing alpha: {expression}")
        
        simulation_data = {
            'type': 'REGULAR',
            'settings': {
                'instrumentType': 'EQUITY',
                'region': 'USA',
                'universe': 'TOP3000',
                'delay': 1,
                'decay': 0,
                'neutralization': 'INDUSTRY',
                'truncation': 0.08,
                'pasteurization': 'ON',
                'unitHandling': 'VERIFY',
                'nanHandling': 'OFF',
                'language': 'FASTEXPR',
                'visualization': False,
            },
            'regular': expression
        }

        for attempt in range(max_retries):
            try:
                sim_resp = self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
                if sim_resp.status_code == 401:  # Unauthorized
                    logging.info("Session expired, re-authenticating...")
                    self.setup_auth('./credential.txt')
                    continue
                    
                if sim_resp.status_code != 201:
                    logging.error(f"Simulation failed (attempt {attempt+1}/{max_retries}): {sim_resp.text}")
                    sleep(60 * (attempt + 1))  # Exponential backoff
                    continue

                sim_progress_url = sim_resp.headers.get('location')
                if not sim_progress_url:
                    logging.error("No location header in response")
                    continue

                sim_progress_resp = self.sess.get(sim_progress_url)
                retry_after_sec = float(sim_progress_resp.headers.get('Retry-After', 0))
                if retry_after_sec > 0:
                    sleep(retry_after_sec)
                else:
                    break

                sim_result = sim_progress_resp.json()
                if sim_result.get("status") == "ERROR":
                    return None

                alpha_id = sim_result.get("alpha")
                if not alpha_id:
                    return None

                alpha_resp = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                if alpha_resp.status_code != 200:
                    return None

                return alpha_resp.json()

            except Exception as e:
                logging.error(f"Error in simulation (attempt {attempt+1}/{max_retries}): {str(e)}")
                sleep(60 * (attempt + 1))  # Exponential backoff
                self.setup_auth('./credential.txt')  # Re-establish session
                continue
            
        logging.error(f"Failed to test alpha after {max_retries} attempts")
        return None

    def test_alpha_batch(self, variants: List[str]) -> List[Dict]:
        """Test multiple alpha variants concurrently."""
        futures = []
        results = []
        
        for variant in variants:
            self.rate_limiter.wait_for_slot()
            futures.append(self.executor.submit(self.test_alpha, variant))
            
        for future in futures:
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"Error in concurrent simulation: {str(e)}")
                
        return results

    def meets_criteria(self, alpha_data: Dict) -> bool:
        """Check if alpha meets submission criteria."""
        if not alpha_data.get("is"):
            return False
            
            
        is_data = alpha_data["is"]
        checks = {check["name"]: check for check in is_data["checks"]}
        
        # Check other criteria
        if (is_data["sharpe"] <= 1.25 or
            is_data["turnover"] <= 0.01 or
            is_data["turnover"] >= 0.7 or
            is_data["fitness"] < 1.0 or
            checks.get("CONCENTRATED_WEIGHT", {}).get("result") != "PASS" or
            checks.get("LOW_SUB_UNIVERSE_SHARPE", {}).get("result") != "PASS"):
            return False
            
        logging.info(f"Found promising alpha with grade {alpha_data.get('grade')}, "
                    f"sharpe: {is_data['sharpe']:.2f}, fitness: {is_data['fitness']:.2f}")
        return True

    def submit_alpha(self, alpha_id: str) -> bool:
        """Submit an alpha for review."""
        logging.info(f'Attempting to submit alpha {alpha_id}')
        
        self.sess.post(f'https://api.worldquantbrain.com/alphas/{alpha_id}/submit')
        
        while True:
            submit_resp = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}/submit')
            if submit_resp.status_code == 404:
                logging.info('Alpha already submitted')
                return False
                
            if submit_resp.content:
                result = submit_resp.json()
                for check in result['is']['checks']:
                    if check['name'] == 'SELF_CORRELATION':
                        logging.info(f'Submission check result: {check}')
                        return check['result'] == 'PASS'
                break
                
            sleep(5)
            
        logging.info(f'Submission response: {submit_resp.json()}')
        return False

    def clean_expression(self, alpha_data: Dict) -> Dict:
        """Clean alpha expression data to extract only valid expressions."""
        if not isinstance(alpha_data, dict) or "expression" not in alpha_data:
            return None
        
        expression = alpha_data["expression"]
        
        # Skip if expression is just a number or single word
        if re.match(r'^\d+\.?$|^[a-zA-Z]+$', expression):
            logging.info(f"Skipping invalid expression: {expression}")
            return None
        
        # Skip if expression is a description (contains common English words)
        common_words = ['it', 'the', 'is', 'are', 'captures', 'provides', 'measures']
        if any(word in expression.lower() for word in common_words):
            logging.info(f"Skipping description text: {expression}")
            return None
        
        # Verify expression contains valid operators/functions
        valid_functions = ['ts_mean', 'divide', 'subtract', 'add', 'multiply', 'zscore', 
                          'ts_rank', 'ts_std_dev', 'rank', 'log', 'sqrt']
        if not any(func in expression for func in valid_functions):
            logging.info(f"Expression lacks valid functions: {expression}")
            return None
        
        return alpha_data

    def process_hopeful_alpha(self, alpha_data: Dict) -> bool:
        """Process a single hopeful alpha, trying parameter variations."""
        
        cleaned_data = self.clean_expression(alpha_data)
        if not cleaned_data:
            return True
        
        expression = cleaned_data["expression"]
        positions, base_values = self.parse_expression(expression)
        
        if not positions:
            logging.info(f"No numeric parameters found in expression: {expression}")
            return True
        
        # Generate parameter variations
        combinations = self.generate_parameter_combinations(len(base_values), base_values)
        logging.info(f"Generated {len(combinations)} parameter combinations")
        
        variants = []
        for params in combinations:
            variant = self.create_expression_variant(expression, positions, params)
            variants.append(variant)
            logging.debug(f"Generated variant: {variant}")
        
        # Test variants in batches
        batch_size = 10
        for i in range(0, len(variants), batch_size):
            batch = variants[i:i+batch_size]
            logging.info(f"Testing batch {i//batch_size + 1}/{(len(variants)-1)//batch_size + 1}")
            
            results = self.test_alpha_batch(batch)
            
            for result in results:
                if self.meets_criteria(result):
                    logging.info(f"Found successful variant in batch")
                    if self.submit_alpha(result["id"]):
                        logging.info("Alpha submitted successfully!")
                        return True
                        
            sleep(5)  # Small delay between batches
            
        return True

    def run(self):
        """Main loop to process hopeful alphas."""
        logging.info("Starting promising alpha miner...")
        
        while True:
            try:
                if not os.path.exists('hopeful_alphas.json'):
                    logging.info("No hopeful alphas file found. Waiting...")
                    sleep(60)
                    continue
                    
                with open('hopeful_alphas.json', 'r') as f:
                    hopeful_alphas = json.load(f)
                
                if not hopeful_alphas:
                    logging.info("No hopeful alphas to process. Waiting...")
                    sleep(60)
                    continue
                
                to_remove = []
                for i, alpha in enumerate(hopeful_alphas):
                    logging.info(f"Processing alpha {i+1}/{len(hopeful_alphas)}: {alpha['expression']}")
                    if self.process_hopeful_alpha(alpha):
                        to_remove.append(alpha)
                        
                # Remove processed alphas
                if to_remove:
                    hopeful_alphas = [a for a in hopeful_alphas if a not in to_remove]
                    with open('hopeful_alphas.json', 'w') as f:
                        json.dump(hopeful_alphas, f, indent=2)
                    logging.info(f"Removed {len(to_remove)} processed alphas from hopeful_alphas.json")
                
                logging.info("Waiting for new hopeful alphas...")
                sleep(60)
                
            except KeyboardInterrupt:
                logging.info("Stopping alpha miner...")
                break
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                sleep(300)
                continue

def main():
    parser = argparse.ArgumentParser(description='Mine promising alphas by varying parameters')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to credentials file (default: ./credential.txt)')
    
    args = parser.parse_args()
    
    miner = PromisingAlphaMiner(args.credentials)
    miner.run()

if __name__ == "__main__":
    main() 