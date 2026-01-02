import argparse
import requests
import json
import os
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict
import time
import logging
import re

# Configure logger with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level for more detailed logs
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    filename='alpha_polisher.log'
)
logger = logging.getLogger(__name__)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class AlphaPolisher:
    def __init__(self, credentials_path: str, moonshot_api_key: str):
        logger.info("Initializing AlphaPolisher...")
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.setup_auth(credentials_path)
        self.moonshot_api_key = moonshot_api_key
        self.operators = self.fetch_operators()
        logger.info("AlphaPolisher initialized successfully")
        
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logger.info(f"Loading credentials from {credentials_path}")
        try:
            with open(credentials_path) as f:
                credentials = json.load(f)
            
            username, password = credentials
            self.sess.auth = HTTPBasicAuth(username, password)
            
            logger.info("Authenticating with WorldQuant Brain...")
            response = self.sess.post('https://api.worldquantbrain.com/authentication')
            logger.debug(f"Authentication response status: {response.status_code}")
            logger.debug(f"Authentication response: {response.text[:500]}...")
            
            if response.status_code != 201:
                raise Exception(f"Authentication failed: {response.text}")
            logger.info("Authentication successful")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    def fetch_operators(self) -> Dict:
        """Fetch available operators from WorldQuant Brain API."""
        logger.info("Fetching available operators...")
        try:
            response = self.sess.get('https://api.worldquantbrain.com/operators')
            logger.debug(f"Operators response status: {response.status_code}")
            
            if response.status_code == 200:
                operators = response.json()
                logger.info(f"Successfully fetched {len(operators)} operators")
                logger.debug(f"Operators: {json.dumps(operators, indent=2)}")
                return operators
            else:
                logger.error(f"Failed to fetch operators: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching operators: {str(e)}")
            return {}

    def analyze_alpha(self, expression: str) -> Dict:
        """Analyze the alpha expression and provide insights using Moonshot API."""
        logger.info(f"Analyzing expression: {expression}")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.moonshot_api_key}'
        }
        
        operators_info = json.dumps(self.operators, indent=2) if self.operators else "Operator information not available"
        logger.debug(f"Using operators info: {operators_info[:500]}...")
        
        data = {
            "model": "moonshot-v1-32k",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert in WorldQuant alpha expressions. Available operators:\n{operators_info}"
                },
                {
                    "role": "user",
                    "content": f"""Please analyze this WorldQuant alpha expression:
{expression}

Provide a concise analysis covering:
1. What market inefficiency or anomaly does this alpha try to capture?
2. What are potential risks or limitations?
3. How could this alpha be improved?
4. What are the key components and their roles?
5. Are all operators used in the expression valid according to the available operators?"""
                }
            ],
            "temperature": 0.7
        }
        
        try:
            logger.info("Sending analysis request to Moonshot API...")
            response = requests.post(
                'https://api.moonshot.ai/v1/chat/completions',
                headers=headers,
                json=data
            )
            logger.debug(f"Analysis response status: {response.status_code}")
            logger.debug(f"Analysis response: {response.text[:500]}...")
            
            if response.status_code == 200:
                analysis = response.json()['choices'][0]['message']['content']
                logger.info("Successfully generated analysis")
                logger.debug(f"Analysis result: {analysis}")
            else:
                logger.error(f"Moonshot API error: {response.text}")
                analysis = "Error generating analysis."
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            analysis = "Error generating analysis."

        return {"analysis": analysis}

    def polish_expression(self, expression: str, user_requirements: str = "") -> Dict:
        """Request polished version of the alpha expression from Moonshot API."""
        logger.info(f"Polishing expression: {expression}")
        logger.info(f"User requirements: {user_requirements}")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.moonshot_api_key}'
        }
        
        operators_info = json.dumps(self.operators, indent=2) if self.operators else "Operator information not available"
        logger.debug(f"Using operators info: {operators_info[:500]}...")
        
        # Prepare the user message with requirements if provided
        user_message = f"Please polish this WorldQuant alpha expression to improve its performance while maintaining its core strategy."
        if user_requirements:
            user_message += f"\nSpecific requirements:\n{user_requirements}"
        user_message += f"\nExpression to polish: {expression}\nReturn the polished expression only, no other text. For example: -abs(subtract(news_max_up_ret, news_max_dn_ret))"
        
        data = {
            "model": "moonshot-v1-32k",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert in WorldQuant alpha expressions. Available operators:\n{operators_info}"
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "temperature": 0.7
        }
        
        try:
            logger.info("Sending polish request to Moonshot API...")
            response = requests.post(
                'https://api.moonshot.ai/v1/chat/completions',
                headers=headers,
                json=data
            )
            logger.debug(f"Polish response status: {response.status_code}")
            logger.debug(f"Polish response: {response.text[:500]}...")
            
            if response.status_code == 200:
                polished = response.json()['choices'][0]['message']['content']
                logger.info("Successfully polished expression")
                logger.debug(f"Polished result: {polished}")
            else:
                logger.error(f"Moonshot API error: {response.text}")
                polished = "Error polishing expression."
        except Exception as e:
            logger.error(f"Error in expression polishing: {str(e)}")
            polished = "Error polishing expression."

        return {"polished_expression": polished}

    def simulate_alpha(self, expression: str) -> Dict:
        """Simulate the alpha expression using WorldQuant Brain API."""
        logger.info(f"Simulating expression: {expression}")
        url = 'https://api.worldquantbrain.com/simulations'
        
        data = {
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
        logger.debug(f"Simulation request data: {json.dumps(data, indent=2)}")
        
        try:
            logger.info("Sending simulation request...")
            response = self.sess.post(url, json=data)
            logger.debug(f"Simulation creation response status: {response.status_code}")
            logger.debug(f"Simulation creation response: {response.text[:500]}...")
            
            if response.status_code == 201:
                simulation_id = response.json()['id']
                logger.info(f"Simulation created with ID: {simulation_id}")
                
                # Wait for simulation to complete
                while True:
                    sleep(1)
                    result = self.sess.get(f"{url}/{simulation_id}")
                    
                    if result.status_code == 200:
                        sim_data = result.json()
                        status = sim_data.get('status')
                        logger.info(f"Simulation status: {status}")
                        
                        if status == 'COMPLETED':
                            logger.info("Simulation completed successfully")
                            # Get the simulation results
                            result_response = self.sess.get(f"{url}/{simulation_id}/result")
                            if result_response.status_code == 200:
                                sim_results = result_response.json()
                                logger.info("Successfully retrieved simulation results")
                                return {
                                    "status": "success",
                                    "results": sim_results
                                }
                            else:
                                error_msg = f"Failed to get simulation results: {result_response.text}"
                                logger.error(error_msg)
                                return {"status": "error", "message": error_msg}
                        elif status == 'FAILED':
                            error_msg = f"Simulation failed: {sim_data.get('message', 'Unknown error')}"
                            logger.error(error_msg)
                            return {"status": "error", "message": error_msg}
                        elif status == 'RUNNING':
                            logger.debug("Simulation still running...")
                            continue
                        else:
                            error_msg = f"Unknown simulation status: {status}"
                            logger.error(error_msg)
                            return {"status": "error", "message": error_msg}
                    else:
                        error_msg = f"Failed to check simulation status: {result.text}"
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
            
            error_msg = f"Failed to create simulation: {response.text}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in simulation: {error_msg}")
            return {"status": "error", "message": error_msg}

def main():
    parser = argparse.ArgumentParser(description='Polish and analyze WorldQuant alpha expressions')
    parser.add_argument('--credentials', type=str, required=True, help='Path to credentials file')
    parser.add_argument('--moonshot-key', type=str, required=True, help='Moonshot API key')
    args = parser.parse_args()

    try:
        logger.info("Starting Alpha Polisher...")
        polisher = AlphaPolisher(args.credentials, args.moonshot_key)
        
        # Print available operators
        print("\nAvailable Operators:")
        print(json.dumps(polisher.operators, indent=2))
        
        while True:
            print("\nEnter your alpha expression (or 'quit' to exit):")
            expression = input().strip()
            
            if expression.lower() == 'quit':
                logger.info("User requested to quit")
                break
            
            print("\nEnter your polishing requirements (optional, press Enter to skip):")
            print("Examples:")
            print("- Focus on improving IR")
            print("- Reduce turnover")
            print("- Make it more market neutral")
            print("- Add more technical indicators")
            user_requirements = input().strip()
            
            logger.info(f"Processing expression: {expression}")
            logger.info(f"User requirements: {user_requirements}")
            
            print("\nPolishing expression...")
            polished = polisher.polish_expression(expression, user_requirements)
            print("\nPolished expression:")
            print(polished['polished_expression'])
            
            print("\nSimulating original expression...")
            sim_result = polisher.simulate_alpha(expression)
            print("\nSimulation results:")
            print(json.dumps(sim_result, indent=2))
            
            if polished['polished_expression'] != "Error polishing expression.":
                print("\nSimulating polished expression...")
                polished_sim = polisher.simulate_alpha(polished['polished_expression'])
                print("\nPolished simulation results:")
                print(json.dumps(polished_sim, indent=2))
        
        logger.info("Alpha Polisher completed successfully")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 