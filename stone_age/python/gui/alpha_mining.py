from dataclasses import dataclass
from typing import List, Dict, Any, Generator
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

@dataclass
class AlphaMiningParams:
    max_iterations: int = 1000
    population_size: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    data_window: int = 252
    min_samples: int = 1000

class AlphaMiner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.params = AlphaMiningParams(
            max_iterations=config.get('max_iterations', 1000),
            population_size=config.get('population_size', 100)
        )
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.history = []
        self.is_running = False

    def initialize_population(self, size: int) -> List[Dict[str, Any]]:
        population = []
        for _ in range(size):
            alpha = {
                'parameters': self._generate_random_parameters(),
                'fitness': 0.0,
                'metrics': {},
                'creation_time': datetime.now().isoformat()
            }
            population.append(alpha)
        return population

    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters for alpha strategy"""
        operators = ['rank', 'decay', 'scale', 'delta']
        windows = [5, 10, 20, 60]
        
        params = {
            'operator': np.random.choice(operators),
            'window': np.random.choice(windows),
            'weight': np.random.uniform(-1, 1),
            'decay_rate': np.random.uniform(0, 0.5),
            'scale_factor': np.random.uniform(0.5, 2.0)
        }
        return params

    def evaluate_fitness(self, alpha: Dict[str, Any], data: pd.DataFrame) -> float:
        """Evaluate alpha strategy fitness using historical data"""
        try:
            params = alpha['parameters']
            series = self._apply_alpha_strategy(data, params)
            
            # Calculate key metrics
            sharpe = self._calculate_sharpe_ratio(series)
            turnover = self._calculate_turnover(series)
            ic = self._calculate_information_coefficient(series, data['returns'])
            
            # Store metrics in alpha
            alpha['metrics'] = {
                'sharpe_ratio': sharpe,
                'turnover': turnover,
                'ic': ic
            }
            
            # Combined fitness score
            fitness = sharpe * 0.4 + ic * 0.4 - turnover * 0.2
            return fitness
            
        except Exception as e:
            print(f"Error evaluating alpha: {str(e)}")
            return float('-inf')

    def _apply_alpha_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Apply alpha strategy to data using parameters"""
        operator = params['operator']
        window = params['window']
        weight = params['weight']
        
        if operator == 'rank':
            series = data['close'].rolling(window).rank()
        elif operator == 'decay':
            decay_rate = params['decay_rate']
            weights = np.exp(-decay_rate * np.arange(window))
            series = data['close'].rolling(window).apply(
                lambda x: np.sum(x * weights) / np.sum(weights)
            )
        elif operator == 'scale':
            scale = params['scale_factor']
            series = data['close'].rolling(window).mean() * scale
        elif operator == 'delta':
            series = data['close'].diff(window)
        
        return series * weight

    def _calculate_sharpe_ratio(self, series: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        returns = series.pct_change()
        annual_sharpe = np.sqrt(252) * returns.mean() / returns.std()
        return float(annual_sharpe)

    def _calculate_turnover(self, series: pd.Series) -> float:
        """Calculate strategy turnover"""
        changes = np.abs(series.diff())
        turnover = changes.mean()
        return float(turnover)

    def _calculate_information_coefficient(self, series: pd.Series, returns: pd.Series) -> float:
        """Calculate information coefficient (IC)"""
        correlation = series.corr(returns)
        return float(correlation)

    def mine_alphas(self) -> Generator[Dict[str, Any], None, None]:
        """Main alpha mining process with progress updates"""
        self.is_running = True
        population = self.initialize_population(self.params.population_size)
        
        # Simulate some data for testing
        # In production, this would be real market data
        dates = pd.date_range(start='2020-01-01', periods=1000)
        data = pd.DataFrame({
            'close': np.random.random(1000),
            'returns': np.random.random(1000)
        }, index=dates)
        
        for iteration in range(self.params.max_iterations):
            if not self.is_running:
                break
                
            # Evaluate population
            for alpha in population:
                fitness = self.evaluate_fitness(alpha, data)
                alpha['fitness'] = fitness
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = alpha.copy()
                    
                    # Yield progress update with new best alpha
                    yield {
                        'progress': (iteration + 1) / self.params.max_iterations * 100,
                        'alpha': alpha['parameters'],
                        'score': fitness,
                        'metrics': alpha['metrics']
                    }
            
            # Generate new population
            new_population = []
            while len(new_population) < self.params.population_size:
                if np.random.random() < self.params.crossover_rate:
                    parent1, parent2 = np.random.choice(population, 2)
                    child = self._crossover(parent1, parent2)
                else:
                    child = np.random.choice(population).copy()
                
                if np.random.random() < self.params.mutation_rate:
                    self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Regular progress update
            if iteration % 10 == 0:
                yield {
                    'progress': (iteration + 1) / self.params.max_iterations * 100
                }

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parent alphas"""
        child = {
            'parameters': {},
            'fitness': 0.0,
            'metrics': {},
            'creation_time': datetime.now().isoformat()
        }
        
        for key in parent1['parameters']:
            if np.random.random() < 0.5:
                child['parameters'][key] = parent1['parameters'][key]
            else:
                child['parameters'][key] = parent2['parameters'][key]
        return child

    def _mutate(self, alpha: Dict[str, Any]) -> None:
        """Mutate alpha parameters"""
        params = alpha['parameters']
        param_to_mutate = np.random.choice(list(params.keys()))
        
        if param_to_mutate == 'operator':
            operators = ['rank', 'decay', 'scale', 'delta']
            params['operator'] = np.random.choice(operators)
        elif param_to_mutate == 'window':
            windows = [5, 10, 20, 60]
            params['window'] = np.random.choice(windows)
        elif param_to_mutate in ['weight', 'decay_rate', 'scale_factor']:
            current_value = params[param_to_mutate]
            params[param_to_mutate] = current_value + np.random.normal(0, 0.1)

    def save_alpha(self, alpha: Dict[str, Any], filepath: str) -> None:
        """Save alpha strategy to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(alpha, f, indent=2)

    def load_alpha(self, filepath: str) -> Dict[str, Any]:
        """Load alpha strategy from file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        return self.history 