"""
One-Man Quant System Orchestrator
Complete system that ties all components together
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .data_gathering_engine import DataGatheringEngine
from .quant_research_module import QuantResearchModule
from .alpha_backtesting_system import AlphaBacktestingSystem
from .alpha_pool_storage import AlphaPoolStorage
from .trading_algorithm_engine import TradingAlgorithmEngine, BrokerAccessLayer

logger = logging.getLogger(__name__)


class OneManQuantSystem:
    """
    Complete one-man quant trading firm system
    
    Orchestrates the entire lifecycle from research to execution.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize complete system
        
        Args:
            config: Configuration dictionary with:
                - database: Database path
                - brokers: List of broker configurations
                - alpha_generator: Optional alpha generator (Generation Two)
        """
        # Initialize all components
        self.data_engine = DataGatheringEngine()
        self.research_module = QuantResearchModule(
            alpha_generator=config.get('alpha_generator')
        )
        self.backtesting = AlphaBacktestingSystem(self.data_engine)
        self.alpha_pool = AlphaPoolStorage(config.get('database', 'alpha_pool.db'))
        self.broker_access = BrokerAccessLayer()
        self.trading_engine = TradingAlgorithmEngine(
            self.alpha_pool, 
            self.broker_access
        )
        
        # Connect brokers
        for broker_config in config.get('brokers', []):
            self.broker_access.connect_broker(
                broker_config['name'],
                broker_config.get('credentials', {})
            )
        
        logger.info("One-Man Quant System initialized")
    
    def run_complete_workflow(
        self,
        regions: List[str] = None,
        market_condition: str = "normal"
    ):
        """
        Run complete workflow from research to execution
        
        Args:
            regions: List of regions to work with
            market_condition: Current market condition
        """
        if regions is None:
            regions = ['USA', 'EMEA', 'CHN']
        
        logger.info("=== Starting Complete Workflow ===")
        
        # Phase 1: Data Gathering
        logger.info("Phase 1: Data Gathering")
        # Data is gathered on-demand, so this is implicit
        
        # Phase 2: Alpha Ideation
        logger.info("Phase 2: Alpha Ideation")
        all_alphas = []
        for region in regions:
            alphas = self.research_module.generate_alphas_for_region(
                region, 
                market_condition
            )
            all_alphas.extend(alphas)
            logger.info(f"Generated {len(alphas)} alphas for {region}")
        
        # Phase 3: Backtesting
        logger.info("Phase 3: Multi-Region Backtesting")
        start_date = datetime.now() - timedelta(days=365*2)  # 2 years
        end_date = datetime.now()
        
        for alpha_expr in all_alphas[:10]:  # Limit for demo
            logger.info(f"Backtesting: {alpha_expr[:50]}...")
            backtest_results = self.backtesting.backtest_alpha_multi_region(
                alpha_expr,
                start_date,
                end_date
            )
            
            # Phase 4: Alpha Management
            alpha_id = f"alpha_{hash(alpha_expr) % 1000000}"
            passes, reasons = self.alpha_pool.evaluate_alpha(
                {'expression': alpha_expr},
                backtest_results
            )
            
            if passes:
                logger.info(f"Alpha {alpha_id} passed evaluation")
                self.alpha_pool.add_alpha(alpha_id, alpha_expr)
                self.alpha_pool.store_backtest_results(alpha_id, backtest_results)
            else:
                logger.info(f"Alpha {alpha_id} failed: {reasons}")
        
        # Phase 5: Trade Execution
        logger.info("Phase 5: Trade Execution")
        selected_alphas = self.alpha_pool.select_alphas_for_trading(limit=5)
        
        logger.info(f"Selected {len(selected_alphas)} alphas for trading")
        
        # Execute trades (simplified - would run continuously)
        for alpha in selected_alphas:
            alpha_id = alpha['id']
            # Get market data
            market_data = self.broker_access.get_market_data(
                'default',
                ['AAPL', 'MSFT']  # Example symbols
            )
            
            # Evaluate and execute
            for symbol, price in market_data.items():
                self.trading_engine.evaluate_and_execute(
                    alpha_id,
                    {'symbol': symbol, 'price': price, 'broker': 'default'}
                )
        
        logger.info("=== Complete Workflow Finished ===")
    
    def run_continuous_operation(self):
        """Run continuous operation mode"""
        logger.info("Starting continuous operation mode")
        
        # This would run in a loop, continuously:
        # 1. Generate new alphas
        # 2. Backtest them
        # 3. Evaluate and add to pool
        # 4. Execute trades from selected alphas
        # 5. Monitor performance
        # 6. Remove degrading alphas
        
        # For demo, just run one cycle
        self.run_complete_workflow()
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        portfolio_status = self.trading_engine.get_portfolio_status()
        top_alphas = self.alpha_pool.get_top_alphas(limit=5)
        
        return {
            'portfolio': portfolio_status,
            'top_alphas': top_alphas,
            'active_alphas_count': len(self.alpha_pool.active_alphas),
            'connected_brokers': list(self.broker_access.connected_brokers.keys())
        }

