"""
Trading Algorithm Engine for Mini-Quant
Real-time alpha execution engine
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Trading order"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', etc.
    time_in_force: str  # 'DAY', 'GTC', etc.
    alpha_id: Optional[str] = None


class PositionManager:
    """Manage trading positions"""
    
    def __init__(self):
        self.positions = {}  # {symbol: quantity}
        self.cash = 100000.0  # Starting cash
    
    def get_position(self, symbol: str) -> float:
        """Get current position for symbol"""
        return self.positions.get(symbol, 0.0)
    
    def update_position(self, symbol: str, quantity: float):
        """Update position"""
        self.positions[symbol] = quantity
    
    def get_total_value(self, market_prices: Dict[str, float]) -> float:
        """Get total portfolio value"""
        positions_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in market_prices.items()
        )
        return self.cash + positions_value


class RiskManager:
    """Risk management"""
    
    def __init__(self):
        self.max_position_size = 0.1  # 10% max per position
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.max_exposure = 1.0  # 100% max exposure
        self.daily_pnl = 0.0
    
    def check_position(self, position: Dict) -> bool:
        """
        Check if position passes risk limits
        
        Args:
            position: Position dictionary with symbol, quantity, price
            
        Returns:
            True if passes, False otherwise
        """
        # Check position size
        position_value = abs(position.get('quantity', 0) * position.get('price', 0))
        portfolio_value = position.get('portfolio_value', 100000)
        
        if position_value / portfolio_value > self.max_position_size:
            logger.warning(f"Position size exceeds limit: {position_value / portfolio_value:.2%}")
            return False
        
        # Check daily loss
        if self.daily_pnl < -self.max_daily_loss * portfolio_value:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False
        
        return True
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl


class BrokerAccessLayer:
    """Broker access layer"""
    
    def __init__(self):
        self.connected_brokers = {}
    
    def connect_broker(self, broker_name: str, credentials: Dict):
        """Connect to a broker"""
        self.connected_brokers[broker_name] = {
            'name': broker_name,
            'credentials': credentials,
            'connected': True
        }
        logger.info(f"Connected to broker: {broker_name}")
    
    def submit_order(self, broker_name: str, order: Order) -> bool:
        """
        Submit order to broker
        
        Args:
            broker_name: Name of broker
            order: Order to submit
            
        Returns:
            True if successful, False otherwise
        """
        if broker_name not in self.connected_brokers:
            logger.error(f"Broker {broker_name} not connected")
            return False
        
        # Placeholder - would actually submit to broker API
        logger.info(
            f"Submitting order to {broker_name}: "
            f"{order.side} {order.quantity} {order.symbol}"
        )
        return True
    
    def get_market_data(self, broker_name: str, symbols: List[str]) -> Dict:
        """Get market data from broker"""
        # Placeholder - would fetch from broker API
        return {symbol: 100.0 for symbol in symbols}


class TradingAlgorithmEngine:
    """
    Real-time alpha execution engine
    
    Executes alpha signals as trades with proper risk management.
    """
    
    def __init__(
        self, 
        alpha_pool, 
        broker_access: BrokerAccessLayer
    ):
        """
        Initialize trading engine
        
        Args:
            alpha_pool: AlphaPoolStorage instance
            broker_access: BrokerAccessLayer instance
        """
        self.alpha_pool = alpha_pool
        self.broker = broker_access
        self.active_alphas = {}
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()
    
    def execute_alpha_signal(
        self, 
        alpha_id: str, 
        signal: float,
        market_data: Dict
    ) -> Optional[Order]:
        """
        Execute trade based on alpha signal
        
        Args:
            alpha_id: Alpha identifier
            signal: Signal value (-1 to 1)
            market_data: Current market data
            
        Returns:
            Order if executed, None otherwise
        """
        # Get alpha configuration
        alpha = self.alpha_pool.get_alpha(alpha_id)
        if not alpha:
            logger.warning(f"Alpha {alpha_id} not found")
            return None
        
        # Calculate target position
        target_position = self.calculate_target_position(
            signal, alpha, market_data
        )
        
        # Risk checks
        portfolio_value = self.position_manager.get_total_value(
            {symbol: market_data.get('price', 100.0) for symbol in market_data.keys()}
        )
        
        position_check = {
            'quantity': target_position['quantity'],
            'price': market_data.get('price', 100.0),
            'portfolio_value': portfolio_value
        }
        
        if not self.risk_manager.check_position(position_check):
            logger.warning(f"Position rejected by risk manager for {alpha_id}")
            return None
        
        # Create order
        order = Order(
            symbol=market_data.get('symbol', 'UNKNOWN'),
            side='buy' if signal > 0 else 'sell',
            quantity=abs(target_position['quantity']),
            order_type='market',
            time_in_force='DAY',
            alpha_id=alpha_id
        )
        
        # Submit to broker
        broker_name = market_data.get('broker', 'default')
        if self.broker.submit_order(broker_name, order):
            # Update position
            current_pos = self.position_manager.get_position(order.symbol)
            new_pos = current_pos + (order.quantity if order.side == 'buy' else -order.quantity)
            self.position_manager.update_position(order.symbol, new_pos)
            
            logger.info(
                f"Executed order for {alpha_id}: "
                f"{order.side} {order.quantity} {order.symbol}"
            )
            return order
        
        return None
    
    def calculate_target_position(
        self, 
        signal: float, 
        alpha: Dict, 
        market_data: Dict
    ) -> Dict:
        """
        Calculate target position based on signal
        
        Args:
            signal: Signal value
            alpha: Alpha configuration
            market_data: Market data
            
        Returns:
            Dictionary with target position details
        """
        # Normalize signal to -1 to 1
        normalized_signal = max(-1.0, min(1.0, signal))
        
        # Get allocation from alpha metadata
        allocation = alpha.get('metadata', {}).get('allocation', 0.1)  # 10% default
        
        # Calculate target value
        portfolio_value = self.position_manager.get_total_value(
            {market_data.get('symbol', 'UNKNOWN'): market_data.get('price', 100.0)}
        )
        target_value = portfolio_value * allocation * abs(normalized_signal)
        
        # Calculate quantity
        price = market_data.get('price', 100.0)
        quantity = target_value / price if price > 0 else 0
        
        return {
            'quantity': quantity,
            'value': target_value,
            'signal': normalized_signal
        }
    
    def evaluate_and_execute(self, alpha_id: str, market_data: Dict):
        """
        Evaluate alpha expression and execute if signal is strong
        
        Args:
            alpha_id: Alpha identifier
            market_data: Current market data
        """
        alpha = self.alpha_pool.get_alpha(alpha_id)
        if not alpha:
            return
        
        # Evaluate expression (simplified - would use proper evaluator)
        expression = alpha.get('expression', '')
        
        # Generate signal (placeholder - would evaluate expression)
        signal = self._evaluate_expression(expression, market_data)
        
        # Execute if signal is strong enough
        if abs(signal) > 0.3:  # Threshold
            self.execute_alpha_signal(alpha_id, signal, market_data)
    
    def _evaluate_expression(self, expression: str, market_data: Dict) -> float:
        """Evaluate alpha expression (simplified)"""
        # Placeholder - would use proper expression evaluator
        import random
        return random.uniform(-1, 1)
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return {
            'cash': self.position_manager.cash,
            'positions': self.position_manager.positions.copy(),
            'daily_pnl': self.risk_manager.daily_pnl,
            'active_alphas': len(self.active_alphas)
        }

