"""
Alpha Pool Storage for Mini-Quant
Database for alpha management and evaluation
"""

import logging
import sqlite3
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def track(self, alpha_id: str, metrics: Dict):
        """Track metrics for an alpha"""
        if alpha_id not in self.metrics:
            self.metrics[alpha_id] = []
        self.metrics[alpha_id].append({
            **metrics,
            'timestamp': datetime.now().isoformat()
        })


class AlphaPoolStorage:
    """
    Database for alpha management
    
    Stores alphas, tracks performance, and selects alphas for trading.
    """
    
    def __init__(self, db_path: str = "alpha_pool.db"):
        """
        Initialize alpha pool storage
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.active_alphas = {}
        self.performance_tracker = PerformanceTracker()
        self.create_tables()
    
    def create_tables(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Alphas table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alphas (
                id TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                metadata TEXT
            )
        ''')
        
        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alpha_id TEXT,
                region TEXT,
                sharpe REAL,
                returns REAL,
                max_drawdown REAL,
                win_rate REAL,
                num_trades INTEGER,
                backtest_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (alpha_id) REFERENCES alphas(id)
            )
        ''')
        
        # Performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alpha_id TEXT,
                sharpe REAL,
                fitness REAL,
                returns REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (alpha_id) REFERENCES alphas(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database tables created")
    
    def add_alpha(self, alpha_id: str, expression: str, metadata: Dict = None):
        """Add alpha to pool"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO alphas (id, expression, metadata)
            VALUES (?, ?, ?)
        ''', (alpha_id, expression, metadata_json))
        
        conn.commit()
        conn.close()
        
        self.active_alphas[alpha_id] = {
            'expression': expression,
            'metadata': metadata or {}
        }
        
        logger.info(f"Added alpha {alpha_id} to pool")
    
    def store_backtest_results(self, alpha_id: str, backtest_results: Dict):
        """Store backtest results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for region, result in backtest_results.items():
            if hasattr(result, 'sharpe'):
                cursor.execute('''
                    INSERT INTO backtest_results 
                    (alpha_id, region, sharpe, returns, max_drawdown, win_rate, num_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alpha_id,
                    region,
                    result.sharpe,
                    result.returns,
                    result.max_drawdown,
                    result.win_rate,
                    result.num_trades
                ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored backtest results for alpha {alpha_id}")
    
    def evaluate_alpha(
        self, 
        alpha: Dict, 
        backtest_results: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate if alpha should be included in pool
        
        Args:
            alpha: Alpha dictionary
            backtest_results: Backtest results by region
            
        Returns:
            Tuple of (passes, reasons)
        """
        criteria = {
            'min_sharpe': 1.5,
            'min_positive_regions': 3,  # Must work in at least 3 regions
            'max_drawdown': -0.15,  # Max 15% drawdown
            'min_win_rate': 0.55,  # 55% win rate
            'min_trades': 50  # At least 50 trades
        }
        
        passes = True
        reasons = []
        
        # Check Sharpe ratio
        sharpe_values = [
            r.sharpe for r in backtest_results.values() 
            if hasattr(r, 'sharpe') and r.sharpe is not None
        ]
        if sharpe_values:
            avg_sharpe = np.mean(sharpe_values)
            if avg_sharpe < criteria['min_sharpe']:
                passes = False
                reasons.append(f"Sharpe {avg_sharpe:.2f} < {criteria['min_sharpe']}")
        else:
            passes = False
            reasons.append("No valid Sharpe ratios")
        
        # Check positive regions
        positive_regions = sum(
            1 for r in backtest_results.values() 
            if hasattr(r, 'sharpe') and r.sharpe and r.sharpe > 1.0
        )
        if positive_regions < criteria['min_positive_regions']:
            passes = False
            reasons.append(f"Only {positive_regions} positive regions")
        
        # Check drawdown
        max_dd_values = [
            r.max_drawdown for r in backtest_results.values()
            if hasattr(r, 'max_drawdown') and r.max_drawdown is not None
        ]
        if max_dd_values:
            worst_dd = min(max_dd_values)
            if worst_dd < criteria['max_drawdown']:
                passes = False
                reasons.append(f"Max drawdown {worst_dd:.2%} < {criteria['max_drawdown']:.2%}")
        
        return passes, reasons
    
    def select_alphas_for_trading(self, limit: int = 10) -> List[Dict]:
        """
        Select top alphas for live trading
        
        Args:
            limit: Maximum number of alphas to select
            
        Returns:
            List of selected alpha dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all active alphas with their backtest results
        cursor.execute('''
            SELECT a.id, a.expression, a.metadata,
                   AVG(br.sharpe) as avg_sharpe,
                   COUNT(DISTINCT br.region) as num_regions,
                   MIN(br.max_drawdown) as worst_drawdown
            FROM alphas a
            LEFT JOIN backtest_results br ON a.id = br.alpha_id
            WHERE a.status = 'active'
            GROUP BY a.id
        ''')
        
        alphas = []
        for row in cursor.fetchall():
            alpha_id, expression, metadata_json, avg_sharpe, num_regions, worst_drawdown = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            alphas.append({
                'id': alpha_id,
                'expression': expression,
                'metadata': metadata,
                'avg_sharpe': avg_sharpe or 0.0,
                'num_regions': num_regions or 0,
                'worst_drawdown': worst_drawdown or 0.0
            })
        
        conn.close()
        
        # Calculate composite scores
        scored_alphas = []
        for alpha in alphas:
            score = self.calculate_composite_score(alpha)
            scored_alphas.append((alpha, score))
        
        # Sort by score
        scored_alphas.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = [alpha for alpha, score in scored_alphas[:limit]]
        
        logger.info(f"Selected {len(selected)} alphas for trading")
        return selected
    
    def calculate_composite_score(self, alpha: Dict) -> float:
        """
        Calculate composite score for alpha selection
        
        Args:
            alpha: Alpha dictionary
            
        Returns:
            Composite score
        """
        sharpe_weight = 0.4
        consistency_weight = 0.3
        robustness_weight = 0.2
        recency_weight = 0.1
        
        sharpe_score = min(1.0, alpha.get('avg_sharpe', 0) / 2.0)
        consistency_score = min(1.0, alpha.get('num_regions', 0) / 5.0)
        robustness_score = 1.0 - abs(alpha.get('worst_drawdown', 0)) / 0.2
        recency_score = 1.0 if alpha.get('recent', False) else 0.5
        
        composite = (
            sharpe_score * sharpe_weight +
            consistency_score * consistency_weight +
            robustness_score * robustness_weight +
            recency_score * recency_weight
        )
        
        return composite
    
    def get_alpha(self, alpha_id: str) -> Optional[Dict]:
        """Get alpha by ID"""
        if alpha_id in self.active_alphas:
            return self.active_alphas[alpha_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, expression, metadata FROM alphas WHERE id = ?', (alpha_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            alpha_id, expression, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            return {
                'id': alpha_id,
                'expression': expression,
                'metadata': metadata
            }
        
        return None
    
    def get_top_alphas(self, limit: int = 20) -> List[Dict]:
        """Get top performing alphas"""
        return self.select_alphas_for_trading(limit)
    
    def deactivate_alpha(self, alpha_id: str):
        """Deactivate an alpha"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE alphas SET status = ? WHERE id = ?', ('inactive', alpha_id))
        conn.commit()
        conn.close()
        
        if alpha_id in self.active_alphas:
            del self.active_alphas[alpha_id]
        
        logger.info(f"Deactivated alpha {alpha_id}")

