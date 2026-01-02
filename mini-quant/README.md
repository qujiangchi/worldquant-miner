# Mini-Quant: One-Man Quant Trading Firm

A complete, self-sustained quantitative research and trading platform optimized for one-man operations. The system encompasses the entire lifecycle from research to execution.

## Features

### Complete Lifecycle
1. **Data Gathering**: Multi-source data collection from free/public APIs
2. **Alpha Ideation**: Generate alpha expressions using Generation Two system
3. **Multi-Region Backtesting**: Test alphas across USA, EMEA, CHN, IND, AMER
4. **Alpha Management**: Evaluate, select, and track alphas
5. **Trade Execution**: Real-time execution with risk management

### Key Components

#### DataGatheringEngine
- Multi-source data collection (Yahoo Finance, Alpha Vantage, etc.)
- Region-specific symbol handling
- Data caching and quality monitoring

#### QuantResearchModule
- Generate research hypotheses
- Convert hypotheses to alpha expressions
- Region-specific field availability

#### AlphaBacktestingSystem
- Multi-region backtesting
- Comprehensive performance metrics
- Risk-adjusted returns calculation

#### AlphaPoolStorage
- SQLite database for alpha management
- Performance tracking over time
- Composite scoring for alpha selection

#### TradingAlgorithmEngine
- Real-time signal evaluation
- Position sizing and risk management
- Multi-broker integration

#### OneManQuantSystem
- Complete system orchestrator
- End-to-end workflow automation
- Continuous operation mode

## Architecture

```
OneManQuantSystem
    ├── DataGatheringEngine
    ├── QuantResearchModule
    ├── AlphaBacktestingSystem
    ├── AlphaPoolStorage
    ├── TradingAlgorithmEngine
    └── BrokerAccessLayer
```

## Usage

### Basic Setup

```python
from mini_quant import OneManQuantSystem

# Configuration
config = {
    'database': 'alpha_pool.db',
    'brokers': [
        {
            'name': 'default',
            'credentials': {}
        }
    ],
    'alpha_generator': None  # Optional: Generation Two generator
}

# Initialize system
system = OneManQuantSystem(config)

# Run complete workflow
system.run_complete_workflow(
    regions=['USA', 'EMEA', 'CHN'],
    market_condition='normal'
)
```

### Continuous Operation

```python
# Run in continuous mode
system.run_continuous_operation()
```

### Get System Status

```python
status = system.get_system_status()
print(f"Portfolio value: {status['portfolio']}")
print(f"Active alphas: {status['active_alphas_count']}")
```

## Data Sources

### Market Data
- **Yahoo Finance**: Free historical and real-time data via `yfinance`
- **Alpha Vantage**: Free API (5 calls/minute, 500 calls/day)
- **Polygon.io**: Free tier (5 calls/minute)

### Fundamental Data
- **Financial Modeling Prep API**: Free tier
- **SEC EDGAR**: Free company filings

### Alternative Data
- **Twitter API**: Social sentiment
- **Reddit API**: Community sentiment
- **News APIs**: NewsAPI.org free tier

## Region Support

- **USA**: S&P 500 universe
- **AMER**: LATAM indices
- **EMEA**: STOXX 600
- **CHN**: CSI 300
- **IND**: NIFTY 500

## Alpha Evaluation Criteria

- **Minimum Sharpe**: 1.5
- **Minimum Positive Regions**: 3
- **Maximum Drawdown**: -15%
- **Minimum Win Rate**: 55%
- **Minimum Trades**: 50

## Risk Management

- **Position Size Limit**: 10% per position
- **Daily Loss Limit**: 2%
- **Maximum Exposure**: 100%

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **yfinance**: Yahoo Finance data

## Integration with Generation Two

The system can integrate with Generation Two for enhanced alpha generation:

```python
from generation_two import EnhancedTemplateGeneratorV3

# Create Generation Two generator
gen2 = EnhancedTemplateGeneratorV3(
    credentials_path="credential.txt",
    deepseek_api_key="your_key"
)

# Pass to Mini-Quant
config = {
    'alpha_generator': gen2,
    # ... other config
}
```

## Workflow

1. **Data Gathering**: Collect market data from free sources
2. **Alpha Ideation**: Generate alpha expressions (with or without Generation Two)
3. **Backtesting**: Test across multiple regions
4. **Evaluation**: Evaluate against criteria
5. **Selection**: Select top alphas for trading
6. **Execution**: Execute trades with risk management
7. **Monitoring**: Track performance and remove degrading alphas

## Notes

- Optimized for one-man operations
- Uses free/low-cost data sources
- Automated workflow reduces manual intervention
- Built-in risk management
- SQLite database for persistence
- Extensible broker integration

