# Alpha Mining Workbench

A comprehensive GUI application for developing, testing, optimizing, and submitting alpha strategies for quantitative trading.

## Features

- **Alpha Mining**: Generate alpha strategies using genetic algorithms or machine learning
- **Expression Mining**: Create variations of existing alpha expressions by modifying parameters
- **Parameter Optimization**: Fine-tune parameters to maximize alpha performance
- **Alpha Submission**: Test and submit strategies to the trading platform
- **Results Management**: Save, load, and analyze your alpha strategies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alpha-mining-workbench
```

2. Install the required dependencies:
```bash
pip install -r python/gui/requirements.txt
```

## Usage

To run the application:

```bash
cd python/gui
python alpha_miner_app.py
```

## Application Structure

- `alpha_miner_app.py`: Main GUI application
- `alpha_mining.py`: Core mining algorithms
- `config.py`: Configuration settings
- `results_manager.py`: Results storage and retrieval

## Configuration

The application settings can be configured via the Settings tab in the GUI or by editing the `config.json` file that is generated when the application is run.

## Development

To extend the application:

1. Add new mining strategies in `alpha_mining.py`
2. Extend the GUI in `alpha_miner_app.py`
3. Add new configuration options in `config.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 