# Quantum-Enhanced Alpha Mining: Complete Development Guide

This book provides a comprehensive guide to building quantum-enhanced alpha mining systems, from Generation One to Generation Two, including Mini-WorldQuant full-lifecycle platform.

## Structure

The book is organized as a single `main.tex` file with chapters in the `chapters/` subfolder:

```
paper/
├── main.tex                          # Main book file
├── chapters/
│   ├── generation-one-architecture.tex      # Generation One analysis
│   ├── generation-two-improvements.tex      # Self-optimization & genetic algorithms
│   ├── quantum-computing-integration.tex    # Quantum computing integration
│   ├── mt5-expert-advisor-experience.tex    # MT5 EA development patterns
│   └── mini-worldquant-system.tex          # Full lifecycle platform
└── README.md                         # This file
```

## Compilation

### Prerequisites

Install LaTeX distribution:
- **Windows**: MiKTeX or TeX Live
- **macOS**: MacTeX
- **Linux**: `sudo apt-get install texlive-full` (or equivalent)

### Compile the Book

```bash
cd paper
pdflatex main.tex
pdflatex main.tex  # Run twice for references
makeindex main.idx  # Generate index (if using \printindex)
pdflatex main.tex   # Final pass
```

Or use a build script:

```bash
# Windows PowerShell
.\build.ps1

# Linux/macOS
./build.sh
```

## Book Contents

### Part I: Generation One Architecture
- Complete analysis of `consultant-templates-ollama` system
- Architecture, modules, and code snippets
- Multi-arm bandit, persona system, concurrent execution
- Limitations and areas for improvement

### Part II: Generation Two Improvements
- Self-optimization system
- Genetic algorithm-based evolution
- On-the-fly testing
- Alpha quality tracking over time
- Expected performance improvements

### Part III: Advanced Technologies
- Quantum computing integration (QAOA, VQE)
- Hybrid quantum-classical framework
- MT5 Expert Advisor development patterns
- Real-time execution architectures

### Part IV: Mini-WorldQuant System
- Complete full-lifecycle platform
- Quant research module
- Data gathering engine
- Alpha backtesting system
- Alpha pool storage
- Trading algorithm engine
- Broker access layer
- Web-based cockpit

## Key Features Documented

1. **Generation One**: Complete recreation guide with all code snippets
2. **Generation Two**: Self-optimization and genetic evolution
3. **Quantum Integration**: QAOA and VQE for alpha discovery
4. **MT5 Patterns**: Expert Advisor development experience
5. **Mini-WorldQuant**: End-to-end trading platform

## Code Examples

All code examples are provided in:
- Python (for alpha mining systems)
- MQL5 (for MT5 Expert Advisors)
- HTML/JavaScript (for web cockpit)

## Figures

The book references several figures. Create placeholder images or use actual diagrams:
- `architecture-gen1.png` - Generation One architecture
- `quantum-hybrid-arch.png` - Quantum-classical architecture
- `miniwq-architecture.png` - Mini-WorldQuant architecture

## Customization

To customize the book:
1. Edit `main.tex` to modify structure
2. Edit individual chapter files in `chapters/`
3. Adjust styling in the preamble of `main.tex`

## Contributing

When adding new content:
1. Create new chapter file in `chapters/`
2. Add `\input{chapters/new-chapter.tex}` to `main.tex`
3. Update table of contents if needed
4. Recompile to verify

## License

This documentation is provided for research and development purposes.

