# FinanceViz - Investment Portfolio Simulator

A Python-based investment portfolio simulator that analyzes and visualizes different investment strategies focusing on SCHD (Schwab U.S. Dividend Equity ETF) and VOO (Vanguard S&P 500 ETF).

## Features

- Simulates long-term investment strategies with monthly contributions
- Analyzes dividend reinvestment strategies
- Optimizes portfolio allocation between SCHD and VOO
- Visualizes investment growth over time
- Supports parameter optimization for maximum returns
- Generates comparative subplot analysis for different investment strategies

## Requirements

```bash
pip install yfinance pandas numpy matplotlib scipy
```

## Usage

```python
from ClaudeModel import InvestmentSimulator

# Initialize simulator
simulator = InvestmentSimulator(
    monthly_investment=1000,        # Monthly investment amount
    simulation_years=40,            # Simulation duration
    dividend_reinvestment_percent=0.5,  # % of dividends to reinvest in SCHD
    dividend_income_investment_percent=0.5  # % of monthly investment for SCHD
)

# Run optimization
results = simulator.optimize_percentages()

# Visualize different strategies
simulator.iterate_over_percentages()
```

## Key Parameters

- `monthly_investment`: Amount invested monthly
- `simulation_years`: Duration of investment simulation
- `dividend_reinvestment_percent`: Percentage of dividends reinvested in SCHD
- `dividend_income_investment_percent`: Percentage of monthly investment allocated to SCHD

## Analysis Features

1. **Portfolio Optimization**
   - Finds optimal allocation percentages
   - Uses scipy's L-BFGS-B optimization algorithm

2. **Visualization**
   - Investment growth over time
   - Comparative analysis of different strategies
   - Grid visualization of multiple scenarios

3. **Performance Metrics**
   - Total portfolio value
   - Individual fund performance (SCHD vs VOO)
   - Total invested amount tracking

## Example Output

The simulator provides both numerical results and visual representations:
- Optimization results showing best parameter combinations
- Growth charts comparing different investment strategies
- Grid analysis of various percentage combinations

## Conclusions

### Investment Strategy Insights
- Higher allocation to SCHD (dividend-focused ETF) tends to perform better in the long term due to dividend reinvestment compounding
- Optimal dividend reinvestment typically favors higher percentages (>75%) reinvested back into SCHD
- The power of dividend reinvestment becomes more apparent in longer simulation periods (20+ years)

### Performance Characteristics
- Initial years show minimal difference between strategies
- Dividend reinvestment impact becomes exponential after 10-15 years
- Portfolio value growth accelerates faster with higher dividend reinvestment rates
- Market volatility impact is reduced through consistent monthly investments

### Key Takeaways
- Long-term dividend-focused strategy can outperform pure S&P 500 investment
- Consistent monthly investments are crucial for portfolio growth
- Optimal strategy suggests balancing between:
  - High dividend reinvestment (capitalizing on compound growth)
  - Strategic allocation between growth (VOO) and dividend (SCHD) funds

### Limitations
- Past performance doesn't guarantee future results
- Model assumes consistent dividend payments
- Does not account for tax implications
- Limited to historical data availability of SCHD (launched in 2011)

## License

MIT License

## Contributing

Feel free to open issues and pull requests for improvements or bug fixes.