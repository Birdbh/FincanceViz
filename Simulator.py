import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from scipy.optimize import minimize

class InvestmentSimulator:
    def __init__(self, 
                 monthly_investment, 
                 simulation_years, 
                 dividend_reinvestment_percent,
                 dividend_income_investment_percent,
                 start_date=pd.Timestamp(datetime(2010, 1, 1), tz='America/New_York')):
        """
        Initialize investment simulation parameters
        
        :param monthly_investment: Amount invested monthly
        :param simulation_years: Total years to simulate
        :param dividend_reinvestment_percent: Percentage of dividends to reinvest in SCHD
        :param start_date: Starting date of simulation
        """
        self.monthly_investment = monthly_investment
        self.simulation_years = simulation_years
        self.dividend_reinvestment_percent = dividend_reinvestment_percent
        self.dividend_income_investment_percent = dividend_income_investment_percent
        self.start_date = start_date
        
        # Fetch historical data for SCHD and VOO
        self.schd_data = self.fetch_stock_data('SCHD', start_date, simulation_years)
        self.voo_data = self.fetch_stock_data('VOO', start_date, simulation_years)
        
    def fetch_stock_data(self, ticker, start_date, years):
        """
        Fetch historical stock data for a given ticker
        
        :param ticker: Stock ticker symbol
        :param start_date: Start date of data retrieval
        :param years: Number of years of data to retrieve
        :return: Pandas DataFrame with stock data
        """
        end_date = start_date + pd.DateOffset(years=years)
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        # Fetch dividend history
        dividends = stock.dividends
        dividends = dividends[(dividends.index >= start_date) & (dividends.index <= end_date)]
        dividends = dividends.tz_convert('America/New_York')
        
        return {
            'prices': data['Close'],
            'dividends': dividends
        }
    
    def simulate_investment(self):
        """
        Simulate investment strategy over specified years
        
        :return: Dictionary with investment results
        """
        # Initialize tracking variables
        schd_shares = 0
        voo_shares = 0
        total_invested = 0
        
        # Prepare results storage
        monthly_results = []
        
        # Iterate through each month of simulation
        current_date = self.start_date
        end_date = pd.Timestamp(self.start_date + pd.DateOffset(years=self.simulation_years))
        
        while current_date < end_date:
            # Determine stock prices for current month
            try:
                schd_price = self.schd_data['prices'].loc[current_date:].iloc[0]
                voo_price = self.voo_data['prices'].loc[current_date:].iloc[0]
            except IndexError:
                break
            
            # Buy shares of SCHD and VOO with monthly investment
            schd_monthly_investment = self.monthly_investment * self.dividend_income_investment_percent
            voo_monthly_investment = self.monthly_investment * (1 - self.dividend_income_investment_percent)
            
            schd_shares_bought = schd_monthly_investment / schd_price
            voo_shares_bought = voo_monthly_investment / voo_price
            
            schd_shares += schd_shares_bought
            voo_shares += voo_shares_bought
            total_invested += self.monthly_investment
            total_value = schd_shares * schd_price + voo_shares * voo_price
            
            # Check for dividends
            schd_dividends = self.schd_data['dividends'].loc[
                (self.schd_data['dividends'].index.year == current_date.year) & 
                (self.schd_data['dividends'].index.month == current_date.month)
            ]
            
            for dividend_date, dividend_amount in schd_dividends.items():
                # Total dividends from SCHD shares
                total_dividend = schd_shares * dividend_amount
                
                # Reinvest portion of dividends in SCHD
                schd_dividend_reinvest = total_dividend * self.dividend_reinvestment_percent
                schd_dividend_invest_shares = schd_dividend_reinvest / schd_price
                
                # Invest remaining dividends in VOO
                voo_dividend_invest = total_dividend * (1 - self.dividend_reinvestment_percent)
                voo_dividend_invest_shares = voo_dividend_invest / voo_price
                
                schd_shares += schd_dividend_invest_shares
                voo_shares += voo_dividend_invest_shares
            
            # Store monthly results
            monthly_results.append({
                'date': current_date,
                'schd_shares': schd_shares,
                'voo_shares': voo_shares,
                'total_invested': total_invested,
                'schd_value': schd_shares * schd_price,
                'voo_value': voo_shares * voo_price,
                'total_value': total_value
            })
            
            # Move to next month
            current_date += pd.DateOffset(days=30)
        
        return {
            'monthly_results': monthly_results,
            'final_schd_shares': schd_shares,
            'final_voo_shares': voo_shares,
            'total_invested': total_invested,
            'total_value': total_value
        }
    
    def evaluate_portfolio(self, percentages):
        """
        Evaluate portfolio value for given percentage combinations
        
        :param percentages: Array of [dividend_reinvestment_percent, dividend_income_investment_percent]
        :return: Negative total portfolio value (for minimization)
        """
        # Ensure percentages are within bounds [0,1]
        dividend_reinvestment_percent = np.clip(percentages[0], 0, 1)
        dividend_income_investment_percent = np.clip(percentages[1], 0, 1)
        
        # Update simulator parameters
        self.dividend_reinvestment_percent = dividend_reinvestment_percent
        self.dividend_income_investment_percent = dividend_income_investment_percent
        
        # Run simulation
        results = self.simulate_investment()
        
        # Return negative value for minimization
        return -results['total_value']

    def optimize_percentages(self):
        """
        Find optimal percentage values using scipy.optimize
        
        :return: Dictionary with optimal values and results
        """
        initial_guess = [1, 1]  # Start with 50-50 split
        bounds = [(0, 1), (0, 1)]   # Percentages must be between 0 and 1
        
        # Run optimization
        result = minimize(
            self.evaluate_portfolio,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )

        print(result.x)
        print(result)
        
        optimal_percentages = np.clip(result.x, 0, 1)
        
        # Run final simulation with optimal values
        self.dividend_reinvestment_percent = optimal_percentages[0]
        self.dividend_income_investment_percent = optimal_percentages[1]
        final_results = self.simulate_investment()
        
        return {
            'optimal_dividend_reinvestment': optimal_percentages[0],
            'optimal_income_investment': optimal_percentages[1],
            'optimal_total_value': final_results['total_value'],
            'simulation_results': final_results
        }
    
    def plot_simulated_portfolio(self, results, divident_reinvestment_percent, dividend_income_investment_percent):
        """
        Plot percentages as a title and the simulated portfolio
        """
        monthly_results = results['monthly_results']
        
        dates = [pd.Timestamp(result['date'])  for result in monthly_results]
        schd_values = [result['schd_value'] for result in monthly_results]
        voo_values = [result['voo_value'] for result in monthly_results]
        total_invested = [result['total_invested'] for result in monthly_results]
        total_value = [result['total_value'] for result in monthly_results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, schd_values, label='SCHD Portfolio')
        plt.plot(dates, voo_values, label='VOO Portfolio')
        plt.plot(dates, total_value, label='Total Portfolio Value')
        plt.plot(dates, total_invested, label='Total Invested')
        
        plt.title(f'Optimal Investment Growth Over Time\nDividend Reinvestment: {divident_reinvestment_percent:.2%}, Dividend Income Investment: {dividend_income_investment_percent:.2%}, Total Portfolio Value: ${results['total_value']:,.2f}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def iterate_over_percentages(self):
        """
        Iterate over all possible percentage combinations and display as subplots
        """
        dividend_reinvestment_percents = np.linspace(0, 1, 3)
        dividend_income_investment_percents = np.linspace(0, 1, 3)
        
        # Calculate grid dimensions
        n_plots = len(dividend_reinvestment_percents) * len(dividend_income_investment_percents)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure and subplots
        fig = plt.figure(figsize=(10, 3 * n_rows))
        
        for i, reinvest_pct in enumerate(dividend_reinvestment_percents):
            for j, income_pct in enumerate(dividend_income_investment_percents):
                plot_num = i * len(dividend_income_investment_percents) + j + 1
                ax = fig.add_subplot(n_rows, n_cols, plot_num)
                
                simulator = InvestmentSimulator(
                    monthly_investment=1000,
                    simulation_years=40,
                    dividend_reinvestment_percent=reinvest_pct,
                    dividend_income_investment_percent=income_pct
                )
                results = simulator.simulate_investment()
                
                # Plot data
                monthly_results = results['monthly_results']
                dates = [pd.Timestamp(result['date']) for result in monthly_results]
                schd_values = [result['schd_value'] for result in monthly_results]
                voo_values = [result['voo_value'] for result in monthly_results]
                total_value = [result['total_value'] for result in monthly_results]
                total_invested = [result['total_invested'] for result in monthly_results]
                
                ax.plot(dates, schd_values, label='SCHD')
                ax.plot(dates, voo_values, label='VOO')
                ax.plot(dates, total_value, label='Total')
                ax.plot(dates, total_invested, label='Invested')
                
                ax.set_title(f'Reinvestment: {reinvest_pct:.0%}, Dividend Allocation: {income_pct:.0%}\nValue: ${results["total_value"]:,.0f}')
                ax.tick_params(axis='x', rotation=45)
                if plot_num == 1:  # Only show legend on first subplot
                    ax.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Simulation parameters
    monthly_investment = 1000  # $500 monthly investment
    simulation_years = 40     # 40-year simulation
    # dividend_reinvestment_percent = 0.5  # 50% of dividends reinvested in SCHD
    # dividend_income_investment_percent = 0.5  # 50% of income invested into SCHD
    
    # Initialize simulator with arbitrary percentages (will be optimized)
    simulator = InvestmentSimulator(
        monthly_investment,
        simulation_years,
        dividend_reinvestment_percent=0.5,
        dividend_income_investment_percent=0.5
    )
    
    # Run optimization
    optimization_results = simulator.optimize_percentages()

    simulator.iterate_over_percentages()
    
    # Print optimization results
    print("\nOptimization Results:")
    print(f"Optimal Dividend Reinvestment: {optimization_results['optimal_dividend_reinvestment']:.2%}")
    print(f"Optimal Income Investment: {optimization_results['optimal_income_investment']:.2%}")
    print(f"Optimal Total Value: ${optimization_results['optimal_total_value']:,.2f}")
    
    # Plot results with optimal values
    simulator.plot_simulated_portfolio(optimization_results['simulation_results'], optimization_results['optimal_dividend_reinvestment'], optimization_results['optimal_income_investment'])

if __name__ == "__main__":
    main()