# -*- coding: utf-8 -*-
"""Budget Tracker Application - User-Friendly GUI Version"""

# Import dependencies
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

# User class - Encapsulated financial details with clear methods for operations
class User:
    """Class to represent a user with financial details."""
    def __init__(self, name):
        """Initialize user with name and financial attributes."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Name must be a non-empty string.")
        self._name = name.strip()
        self._total_income = 0.0
        self._total_expenses = 0.0
        self._savings_goal = 0.0

    def add_income(self, amount):
        """Add income to the user's total income."""
        if not isinstance(amount, (int, float)) or amount <= 0:
            return False
        self._total_income += amount
        return True

    def add_expense(self, amount):
        """Add expense to the user's total expenses."""
        if not isinstance(amount, (int, float)) or amount <= 0 or amount > self._total_income:
            return False
        self._total_expenses += amount
        return True

    def update_savings_goal(self, goal):
        """Update the user's savings goal."""
        if not isinstance(goal, (int, float)) or goal < 0:
            return False
        self._savings_goal = goal
        return True

    @property
    def name(self):
        return self._name

    @property
    def total_income(self):
        return self._total_income

    @property
    def total_expenses(self):
        return self._total_expenses

    @property
    def savings_goal(self):
        return self._savings_goal

# Transaction class - Encapsulated transaction data with validation for consistency
class Transaction:
    """Class to represent a financial transaction."""
    def __init__(self, amount, category, transaction_type):
        """Initialize transaction with amount, category, and type."""
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("Amount must be positive.")
        if transaction_type not in ["income", "expense"]:
            raise ValueError("Transaction type must be 'income' or 'expense'.")
        self._amount = float(amount)
        self._category = category.strip() if isinstance(category, str) else "General"
        self._transaction_type = transaction_type
        self._date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def record_transaction(self):
        """Record the transaction (returns True if valid)."""
        return self._amount > 0

    def get_details(self):
        """Return transaction details as a string."""
        return (f"Type: {self._transaction_type}, Amount: ${self._amount:.2f}, "
                f"Category: {self._category}, Date: {self._date}")

    @property
    def amount(self):
        return self._amount

    @property
    def category(self):
        return self._category

# BudgetManager class - Manages budget with composition of User and Transaction
class BudgetManager:
    """Class to manage the user's budget and transactions."""
    def __init__(self, user):
        """Initialize with a user and an empty transaction list."""
        if not isinstance(user, User):
            raise ValueError("User must be an instance of User class.")
        self._user = user
        self._transactions = []

    def calculate_balance(self):
        """Calculate the remaining balance."""
        return self._user.total_income - self._user.total_expenses

    def add_transaction(self, transaction):
        """Add a transaction if valid."""
        if not isinstance(transaction, Transaction) or not transaction.record_transaction():
            return False
        self._transactions.append(transaction)
        if transaction._transaction_type == "income":
            return self._user.add_income(transaction.amount)
        return self._user.add_expense(transaction.amount)

    def check_savings_progress(self):
        """Check progress toward savings goal."""
        if self._user.savings_goal <= 0:
            return 0.0
        balance = self.calculate_balance()
        return min(balance / self._user.savings_goal * 100, 100)

    def generate_report(self):
        """Generate a summary report of transactions and balance."""
        try:
            report = f"\nBudget Report for {self._user.name}:\n"
            report += f"Total Income: ${self._user.total_income:.2f}\n"
            report += f"Total Expenses: ${self._user.total_expenses:.2f}\n"
            report += f"Remaining Balance: ${self.calculate_balance():.2f}\n"
            report += f"Savings Goal: ${self._user.savings_goal:.2f} "
            report += f"({self.check_savings_progress():.1f}% achieved)\n"
            report += "Transactions:\n"
            if not self._transactions:
                report += "- No transactions recorded.\n"
            for t in self._transactions:
                report += f"- {t.get_details()}\n"
            return report
        except Exception as e:
            return f"Error generating report: {str(e)}"

# ReportGenerator class - Handles reports and visualizations with clear feedback
class ReportGenerator:
    """Class to generate visual and file-based reports."""
    def __init__(self, budget_manager):
        """Initialize with a BudgetManager instance."""
        if not isinstance(budget_manager, BudgetManager):
            raise ValueError("BudgetManager must be an instance of BudgetManager class.")
        self._budget_manager = budget_manager
        self._category_totals = {}

    def create_pie_chart(self):
        """Create a pie chart of expense categories."""
        self._category_totals.clear()
        for t in self._budget_manager._transactions:
            if t._transaction_type == "expense":
                self._category_totals[t.category] = self._category_totals.get(t.category, 0) + t.amount

        if not self._category_totals:
            return "No expenses to visualize."

        plt.figure(figsize=(8, 6))
        plt.pie(self._category_totals.values(), labels=self._category_totals.keys(), autopct='%1.1f%%')
        plt.title("Expense Distribution")
        plt.savefig("expense_pie_chart.png")
        plt.close()
        return "Pie chart saved as 'expense_pie_chart.png'"

    def export_to_file(self, filename="budget_report.txt"):
        """Export the report to a text file."""
        try:
            with open(filename, 'w') as f:
                f.write(self._budget_manager.generate_report())
            return f"Report exported to {filename}"
        except Exception as e:
            return f"Error exporting report: {str(e)}"

# InvestmentPortfolio class - Uses strategy pattern for consistent risk-based allocations
class InvestmentPortfolio:
    """Class to manage a user's investment portfolio."""
    def __init__(self, user_balance, risk_tolerance="moderate"):
        """Initialize portfolio with balance and risk tolerance."""
        if not isinstance(user_balance, (int, float)) or user_balance < 0:
            raise ValueError("Balance must be non-negative.")
        if risk_tolerance.lower() not in ["low", "moderate", "high"]:
            raise ValueError("Risk tolerance must be 'low', 'moderate', or 'high'.")
        self._balance = user_balance
        self._risk_tolerance = risk_tolerance.lower()
        self._allocations = {"stocks": 0.0, "bonds": 0.0, "cash": 0.0}
        self._set_default_allocation()

    def _set_default_allocation(self):
        """Set default allocation based on risk tolerance (strategy pattern)."""
        if self._risk_tolerance == "low":
            self._allocations = {"stocks": 0.3, "bonds": 0.5, "cash": 0.2}
        elif self._risk_tolerance == "moderate":
            self._allocations = {"stocks": 0.7, "bonds": 0.2, "cash": 0.1}
        else:  # high
            self._allocations = {"stocks": 0.9, "bonds": 0.0, "cash": 0.1}

    def calculate_investment_amounts(self):
        """Calculate dollar amounts for each asset class."""
        return {asset: amount * self._balance for asset, amount in self._allocations.items()}

    def get_portfolio_summary(self):
        """Return a summary of the portfolio."""
        amounts = self.calculate_investment_amounts()
        return (f"Portfolio for ${self._balance:.2f} with {self._risk_tolerance} risk:\n"
                f"- Stocks: ${amounts['stocks']:.2f} ({self._allocations['stocks']*100:.1f}%)\n"
                f"- Bonds: ${amounts['bonds']:.2f} ({self._allocations['bonds']*100:.1f}%)\n"
                f"- Cash: ${amounts['cash']:.2f} ({self._allocations['cash']*100:.1f}%)")

# InvestmentPredictor class - ML for efficient, error-reduced predictions
class InvestmentPredictor:
    """Class to predict optimal investment allocations using supervised learning."""
    def __init__(self):
        """Initialize with a pre-trained model or train on synthetic data."""
        self._model = LinearRegression()
        self._scaler = StandardScaler()
        self._train_model()

    def _train_model(self):
        """Train the model on synthetic data."""
        try:
            np.random.seed(42)
            n_samples = 100
            balance = np.random.uniform(1000, 5000, n_samples)
            savings_rate = np.random.uniform(0.1, 0.5, n_samples)
            risk_tolerance = np.random.randint(0, 3, n_samples)  # 0:low, 1:moderate, 2:high
            goal_amount = np.random.uniform(500, 2000, n_samples)

            allocation_stocks = np.where(risk_tolerance == 2, 0.9, np.where(risk_tolerance == 1, 0.7, 0.3))
            allocation_bonds = np.where(risk_tolerance == 0, 0.5, np.where(risk_tolerance == 1, 0.2, 0.0))
            allocation_cash = 1 - allocation_stocks - allocation_bonds

            X = np.column_stack((balance, savings_rate, risk_tolerance, goal_amount))
            y = np.column_stack((allocation_stocks, allocation_bonds, allocation_cash))

            X_scaled = self._scaler.fit_transform(X)
            self._model.fit(X_scaled, y)
        except Exception as e:
            return f"Error training ML model: {str(e)}"

    def predict_allocation(self, balance, savings_rate, risk_tolerance, goal_amount):
        """Predict optimal allocation based on user financial data."""
        try:
            risk_encoded = {"low": 0, "moderate": 1, "high": 2}[risk_tolerance.lower()]
            X_new = np.array([[balance, savings_rate, risk_encoded, goal_amount]])
            X_new_scaled = self._scaler.transform(X_new)
            allocations = self._model.predict(X_new_scaled)[0]
            allocations = np.maximum(allocations, 0)
            allocations /= allocations.sum() if allocations.sum() > 0 else 1
            return {"stocks": allocations[0], "bonds": allocations[1], "cash": allocations[2]}
        except Exception as e:
            return f"Error predicting allocation: {str(e)}"

    def get_prediction_summary(self, balance, savings_rate, risk_tolerance, goal_amount):
        """Return a summary of the predicted allocation."""
        try:
            alloc = self.predict_allocation(balance, savings_rate, risk_tolerance, goal_amount)
            total = balance
            return (f"Predicted Portfolio for ${balance:.2f}:\n"
                    f"- Stocks: ${total * alloc['stocks']:.2f} ({alloc['stocks']*100:.1f}%)\n"
                    f"- Bonds: ${total * alloc['bonds']:.2f} ({alloc['bonds']*100:.1f}%)\n"
                    f"- Cash: ${total * alloc['cash']:.2f} ({alloc['cash']*100:.1f}%)")
        except Exception as e:
            return f"Error generating prediction summary: {str(e)}"

# InvestmentAdvisor class - Integrates real-time data with ML for effective advice
class InvestmentAdvisor:
    """Class to provide personalized investment advice with real-time data and ML predictions."""
    def __init__(self, budget_manager, api_key="AR211YAAO9ZZJY7W"):
        """Initialize with a BudgetManager instance and API key."""
        if not isinstance(budget_manager, BudgetManager):
            raise ValueError("BudgetManager must be an instance of BudgetManager class.")
        self._budget_manager = budget_manager
        self._api_key = api_key
        self._ts = TimeSeries(key=self._api_key, output_format='pandas')
        self._predictor = InvestmentPredictor()
        self._historical_data = None
        self._real_time_data = None
        self._avg_returns = {"stocks": 0.08, "bonds": 0.03, "cash": 0.01}
        self._volatility = {"stocks": 0.15, "bonds": 0.05, "cash": 0.01}
        self._data_status = "Using fallback data (8% return, 15% volatility for stocks)."
        self._load_data()

    def _load_data(self):
        """Load historical and attempt real-time data for AAPL."""
        try:
            try:
                self._historical_data = pd.read_csv('aapl_historical.csv')
                self._historical_data['Date'] = pd.to_datetime(self._historical_data['Date'])
                self._calculate_historical_returns()
                self._data_status = "Loaded historical AAPL data."
            except FileNotFoundError:
                print("Historical data file 'aapl_historical.csv' not found.")

            data, _ = self._ts.get_intraday(symbol='AAPL', interval='1min', outputsize='compact')
            self._real_time_data = data
            self._calculate_real_time_metrics()
            self._data_status = "Loaded real-time AAPL data."
        except Exception as e:
            print(f"Error fetching real-time data: {e}. Using fallback data.")
            self._data_status = f"API error: {str(e)}. Using fallback data (8% return, 15% volatility)."

    def _calculate_historical_returns(self):
        """Calculate average annual returns and volatility from historical data."""
        if self._historical_data is not None and 'Adj Close' in self._historical_data.columns:
            df = self._historical_data.copy()
            df['Returns'] = df['Adj Close'].pct_change().dropna()
            if not df['Returns'].empty:
                self._avg_returns["stocks"] = df['Returns'].mean() * 252
                self._volatility["stocks"] = df['Returns'].std() * (252 ** 0.5)

    def _calculate_real_time_metrics(self):
        """Calculate metrics from the latest real-time data."""
        if self._real_time_data is not None and '4. close' in self._real_time_data.columns:
            returns = self._real_time_data['4. close'].pct_change().dropna()
            if len(returns) > 1:
                self._avg_returns["stocks"] = returns.mean() * (252 * 60)
                self._volatility["stocks"] = returns.std() * ((252 * 60) ** 0.5)

    def suggest_investment(self, goal_amount, risk_tolerance="moderate"):
        """Suggest an investment strategy with real-time data and ML predictions."""
        try:
            if not isinstance(goal_amount, (int, float)) or goal_amount <= 0:
                raise ValueError("Investment goal must be positive.")
            if risk_tolerance.lower() not in ["low", "moderate", "high"]:
                raise ValueError("Risk tolerance must be 'low', 'moderate', or 'high'.")

            balance = self._budget_manager.calculate_balance()
            if balance <= 0:
                return "No funds available for investment."

            invest_amount = min(balance, goal_amount)
            portfolio = InvestmentPortfolio(invest_amount, risk_tolerance)
            advice = f"Investment Recommendation (Balance: ${balance:.2f}, Goal: ${goal_amount:.2f}):\n"
            advice += portfolio.get_portfolio_summary()

            savings_rate = min(balance / goal_amount, 1.0) if goal_amount > 0 else 0.0
            ml_prediction = self._predictor.get_prediction_summary(balance, savings_rate, risk_tolerance, goal_amount)
            advice += f"\nML-Predicted Strategy:\n{ml_prediction}"

            if balance < goal_amount:
                advice += "\nWarning: Insufficient balance to meet investment goal."
            advice += f"\nExpected Annual Return (AAPL): {self._avg_returns['stocks']*100:.1f}%"
            advice += f"\nVolatility (Risk): {self._volatility['stocks']*100:.1f}%"
            advice += f"\n*** Data Status: {self._data_status} ***"
            return advice
        except Exception as e:
            return f"Error generating investment advice: {str(e)}. Ensure valid inputs and API key (line 320)."

    def optimize_allocation(self):
        """Optimize allocation using real-time returns and volatility."""
        try:
            balance = self._budget_manager.calculate_balance()
            if balance <= 0:
                return "No funds available to optimize."

            portfolio = InvestmentPortfolio(balance)
            current_alloc = portfolio._allocations
            risk_tolerance = portfolio._risk_tolerance

            stock_weight = self._avg_returns["stocks"] / self._volatility["stocks"]
            bond_weight = self._avg_returns["bonds"] / self._volatility["bonds"]
            cash_weight = self._avg_returns["cash"] / self._volatility["cash"]
            total_weight = stock_weight + bond_weight + cash_weight

            if risk_tolerance == "high":
                current_alloc["stocks"] = min(0.9, stock_weight / total_weight)
                current_alloc["bonds"] = max(0.0, bond_weight / total_weight)
                current_alloc["cash"] = max(0.1, 1 - current_alloc["stocks"] - current_alloc["bonds"])
            elif risk_tolerance == "low":
                current_alloc["stocks"] = max(0.3, stock_weight / total_weight * 0.5)
                current_alloc["bonds"] = min(0.5, bond_weight / total_weight * 1.5)
                current_alloc["cash"] = max(0.2, 1 - current_alloc["stocks"] - current_alloc["bonds"])
            else:  # moderate
                current_alloc["stocks"] = stock_weight / total_weight
                current_alloc["bonds"] = bond_weight / total_weight
                current_alloc["cash"] = 1 - current_alloc["stocks"] - current_alloc["bonds"]

            portfolio._allocations = current_alloc
            return portfolio.get_portfolio_summary()
        except Exception as e:
            return f"Error optimizing allocation: {str(e)}. Check data and inputs."

def main():
    """Main function to run the Budget Tracker application."""
    try:
        # Create a user
        user = User("Alice")
        budget_mgr = BudgetManager(user)
        report_gen = ReportGenerator(budget_mgr)
        advisor = InvestmentAdvisor(budget_mgr, api_key="NAK7VBRNTSWPE0CT")

        # Interactive loop
        while True:
            print("\nBudget Tracker Menu:")
            print("1. Add Income")
            print("2. Add Expense")
            print("3. Set Savings Goal")
            print("4. Generate Report")
            print("5. Create Pie Chart")
            print("6. Suggest Investment")
            print("7. Exit")

            choice = input("Enter your choice (1-7): ")

            if choice == "1":
                try:
                    amount = float(input("Enter income amount: $"))
                    t = Transaction(amount, "General", "income")
                    if budget_mgr.add_transaction(t):
                        print("Income added successfully.")
                    else:
                        print("Invalid amount.")
                except ValueError:
                    print("Please enter a valid number.")

            elif choice == "2":
                try:
                    amount = float(input("Enter expense amount: $"))
                    category = input("Enter expense category: ").strip()
                    if not category:
                        print("Category cannot be empty.")
                        continue
                    t = Transaction(amount, category, "expense")
                    if budget_mgr.add_transaction(t):
                        print("Expense added successfully.")
                    else:
                        print("Invalid amount or insufficient balance.")
                except ValueError:
                    print("Please enter a valid number.")

            elif choice == "3":
                try:
                    goal = float(input("Enter savings goal: $"))
                    if user.update_savings_goal(goal):
                        print("Savings goal updated.")
                    else:
                        print("Invalid goal amount.")
                except ValueError:
                    print("Please enter a valid number.")

            elif choice == "4":
                print(budget_mgr.generate_report())

            elif choice == "5":
                report_gen.create_pie_chart()

            elif choice == "6":
                try:
                    goal = float(input("Enter investment goal amount: $"))
                    risk = input("Enter risk tolerance (low/moderate/high): ").lower()
                    print(advisor.suggest_investment(goal, risk))
                except ValueError:
                    print("Please enter a valid number for the investment goal.")
                except Exception as e:
                    print(f"Error suggesting investment: {str(e)}. Check API key (line 320) or network.")

            elif choice == "7":
                print("Exiting Budget Tracker. Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

    except Exception as e:
        print(f"Application error: {str(e)}. Ensure dependencies are installed and API key is valid (line 320).")

if __name__ == "__main__":
    main()
