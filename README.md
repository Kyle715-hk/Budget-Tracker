# Budget-Tracker
Budget Tracker App
A Python-based application for tracking finances, managing budgets, and providing ML-driven investment advice. This project showcases my data science skills in Python programming, SQL database management, machine learning (linear regression and ARIMA for forecasting), and data visualization, as applied in fintech contexts. Building on my EV Adoption analysis (k-means clustering and regression), it demonstrates practical data analytics for financial insights—relevant to the Rev-celerator Graduate Programme 2026: Data Scientist and Analyst at Revolut.
Features

Transaction Tracking: Add income and expenses with category and date logging.
Budget Management: Calculate balance, set savings goals, and track progress.
Reports and Visualizations: Generate text reports and pie charts for expense distribution.
Investment Advice: ML-based portfolio allocation using linear regression and ARIMA for return forecasting with real-time Alpha Vantage data.
ETL and SQL Queries: Load data from CSV and perform ad-hoc queries on transactions.
Forecast Returns: ARIMA for time-series prediction of returns.

ML Forecasting Examples

ARIMA for Expense Forecasting: Predicts future monthly expenses based on historical transactions. Example: If historical expenses are [100, 150, 120], ARIMA might forecast the next as 130 (adjusts for trends).
Linear Regression for Multivariate Forecasting: Predicts expenses based on factors like category count or month. Example: With training data [[month=1, categories=2, expense=100], [month=2, categories=3, expense=150]], it might predict expense=130 for month=3, categories=2.

Installation

Clone the repository:
textgit clone https://github.com/Kyle715-hk/Budget_Tracker-.git
cd Budget_Tracker-

Install dependencies:
textpip install -r requirements.txt

Set up environment variables (for API key):

Create .env in the root:
textALPHA_VANTAGE_API_KEY=your_key_here  # From https://www.alphavantage.co

Obtain a free API key from Alpha Vantage.



Running the App
Command-Line Menu Version (Primary)
This version provides a text-based interface for core functionality, ensuring stability and ease of use.

Run the app:
textpython app.py  # Or the main script file

Follow the prompt:
textBudget Tracker Menu:
1. Add Income
2. Add Expense
3. Set Savings Goal
4. Generate Report
5. Create Pie Chart
6. Suggest Investment
7. Exit
Enter your choice (1-7):

Example: Choose 1 to add income, then 2 for expenses (validation ensures positive amounts).




For recruiters: Interact with the command-line menu at this repository: Budget Tracker Command-Line Repo to see the app in action.

Note: If you encounter errors in the Streamlit version (e.g., invalid transaction or SQLite issues), use this command-line version as a fallback. These may occur due to thread safety or validation rules; fixed in local runs.
Streamlit Dashboard and GUI Version (Alternative)
This version provides an interactive web interface for easy use.

Run the app:
textstreamlit run app.py

Open the local URL (e.g., http://localhost:8501) in your browser.
Use the sidebar menu to add transactions, generate reports, etc.
Live Demo: Budget Tracker Streamlit App (deployed for easy access).

Troubleshooting

Invalid Transaction: Ensure amount > 0. For expenses, add income first (balance check). Update add_expense in code to relax if needed.
SQLite ProgrammingError: Use thread-safe connections in Streamlit (e.g., st.connection for SQLite).
API Errors: Verify your Alpha Vantage key is valid (rate-limited free tier).
Deployment on Streamlit Cloud: Set secrets in UI, use requirements.txt for dependencies.

License
MIT License. See LICENSE file for details.
This project highlights my proficiency in Python (Pandas, NumPy, Scikit-learn), SQL, ML (regression, ARIMA), and deployment, as per my resume. For recruiters: Contact wangtikchan715@gmail.com or LinkedIn (linkedin.com/in/wang-tik-chan) for discussions.
