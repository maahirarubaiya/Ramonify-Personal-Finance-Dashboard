# Ramonify — Personal Finance Dashboard

A data-driven personal finance dashboard designed to help students understand their spending patterns through interactive visualizations, machine learning-powered insights, and scenario planning tools.

## Overview

Ramonify analyzes transaction-level data to provide accessible financial insights in plain language. The application focuses on financial literacy for first-generation and student users by combining automated categorization, predictive forecasting, and what-if scenario simulation to help users make informed budgeting decisions.

## Features

### Core Analytics
- **Monthly Financial Summary**: Track income, expenses, and net cash flow across multiple months
- **Category Analysis**: Identify top spending categories with detailed breakdowns by month
- **Interactive Visualizations**: Explore trends with zoomable, hoverable Plotly charts
- **Cash Flow Forecasting**: Predict next month's financial position using rolling averages and linear regression

### Machine Learning
- **Automated Transaction Categorization**: ML classifier using TF-IDF vectorization and logistic regression to automatically categorize uncategorized transactions based on merchant names
- **Model Performance Tracking**: Real-time accuracy metrics displayed to users

### Planning Tools
- **Overspending Alerts**: Customizable warning system comparing current spending to historical averages
- **Scenario Simulation**: Test what-if scenarios by adjusting spending categories and income to see projected outcomes
- **Goal Tracking**: Set monthly savings goals and receive personalized recommendations to achieve them
- **Actionable Tips**: Plain-English budgeting advice based on spending patterns

### Data Export
- Download monthly summaries, category breakdowns, and forecast results as CSV files for further analysis

## Tech Stack

**Backend**: Python, pandas, NumPy, scikit-learn  
**Visualization**: Plotly  
**Interface**: Gradio  
**Testing**: pytest  
**Machine Learning**: TF-IDF Vectorization, Logistic Regression

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ramonify.git
cd ramonify
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python ramonify.py
```

4. Open your browser to the provided local URL (typically `http://localhost:7860`)

## Usage

### Input Data Format

Upload a CSV file with the following required columns:
- `date`: Transaction date in YYYY-MM-DD format
- `amount`: Transaction amount as a number
- `category`: Spending category (e.g., "Groceries", "Dining")
- `merchant`: Merchant or payee name
- `type`: Either "income" or "expense"

Example:
```csv
date,amount,category,merchant,type
2024-01-15,45.50,Groceries,Whole Foods,expense
2024-01-20,2000.00,Salary,Employer,income
2024-01-22,12.99,Dining,Chipotle,expense
```

### Using the Dashboard

1. **Upload Transactions**: Use the file upload interface to load your CSV
2. **Configure Settings**: 
   - Adjust overspending threshold (10-50%)
   - Set tip percentage for budget recommendations
   - Enter monthly savings goal
3. **Run Scenario Simulations**:
   - Adjust top spending categories by percentage
   - Add projected additional income
   - View impact on financial forecast
4. **Review Insights**: 
   - Check spending alerts and warnings
   - Read personalized tips
   - Download detailed reports

### Automated Categorization

The system automatically categorizes transactions with missing categories if at least 10 categorized transactions are present in your dataset. The ML model learns from your existing categorization patterns and applies them to new transactions.

## Project Structure

```
ramonify/
├── ramonify.py              # Main application code
├── test_ramonify.py         # Comprehensive unit tests
├── requirements.txt         # Python dependencies
├── transactions.csv         # Sample transaction data
└── README.md               # Project documentation
```

## Testing

Run the test suite to verify functionality:

```bash
pytest test_ramonify.py -v
```

The test suite includes:
- Data validation and edge case handling
- Monthly summary calculation accuracy
- ML classifier training and prediction
- Forecasting algorithm verification
- Integration tests for complete workflows

Test coverage includes 15+ test cases covering critical functionality and edge cases.

## Design Philosophy

Ramonify was built with the following principles:

**Accessibility First**: Financial insights presented in simple, jargon-free language that anyone can understand, regardless of financial literacy background.

**Student-Centered**: Designed specifically for college students and first-generation users who may be managing finances independently for the first time.

**Actionable Insights**: Every analysis includes concrete, personalized recommendations users can implement immediately.

**Data Privacy**: All processing happens locally; no transaction data is stored or transmitted to external servers.

## Future Enhancements

Potential features for future development:
- Multi-user support with data persistence
- Budget templates for common student scenarios
- Integration with bank APIs for automatic transaction import
- Mobile-responsive interface
- Comparative benchmarks against peer spending patterns
- PDF report generation

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

## License

This project is available under the MIT License.

## Contact

For questions, feedback, or collaboration inquiries, please open an issue on GitHub.

## Acknowledgments

Built to address financial literacy gaps among student populations, with a focus on making personal finance accessible and actionable for users who may be managing money independently for the first time.
