"""
Unit tests for Ramonify Personal Finance Dashboard
Run with: pytest test_ramonify.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import functions from main file (adjust import path as needed)
# from ramonify import (
#     train_category_classifier,
#     auto_categorize_transactions,
#     calculate_monthly_summary,
#     forecast_next_month
# )


# ---------- Test Fixtures ----------

@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    data = {
        'date': ['2024-01-01', '2024-01-05', '2024-01-10', '2024-02-01', '2024-02-15'],
        'amount': [100.00, 50.00, 200.00, 150.00, 75.00],
        'category': ['Groceries', 'Dining', 'Groceries', 'Transport', 'Dining'],
        'merchant': ['Whole Foods', 'Chipotle', 'Trader Joes', 'Uber', 'Starbucks'],
        'type': ['expense', 'expense', 'expense', 'expense', 'expense']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_with_income():
    """Create transaction data including income."""
    data = {
        'date': ['2024-01-01', '2024-01-05', '2024-01-15', '2024-02-01', '2024-02-15'],
        'amount': [100.00, 2000.00, 50.00, 2000.00, 75.00],
        'category': ['Groceries', 'Salary', 'Dining', 'Salary', 'Entertainment'],
        'merchant': ['Whole Foods', 'Employer', 'Chipotle', 'Employer', 'AMC'],
        'type': ['expense', 'income', 'expense', 'income', 'expense']
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df


@pytest.fixture
def large_categorized_dataset():
    """Create larger dataset for ML testing."""
    np.random.seed(42)
    merchants = ['Whole Foods', 'Trader Joes', 'Safeway'] * 5  # Groceries
    merchants += ['Chipotle', 'Panera', 'Subway'] * 5  # Dining
    merchants += ['Shell', 'Chevron', 'BP'] * 5  # Gas
    
    categories = ['Groceries'] * 15 + ['Dining'] * 15 + ['Gas'] * 15
    
    data = {
        'date': pd.date_range('2024-01-01', periods=45, freq='D'),
        'amount': np.random.uniform(10, 100, 45),
        'category': categories,
        'merchant': merchants,
        'type': ['expense'] * 45
    }
    df = pd.DataFrame(data)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df


# ---------- Test Data Validation ----------

def test_sample_transactions_structure(sample_transactions):
    """Test that sample fixture has correct structure."""
    assert len(sample_transactions) == 5
    assert list(sample_transactions.columns) == ['date', 'amount', 'category', 'merchant', 'type']
    assert sample_transactions['type'].unique().tolist() == ['expense']


def test_income_data_structure(sample_with_income):
    """Test that income fixture includes both income and expenses."""
    assert 'income' in sample_with_income['type'].values
    assert 'expense' in sample_with_income['type'].values
    assert len(sample_with_income) == 5


# ---------- Test Monthly Summary Calculation ----------

def test_calculate_monthly_summary_basic(sample_with_income):
    """Test basic monthly summary calculation."""
    from ramonify import calculate_monthly_summary
    
    result = calculate_monthly_summary(sample_with_income)
    
    # Check structure
    assert 'month' in result.columns
    assert 'income' in result.columns
    assert 'expense' in result.columns
    assert 'net' in result.columns
    
    # Check calculations for January
    jan_row = result[result['month'] == '2024-01'].iloc[0]
    assert jan_row['income'] == 2000.00
    assert jan_row['expense'] == 150.00  # 100 + 50
    assert jan_row['net'] == 1850.00


def test_monthly_summary_handles_no_income(sample_transactions):
    """Test monthly summary when there's no income."""
    from ramonify import calculate_monthly_summary
    
    df = sample_transactions.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    
    result = calculate_monthly_summary(df)
    
    assert result['income'].sum() == 0.0
    assert result['expense'].sum() > 0


def test_monthly_summary_rolling_average(sample_with_income):
    """Test that rolling average is calculated correctly."""
    from ramonify import calculate_monthly_summary
    
    # Add more months
    df = sample_with_income.copy()
    new_rows = []
    for month in range(3, 6):
        new_rows.append({
            'date': pd.Timestamp(f'2024-0{month}-01'),
            'amount': 2000.00,
            'category': 'Salary',
            'merchant': 'Employer',
            'type': 'income',
            'month': f'2024-0{month}'
        })
        new_rows.append({
            'date': pd.Timestamp(f'2024-0{month}-15'),
            'amount': 100.00,
            'category': 'Groceries',
            'merchant': 'Store',
            'type': 'expense',
            'month': f'2024-0{month}'
        })
    
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    result = calculate_monthly_summary(df)
    
    # Rolling average should exist and be calculated for months 3+
    assert 'rolling_avg_3' in result.columns
    assert not pd.isna(result['rolling_avg_3'].iloc[-1])


# ---------- Test ML Categorization ----------

def test_train_classifier_with_sufficient_data(large_categorized_dataset):
    """Test that classifier trains successfully with enough data."""
    from ramonify import train_category_classifier
    
    vectorizer, model, accuracy = train_category_classifier(large_categorized_dataset)
    
    assert vectorizer is not None
    assert model is not None
    assert accuracy > 0.5  # Should achieve reasonable accuracy
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(model, LogisticRegression)


def test_train_classifier_with_insufficient_data(sample_transactions):
    """Test that classifier returns None with too little data."""
    from ramonify import train_category_classifier
    
    # Only 5 transactions - not enough
    vectorizer, model, accuracy = train_category_classifier(sample_transactions)
    
    assert vectorizer is None
    assert model is None
    assert accuracy == 0.0


def test_auto_categorize_preserves_existing(large_categorized_dataset):
    """Test that auto-categorization doesn't overwrite existing categories."""
    from ramonify import train_category_classifier, auto_categorize_transactions
    
    df = large_categorized_dataset.copy()
    original_categories = df['category'].copy()
    
    vectorizer, model, _ = train_category_classifier(df)
    result = auto_categorize_transactions(df, vectorizer, model)
    
    # All original categories should be preserved
    assert (result['category'] == original_categories).all()


def test_auto_categorize_fills_missing():
    """Test that auto-categorization fills in missing categories."""
    from ramonify import train_category_classifier, auto_categorize_transactions
    
    # Create dataset with some missing categories
    data = {
        'date': pd.date_range('2024-01-01', periods=20, freq='D'),
        'amount': [50.0] * 20,
        'merchant': ['Whole Foods'] * 10 + ['Chipotle'] * 10,
        'category': ['Groceries'] * 5 + [None] * 5 + ['Dining'] * 5 + [None] * 5,
        'type': ['expense'] * 20
    }
    df = pd.DataFrame(data)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    
    vectorizer, model, _ = train_category_classifier(df)
    result = auto_categorize_transactions(df, vectorizer, model)
    
    # Should have fewer missing categories
    assert result['category'].isna().sum() < df['category'].isna().sum()


# ---------- Test Forecasting ----------

def test_forecast_returns_valid_numbers(sample_with_income):
    """Test that forecasting returns valid numeric predictions."""
    from ramonify import calculate_monthly_summary, forecast_next_month
    
    summary = calculate_monthly_summary(sample_with_income)
    rolling, regression = forecast_next_month(summary)
    
    assert isinstance(rolling, float)
    assert isinstance(regression, float)
    assert not np.isnan(rolling)
    assert not np.isnan(regression)


def test_forecast_with_single_month():
    """Test forecasting behavior with minimal data."""
    from ramonify import calculate_monthly_summary, forecast_next_month
    
    data = {
        'month': ['2024-01'],
        'income': [2000.0],
        'expense': [1500.0],
        'net': [500.0],
        'rolling_avg_3': [500.0]
    }
    summary = pd.DataFrame(data)
    
    rolling, regression = forecast_next_month(summary)
    
    # Should still return valid forecasts
    assert isinstance(rolling, float)
    assert isinstance(regression, float)


# ---------- Test Edge Cases ----------

def test_empty_dataframe_handling():
    """Test handling of empty dataframe."""
    from ramonify import calculate_monthly_summary
    
    empty_df = pd.DataFrame(columns=['month', 'type', 'amount'])
    result = calculate_monthly_summary(empty_df)
    
    # Should return valid structure even if empty
    assert 'income' in result.columns
    assert 'expense' in result.columns
    assert 'net' in result.columns


def test_negative_amounts():
    """Test handling of negative amounts (e.g., refunds)."""
    data = {
        'date': ['2024-01-01', '2024-01-05'],
        'amount': [100.00, -20.00],  # Refund
        'category': ['Groceries', 'Groceries'],
        'merchant': ['Store', 'Store'],
        'type': ['expense', 'expense'],
        'month': ['2024-01', '2024-01']
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    from ramonify import calculate_monthly_summary
    result = calculate_monthly_summary(df)
    
    # Net expense should be 100 - 20 = 80
    assert result['expense'].iloc[0] == 80.00


def test_same_day_multiple_transactions(sample_with_income):
    """Test aggregation of multiple transactions on same day."""
    from ramonify import calculate_monthly_summary
    
    # Add duplicate date
    df = sample_with_income.copy()
    new_row = {
        'date': pd.Timestamp('2024-01-01'),
        'amount': 50.00,
        'category': 'Dining',
        'merchant': 'Restaurant',
        'type': 'expense',
        'month': '2024-01'
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    result = calculate_monthly_summary(df)
    
    # Should properly aggregate both transactions
    jan_expense = result[result['month'] == '2024-01']['expense'].iloc[0]
    assert jan_expense == 200.00  # 100 + 50 + 50


# ---------- Integration Test ----------

def test_full_pipeline_integration(large_categorized_dataset):
    """Test complete workflow from raw data to forecasts."""
    from ramonify import (
        train_category_classifier,
        auto_categorize_transactions,
        calculate_monthly_summary,
        forecast_next_month
    )
    
    # Step 1: Train classifier
    vectorizer, model, accuracy = train_category_classifier(large_categorized_dataset)
    assert vectorizer is not None
    
    # Step 2: Categorize
    df = auto_categorize_transactions(large_categorized_dataset, vectorizer, model)
    assert df is not None
    
    # Step 3: Calculate summary
    summary = calculate_monthly_summary(df)
    assert len(summary) > 0
    
    # Step 4: Forecast
    rolling, regression = forecast_next_month(summary)
    assert isinstance(rolling, float)
    assert isinstance(regression, float)


# ---------- Run Tests ----------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
