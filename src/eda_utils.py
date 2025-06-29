import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data(file_path):
    """Load and preprocess raw transaction data"""
    df = pd.read_csv(file_path, parse_dates=['TransactionStartTime'])
    print(f"✅ Data loaded successfully with {len(df)} records")
    return df


def dataset_overview(df):
    """
    Generate comprehensive overview of dataset structure

    Parameters:
    df (DataFrame): Input dataframe

    Returns:
    dict: Dictionary containing overview metrics
    """
    overview = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'data_types': df.dtypes,
        'missing_values': df.isna().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 ** 2)  # in MB
    }

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"📊 Total Records: {overview['shape'][0]:,}")
    print(f"📋 Features: {overview['shape'][1]}")
    print(f"🧠 Memory Usage: {overview['memory_usage']:.2f} MB")
    print(f"🔍 Duplicates: {overview['duplicates']}")
    print("\n📝 Data Types:")
    print(overview['data_types'])

    if overview['duplicates'] > 0:
        print("\n⚠️ Warning: Duplicates detected - consider cleaning")

    return overview


def generate_summary_stats(df):
    """
    Generate detailed summary statistics with enhanced formatting

    Parameters:
    df (DataFrame): Input dataframe

    Returns:
    dict: Dictionary containing numerical and categorical summaries
    """
    stats = {}

    # Numerical features analysis
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats['numerical'] = df[num_cols].describe(percentiles=[.01, .25, .5, .75, .99]).T
    stats['numerical']['skewness'] = df[num_cols].skew()
    stats['numerical']['kurtosis'] = df[num_cols].kurtosis()

    # Categorical features analysis
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    stats['categorical'] = pd.DataFrame({
        'unique': df[cat_cols].nunique(),
        'top': df[cat_cols].apply(lambda x: x.mode()[0]),
        'freq': df[cat_cols].apply(lambda x: x.value_counts().iloc[0]),
        'missing': df[cat_cols].isna().sum()
    })

    # Display results
    print("\n" + "=" * 60)
    print("NUMERICAL FEATURE STATISTICS")
    print("=" * 60)
    print(stats['numerical'])

    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURE STATISTICS")
    print("=" * 60)
    print(stats['categorical'])

    return stats


