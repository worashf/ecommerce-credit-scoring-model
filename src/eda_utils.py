import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
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


def plot_numerical_distributions(df, figsize=(18, 12)):
    """
    Visualize distributions of all numerical features with histograms and Q-Q plots

    Parameters:
    df (pd.DataFrame): Input dataframe
    figsize (tuple): Figure dimensions
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    plt.figure(figsize=figsize)
    for i, col in enumerate(num_cols):
        # Histogram with KDE
        plt.subplot(len(num_cols), 2, 2 * i + 1)
        sns.histplot(df[col], kde=True, stat='density')
        plt.title(f'Distribution of {col}')
        plt.axvline(df[col].mean(), color='r', linestyle='--', label='Mean')
        plt.axvline(df[col].median(), color='g', linestyle='-', label='Median')
        plt.legend()

        # Q-Q Plot
        plt.subplot(len(num_cols), 2, 2 * i + 2)
        stats.probplot(df[col].dropna(), plot=plt)
        plt.title(f'Q-Q Plot of {col}')

    plt.tight_layout()
    plt.show()

    # Skewness report
    skewness = df[num_cols].skew().sort_values(ascending=False)
    return pd.DataFrame({'Skewness': skewness}).style.background_gradient(cmap='RdBu')


def analyze_categoricals(df, max_categories=10, figsize=(15, 10)):
    """
    Analyze categorical features with frequency plots and statistics

    Parameters:
    df (pd.DataFrame): Input dataframe
    max_categories (int): Max categories to show per feature
    figsize (tuple): Figure dimensions
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    plt.figure(figsize=figsize)
    for i, col in enumerate(cat_cols):
        # Value counts with normalization
        counts = df[col].value_counts(normalize=True).head(max_categories)

        # Bar plot
        plt.subplot(len(cat_cols), 1, i + 1)
        counts.plot(kind='bar')
        plt.title(f'Distribution of {col} (Top {max_categories})')
        plt.ylabel('Proportion')

        # Annotate with percentages
        for p in plt.gca().patches:
            plt.gca().annotate(f'{p.get_height():.1%}',
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 5),
                               textcoords='offset points')

    plt.tight_layout()
    plt.show()

    # Cardinality report
    cardinality = df[cat_cols].nunique().sort_values(ascending=False)
    return pd.DataFrame({'Unique Values': cardinality}).style.background_gradient(cmap='viridis')


def analyze_correlations(df, method='pearson', threshold=0.7, figsize=(12, 8)):
    """
    Analyze feature correlations with heatmap and clustered visualization

    Parameters:
    df (pd.DataFrame): Input dataframe
    method (str): Correlation method ('pearson', 'spearman', 'kendall')
    threshold (float): Highlight correlations above this value
    figsize (tuple): Figure dimensions
    """
    # Calculate correlations
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr(method=method)

    # Plot heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title(f'{method.title()} Correlation Matrix')
    plt.show()

    # Identify strong correlations
    strong_corrs = (corr_matrix.abs() >= threshold) & (corr_matrix.abs() < 1.0)
    strong_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                    for i, j in zip(*np.where(strong_corrs))]

    return pd.DataFrame(strong_pairs, columns=['Feature 1', 'Feature 2', 'Correlation']).sort_values('Correlation', ascending=False)



def analyze_missingness(df, figsize=(10, 6)):
    """
    Analyze missing values patterns with visualization and statistics

    Parameters:
    df (pd.DataFrame): Input dataframe
    figsize (tuple): Figure dimensions
    """
    # Calculate missing stats
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.concat([missing, missing_pct], axis=1)
    missing_df.columns = ['Missing Count', 'Missing %']
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if not missing_df.empty:
        # Bar plot of missing values
        plt.figure(figsize=figsize)
        missing_df['Missing %'].plot(kind='barh')
        plt.title('Percentage of Missing Values by Feature')
        plt.xlabel('Percentage Missing')
        plt.ylabel('Feature')
        plt.xlim(0, 100)
        plt.show()

        # Display missing data statistics

        plt.title('Missing Data Patterns')
        plt.show()


def detect_outliers(df, threshold=3, figsize=(15, 8)):
    """
    Detect and visualize outliers using multiple methods

    Parameters:
    df (pd.DataFrame): Input dataframe
    threshold (float): Z-score threshold for outlier detection
    figsize (tuple): Figure dimensions
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Initialize results storage
    outliers = pd.DataFrame(columns=['Feature', 'Outlier Count', 'Outlier %'])

    plt.figure(figsize=figsize)
    for i, col in enumerate(num_cols):
        # Calculate outliers using Z-score
        z = np.abs(stats.zscore(df[col].dropna()))
        outlier_mask = z > threshold
        outlier_pct = (outlier_mask.sum() / len(df[col].dropna())) * 100

        # Store results
        outliers.loc[i] = [col, outlier_mask.sum(), outlier_pct]

        # Create boxplot and swarmplot
        plt.subplot(len(num_cols), 1, i + 1)
        sns.boxplot(x=df[col], whis=1.5)  # 1.5*IQR range
        plt.title(f'Outlier Detection for {col} (Z > {threshold})')

    plt.tight_layout()
    plt.show()

    # Display outlier statistics
    outliers = outliers.sort_values('Outlier %', ascending=False)
    print(outliers.style.background_gradient(cmap='Oranges', subset=['Outlier %']))

    return outliers

