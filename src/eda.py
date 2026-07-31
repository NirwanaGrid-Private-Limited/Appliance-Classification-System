import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import logging

logger = config.setup_logger("eda", config.LOGS_DIR / "eda.log")

def plot_class_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
    plt.title('Appliance Class Distribution')
    plt.xlabel('Appliance')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'class_distribution.png')
    plt.close()

def plot_feature_distributions(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=['float64']).columns
    for col in num_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue='label', element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {col} by Appliance')
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f'dist_{col}.png')
        plt.close()

def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=['float64']).columns
    plt.figure(figsize=(12, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'correlation_heatmap.png')
    plt.close()

def plot_boxplots(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=['float64']).columns
    for col in num_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='label', y=col)
        plt.title(f'Boxplot of {col} by Appliance (Outlier Visualization)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f'boxplot_{col}.png')
        plt.close()

def generate_all_plots(df: pd.DataFrame):
    logger.info("Generating EDA plots...")
    if 'label' not in df.columns:
        logger.warning("No 'label' column found, skipping EDA.")
        return
        
    try:
        plot_class_distribution(df)
        plot_feature_distributions(df)
        plot_correlation_heatmap(df)
        plot_boxplots(df)
        logger.info("EDA plots generated successfully.")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
