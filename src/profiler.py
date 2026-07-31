import pandas as pd
import config
from datetime import datetime

def generate_overall_profile(df: pd.DataFrame) -> dict:
    """Generates overall profiling stats for a dataframe."""
    profile = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values_total': df.isna().sum().sum()
    }
    return profile

def generate_column_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generates detailed statistics per column."""
    stats = []
    for col in df.columns:
        col_data = df[col]
        stats.append({
            'column': col,
            'dtype': str(col_data.dtype),
            'missing': col_data.isna().sum(),
            'unique_values': col_data.nunique()
        })
    return pd.DataFrame(stats)

def generate_appliance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates appliance-wise (label-wise) statistical summaries 
    for every numerical feature (mean, std, min, max, median).
    """
    if 'label' not in df.columns:
        return pd.DataFrame()
        
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # We want grouped stats
    grouped = df.groupby('label')[num_cols].agg(['mean', 'std', 'min', 'max', 'median'])
    
    # Flatten multi-index columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    return grouped.reset_index()

def format_markdown_report(profile: dict, col_stats: pd.DataFrame, ts_stats: dict, title: str) -> str:
    """Format the profiling info into a Markdown report string."""
    md = f"# Data Profile Report: {title}\n\n"
    md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    md += "## Overall Statistics\n"
    for k, v in profile.items():
        md += f"- **{k}**: {v}\n"
        
    if ts_stats:
        md += "\n## Timestamp Issues\n"
        for k, v in ts_stats.items():
            md += f"- **{k}**: {v}\n"
            
    md += "\n## Column Information\n"
    md += col_stats.to_markdown(index=False)
    md += "\n"
    
    return md
