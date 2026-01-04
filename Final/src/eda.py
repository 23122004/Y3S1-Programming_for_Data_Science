import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

class OutlierAnalyzer:
    @staticmethod
    def has_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> bool:
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        return ((data < lower_bound) | (data > upper_bound)).any()

    @staticmethod
    def get_columns_with_outliers(df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   verbose: bool = True) -> List[str]:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        cols_with_outliers = [
            col for col in columns
            if OutlierAnalyzer.has_outliers_iqr(df[col])
        ]

        if verbose:
            print(f"Số cột có outlier: {len(cols_with_outliers)}")
            print(f"Các cột có outlier: {cols_with_outliers}")

        return cols_with_outliers

class CorrelationAnalyzer:
    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                method: str = 'pearson') -> pd.DataFrame:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[columns].corr(method=method)

    @staticmethod
    def get_high_correlations(corr_matrix: pd.DataFrame,
                               threshold: float = 0.7) -> pd.DataFrame:
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': round(corr_value, 4)
                    })

        if high_corr_pairs:
            return pd.DataFrame(high_corr_pairs).sort_values(
                'Correlation', key=abs, ascending=False
            )
        print(f"Không có cặp đặc trưng nào có tương quan cao.")
        return pd.DataFrame()

class GroupedAnalyzer:
    @staticmethod
    def analyze_by_group(df: pd.DataFrame, group_col: str,
                         metrics: List[str], observed: bool = True) -> pd.DataFrame:
        if group_col not in df.columns:
            raise ValueError(f"Đặc trưng {group_col} không tồn tại")

        available_metrics = [m for m in metrics if m in df.columns]
        return df.groupby(group_col, observed=observed)[available_metrics].mean().round(2)

    @staticmethod
    def perform_one_way_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[float, float]:
        import scipy.stats as stats

        if group_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"Cột '{group_col}' hoặc '{value_col}' không tồn tại")

        groups = [
            group[value_col].dropna()
            for _, group in df.groupby(group_col, observed=True)
        ]

        f_stat, p_value = stats.f_oneway(*groups)
        return f_stat, p_value

