import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

from . import config

class DataProcessor:
    @staticmethod
    def load_data(filepath: Path, verbose: bool = True) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(filepath)
            if verbose:
                print(f"Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột")
            return df
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None

    @staticmethod
    def save_data(df: pd.DataFrame, filepath: Path, index: bool = False):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=index)
        print(f"Đã lưu dữ liệu tại: {filepath}")

    @staticmethod
    def _check_condition(df: pd.DataFrame, condition: pd.Series,
                         description: str, show_pct: bool = False) -> Tuple[int, pd.DataFrame]:
        violations = df[condition]
        n_violations = len(violations)

        if show_pct:
            pct = n_violations / len(df) * 100
            print(f"{description}: {n_violations} ({pct:.2f}%)")
        else:
            print(f"{description}: {n_violations}")

        return n_violations, violations

    @staticmethod
    def log_transform(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        df_transformed = df.copy()
        for feature in features:
            df_transformed[feature] = np.log1p(df_transformed[feature])
        return df_transformed

    @staticmethod
    def split_data(X: pd.DataFrame,
                   y: pd.Series,
                   test_size: float = config.TEST_SIZE,
                   random_state: int = config.RANDOM_STATE,
                   verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if verbose:
            print(f"Kích thước tập huấn luyện: {X_train.shape}")
            print(f"Kích thước tập kiểm tra: {X_test.shape}")

        return X_train, X_test, y_train, y_test