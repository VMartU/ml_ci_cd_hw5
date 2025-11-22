from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


def load_raw_data() -> pd.DataFrame:
    """
    Загружаем датасет iris из sklearn и возвращаем DataFrame с колонкой 'target'.
    """
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Делим данные на train и test.
    """
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
