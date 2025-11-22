from src.preprocess import load_raw_data
from src.model import create_model


def test_load_data_and_model():
    df = load_raw_data()
    assert not df.empty
    assert "target" in df.columns

    model = create_model()
    assert model is not None
