from sklearn.ensemble import RandomForestClassifier


def create_model(
    n_estimators: int = 100,
    max_depth: int | None = 4,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Создаём простую модель RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    return model
