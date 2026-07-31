from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_baseline_models(class_weight=None, random_state=42):
    """
    Returns a dictionary of un-initialized baseline models.
    Tree-models do not receive scaling. Linear models get a StandardScaler pipeline.
    """
    
    # Random Forest
    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    
    # LightGBM handles class weighting natively via class_weight argument.
    lgbm = LGBMClassifier(
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        verbose=-1
    )
    
    # Logistic Regression requires scaling.
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            max_iter=1000,
            n_jobs=-1
        ))
    ])
    
    return {
        "RandomForest": rf,
        "LightGBM": lgbm,
        "LogisticRegression": lr
    }
