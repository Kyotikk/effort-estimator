from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def train_ridge(X_train, y_train, X_val, y_val, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    metrics = {
        "MAE": mean_absolute_error(y_val, y_pred),
        "RMSE": root_mean_squared_error(y_val, y_pred),
        "R2": r2_score(y_val, y_pred),
    }

    return model, metrics
