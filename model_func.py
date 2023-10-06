import joblib
from statsmodels.tsa.vector_ar.var_model import VAR


def load_model(path_model: str):
    try:
        return joblib.load(path_model)
    except Exception as e:
        print(f'Model {path_model} not load. Error: {e}')
        return 0


def save_model(model: any, file_path: str):
    try:
        joblib.dump(model, file_path)
    except Exception as e:
        print(f'Model {model} not save. Error: {e}')


def fit_model(data, order: int = 1, include_const=True):
    try:
        trend_value = 'c' if include_const else 'ct'
        return VAR(data).fit(maxlags=order, ic=None, trend=trend_value)
    except Exception as e:
        print(f"Model not fit. Error: {e}")


def predict_next_values(model, data, steps: int = 1):
    try:
        return model.forecast(data, steps=steps)
    except Exception as e:
        print(f'ERROR : model {model} not predict. Error: {e}')
        return ''


def preprocess_data_for_predict(data, selected_features, count_rows: int = 0):
    if count_rows > 0:
        data = data.tail(count_rows)
        return data[selected_features].values
    else:
        return data[selected_features].values
