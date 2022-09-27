# Libraries
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def r_squared(y_true, y_pred, round=False):
    # * Function that returns r2 score

    if round:
        return round(r2_score(y_true, y_pred), 2)
    else:
        return r2_score(y_true, y_pred)

def mae(y_true, y_pred, round=False):
    # * Function that returns Mean Absolute Error

    if round:
        return round(mean_absolute_error(y_true, y_pred), 2)
    else:
        return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred, round=False):
    # * Function that returns Mean Squared Error

    if round:
        return round(mean_squared_error(y_true, y_pred), 2)
    else:
        return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred, round=False):\
    # * Function that returns Root Mean Squared Error

    if round:
        return round(mean_squared_error(y_pred, y_true, squared=False), 2)
    else:
        return mean_squared_error(y_true, y_pred, squared=False)

def reg_report(y_true, y_pred):
    # * Function that returns regression report
    # * with 4 metrics (r2, mae, mse, rmse)

    # All metrics saved variables
    r2 =    r2_score(y_true, y_pred)
    mae =   mean_absolute_error(y_true, y_pred)
    mse =   mean_squared_error(y_true, y_pred)
    rmse =  mean_squared_error(y_true, y_pred, squared=False)

    # Create dataframe with merics
    report = {'R2':r2, 'MAE':mae, 'MSE':mse, "RMSE":rmse}
    report = pd.DataFrame.from_dict(report, orient='index', columns=['Value'])
    report = report.round(2)
    
    return report