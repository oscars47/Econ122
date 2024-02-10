# file for helper functions for econ

import pandas as pd
import numpy as np
import statsmodels.api as sm

def summarize(col, weight=None):
    '''summarizes a column of data, including mean, median, standard deviation, variance, 25th, 50th, and 75th percentiles, skewness, kurtosis, min, and max. If weight is provided, it will also calculate the weighted mean, weighted standard deviation, and weighted variance.
    '''
    # convert to numpy array
    col = np.array(col).astype(float)
    if weight is not None:
        weight = np.array(weight).astype(float)

    # drop all NaNs
    if weight is None:
        col = col[~np.isnan(col)]
    else:
        col = col[~np.isnan(col) & ~np.isnan(weight)]
        weight = weight[~np.isnan(col) & ~np.isnan(weight)]
        # make sure col and weight are the same length
        assert len(col) == len(weight)

    print('Num observations: ', len(col))
    if type(col) != pd.Series:
        col = pd.Series(col)
    if weight is not None:
        print("Weighted Mean: ", np.average(col, weights=weight))
    else:
        print("Mean: ", col.mean())
        print("Median: ", col.median())
    if weight is not None:
        print("Weighted Standard Deviation: ", np.sqrt(np.average((col - np.average(col, weights=weight))**2, weights=weight)))
        print("Weighted Variance: ", np.average((col - np.average(col, weights=weight))**2, weights=weight))
    else:
        print("Standard Deviation: ", col.std())
        print("Variance: ", col.var())

    # else:
    print("1st percentile: ", col.quantile(0.01))
    print("5th percentile: ", col.quantile(0.05))
    print("10th percentile: ", col.quantile(0.10))
    print("25th percentile: ", col.quantile(0.25))
    print("50th percentile: ", col.quantile(0.50))
    print("75th percentile: ", col.quantile(0.75))
    print("90th percentile: ", col.quantile(0.90))
    print("95th percentile: ", col.quantile(0.95))
    print("99th percentile: ", col.quantile(0.99))

    # get skewness
    print("Skewness: ", col.skew())

    # get kurtosis
    print("Kurtosis: ", col.kurtosis())

    # get min and max
    print("Min: ", col.min())
    print("Max: ", col.max())

def prepare_data(dataframe, condition, X_cols, Y_col, W_col):
    '''Creates X, y, and w for a weighted least squares regression.

    Params:
        dataframe: pd.DataFrame
        condition: np.array
        X_cols: list
        Y_col: str
        W_col: str

    '''
    # make sure no NaN values
    if condition is not None:
        condition = condition & (~np.isnan(dataframe[Y_col]))
        y = dataframe[Y_col][condition]
        X = dataframe[X_cols][condition]
        # make sure all are type float
        w = dataframe[W_col][condition]

    else:
        condition = ~np.isnan(dataframe[Y_col])
        y = dataframe[Y_col]
        X = dataframe[X_cols]
        w = dataframe[W_col]

    y = y.astype(float)
    X = X.astype(float)
    w = w.astype(float)
       

    return X, y, w

def run_WLS(X, y, w, print_summary=True):
    '''runs a weighted least squares regression'''
    # Fitting the model
    X = sm.add_constant(X)

    model = sm.WLS(y, X, weights=w).fit()

    # Using robust standard errors (e.g., HC1)
    robust_model = model.get_robustcov_results(cov_type='HC1')

    # Viewing the summary with robust standard errors
    if print_summary:
        print(robust_model.summary())
    return robust_model

def oaxaca_blinder(X1, X2, y1, y2, w1, w2):
    '''performs an Oaxaca-Blinder decomposition of the mean differences in characteristics and the mean differences in coefficients between two groups'''

    # Mean differences in characteristics
    mean_diff = sm.add_constant(X1).mean() - sm.add_constant(X2).mean()
    print('Mean diff: ', mean_diff)

    # run WLS
    model1 = run_WLS(X1, y1, w1, print_summary=False)
    model2 = run_WLS(X2, y2, w2, print_summary=False)

    # Coefficients
    coeff1 = model1.params
    coeff2 = model2.params

    print('Coeff1: ', coeff1)
    print('Coeff2: ', coeff2)

    # Decomposition
    explained = mean_diff.dot(coeff1)
    unexplained = (sm.add_constant(X2).mean().dot(coeff1 - coeff2))

    # Print results
    print(f"Explained component: {explained}")
    print(f"Unexplained component: {unexplained}")

    # total gap
    print(f"Total gap: {explained + unexplained}")


if __name__ == "__main__":
    pass
    # run_WLS()