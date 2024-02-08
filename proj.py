# code to run my econ project

from sklearn.preprocessing import StandardScaler
import pandas as pd
import sympy as sp
import numpy as np

# process data
def process_data(dataframe):
    '''Normalize each column of the dataframe'''
    # initialize StandardScaler
    scaler = StandardScaler()
    # normalize each column
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
    return dataframe

def compute_similarity_matrix_sympy(df):
    '''compute the similarity matrix for the dataframe, using sympy

    FOR DEMONSTRATION PURPOSES ONLY
    
    '''
    # Calculate pairwise similarities
    n = len(df)
    similarity_matrix = sp.zeros(n, n)

    # create params for each unique pair of individuals
    for i in range(n):
        for j in range(i+1, n):
            # now iterate through each column
            similarity = 0
            for l, col in enumerate(df.columns):
                # get param
                param = sp.symbols(f's_{i}_{j}_{l}')
                similarity += param * np.round(1/(1 + (df.iloc[i][col] - df.iloc[j][col])**2), 3)

            # add to similarity matrix
            similarity_matrix[i, j] = similarity

    # make symmetric
    similarity_matrix = similarity_matrix + similarity_matrix.T
    return similarity_matrix

def run_sample():
    '''sample data for illustration'''
    df = pd.DataFrame()
    df['age'] = [21, 34, 75]
    df['marriage'] = [1, 0, 1]
    df['poverty'] = [1, 4, 3]

    # process data
    df = process_data(df)
    print(df)
    sp.print_latex(compute_similarity_matrix_sympy(df))


if __name__ == '__main__':
    run_sample()

                


   


