# file to implement cluster analysis for econ project

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg as la
from copy import deepcopy
from tqdm import trange
from functools import partial

## plotting and getting statistics ##
def get_combined_df(sc_data_path='data/social_capital_county.csv', um_data_path='data/cty_kfr_rP_gP_p25.csv'):
    '''creates merged dataset with social capital and mobility data'''

    df_sc = pd.read_csv(sc_data_path)
    df_um = pd.read_csv(um_data_path)

    # get a combined dataset
    # the um has a cty prefix for the county name; need to remove to make it a float
    df_um['cty'] = df_um['cty'].str[3:]
    df_um['cty'] = df_um['cty'].astype(float)

    # get the intersection of the two datasets
    # where county and cty are the same
    df = pd.merge(df_sc, df_um, left_on='county', right_on='cty')
    df = df.drop(columns=['cty'])
    df = df.drop(columns=['Name'])

    # make all except the county name a float
    for col in df.columns:
        if col != 'county_name':
            df[col] = df[col].astype(float)

    # save the combined dataset
    df.to_csv('data/combined.csv', index=False)

def plot_scatter(combined_path='data/combined.csv', colors = None, len_data_restrict = None, name=None):
    '''plot just the scatter plots of the combined dataset'''

    df = pd.read_csv(combined_path)

    if len_data_restrict is not None:
        df = df[:len_data_restrict]

    if colors is not None:
        colors_orig = deepcopy(colors)

    upward_mobility = df['Household_Income_at_Age_35_rP_gP_p25']

    ## economic connectedness data ##
    ec = df['ec_county']
    ec_se = df['ec_se_county'] # standard error economic connectedness
    child_ec = df['child_ec_county']
    child_ec_se = df['child_ec_se_county'] # standard error child economic connectedness
    ec_grp_mem = df['ec_grp_mem_county']
    ec_high = df['ec_high_county']
    ec_high_se = df['ec_high_se_county']
    child_high_ec = df['child_high_ec_county']
    child_high_ec_se = df['child_high_ec_se_county']
    ec_grp_mem_high = df['ec_grp_mem_high_county']
    exposure_grp_mem = df['exposure_grp_mem_county']
    exposure_grp_mem_high = df['exposure_grp_mem_high_county']
    child_exposure = df['child_exposure_county']
    child_exposure_high = df['child_high_exposure_county']
    bias_grp_mem = df['bias_grp_mem_county']
    bias_grp_mem_high = df['bias_grp_mem_high_county']
    child_bias = df['child_bias_county']
    child_bias_high = df['child_high_bias_county']

    ## county cohesiveness data ##
    clustering = df['clustering_county']
    support_ratio = df['support_ratio_county']

    ## civic engagement data ##
    volunteer_rate = df['volunteering_rate_county']
    civic_org = df['civic_organizations_county']

    ## upward mobility data ##
    upward_mobility = df['Household_Income_at_Age_35_rP_gP_p25']

    upward_mobility_orig = deepcopy(upward_mobility)

    # now redo all the plots above, except instead of doing histograms, do scatter plots vs upward mobility
    # 18 measures to plot
    fig, ax = plt.subplots(6, 3, figsize=(15, 20))
    ax = ax.ravel()
    connectedness_cols = ['ec_county', 'ec_se_county', 'child_ec_county', 'child_ec_se_county', 'ec_grp_mem_county', 'ec_high_county', 'ec_high_se_county', 'child_high_ec_county', 'child_high_ec_se_county', 'ec_grp_mem_high_county', 'exposure_grp_mem_county', 'exposure_grp_mem_high_county', 'child_exposure_county', 'child_high_exposure_county', 'bias_grp_mem_county', 'bias_grp_mem_high_county', 'child_bias_county', 'child_high_bias_county']
    for i, data in enumerate([ec, ec_se, child_ec, child_ec_se, ec_grp_mem, ec_high, ec_high_se, child_high_ec, child_high_ec_se, ec_grp_mem_high, exposure_grp_mem, exposure_grp_mem_high, child_exposure, child_exposure_high, bias_grp_mem, bias_grp_mem_high, child_bias, child_bias_high]):
    
        um_data = upward_mobility_orig[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        data_orig = deepcopy(data)
        data = data[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        if colors is None:
            ax[i].scatter(um_data, data, alpha=0.5)
        else:
            colors = colors_orig[(~np.isnan(data_orig)) & (~np.isnan(upward_mobility_orig))]
            ax[i].scatter(um_data, data, alpha=0.5, color=colors)
        ax[i].set_title(f'{connectedness_cols[i]} vs Upward Mobility')
        ax[i].set_xlabel('Upward Mobility')
        ax[i].set_ylabel(f'{connectedness_cols[i]}')
    plt.tight_layout()
    if name is None:
        plt.savefig(f'results/connectedness_scatter_{colors is None}_{len(df)}.pdf')
    else:
        plt.savefig(f'results/connectedness_scatter_{colors is None}_{len(df)}_{name}.pdf')

    # 2 measures to plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax = ax.ravel()
    cohesiveness_cols = ['clustering_county', 'support_ratio_county']
    for i, data in enumerate([clustering, support_ratio]):
        um_data = upward_mobility_orig[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        data_orig = deepcopy(data)
        data = data[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        if colors is None:
            ax[i].scatter(um_data, data, alpha=0.5)
        else:
            colors = colors_orig[(~np.isnan(data_orig)) & (~np.isnan(upward_mobility_orig))]
            ax[i].scatter(um_data, data, alpha=0.5, color=colors)
        ax[i].set_title(f'{cohesiveness_cols[i]} vs Upward Mobility')
        ax[i].set_xlabel('Upward Mobility')
        ax[i].set_ylabel(f'{cohesiveness_cols[i]}')
    plt.tight_layout()
    if name is None:
        plt.savefig(f'results/cohesiveness_scatter_{colors is None}_{len(df)}.pdf')
    else:
        plt.savefig(f'results/cohesiveness_scatter_{colors is None}_{len(df)}_{name}.pdf')

    # 2 measures to plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax = ax.ravel()
    civic_cols = ['volunteering_rate_county', 'civic_organizations_county']
    for i, data in enumerate([volunteer_rate, civic_org]):
        um_data = upward_mobility_orig[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        data_orig = deepcopy(data)
        data = data[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        if colors is None:
            ax[i].scatter(um_data, data, alpha=0.5)
        else:
            colors = colors_orig[(~np.isnan(data_orig)) & (~np.isnan(upward_mobility_orig))]
            ax[i].scatter(um_data, data, alpha=0.5, color=colors)
        ax[i].set_title(f'{civic_cols[i]} vs Upward Mobility')
        ax[i].set_xlabel('Upward Mobility')
        ax[i].set_ylabel(f'{civic_cols[i]}')
    plt.tight_layout()
    if name is None:
        plt.savefig(f'results/civic_engagement_scatter_{colors is None}_{len(df)}.pdf')
    else:
        plt.savefig(f'results/civic_engagement_scatter_{colors is None}_{len(df)}_{name}.pdf')

    # plot pop18 vs upward mobility and weight vs upward mobility
    pop_cols = ['pop2018', 'num_below_p50']
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax = ax.ravel()
    for i, data in enumerate([df['pop2018'], df['num_below_p50']]):
        um_data = upward_mobility_orig[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        data_orig = deepcopy(data)
        data = data[(~np.isnan(data)) & (~np.isnan(upward_mobility_orig))]
        if colors is None:
            ax[i].scatter(um_data, data, alpha=0.5)
        else:
            colors = colors_orig[(~np.isnan(data_orig)) & (~np.isnan(upward_mobility_orig))]
            ax[i].scatter(um_data, data, alpha=0.5, color=colors)
        ax[i].set_title(f'{pop_cols[i]} vs Upward Mobility')
        ax[i].set_xlabel('Upward Mobility')
        ax[i].set_ylabel(f'{pop_cols[i]}')
    plt.tight_layout()
    if name is None:
        plt.savefig(f'results/pop_scatter_{colors is None}_{len(df)}.pdf')
    else:
        plt.savefig(f'results/pop_scatter_{colors is None}_{len(df)}_{name}.pdf')
        


def plot_hist_sum(combined_path='data/combined.csv'):
    '''plot histogram of the sum of mobility and social capital'''

    df = pd.read_csv(combined_path)
    print(df.columns)

    upward_mobility = df['Household_Income_at_Age_35_rP_gP_p25']
    weights = df['num_below_p50']

    ## economic connectedness data ##
    ec = df['ec_county']
    ec_se = df['ec_se_county'] # standard error economic connectedness
    child_ec = df['child_ec_county']
    child_ec_se = df['child_ec_se_county'] # standard error child economic connectedness
    ec_grp_mem = df['ec_grp_mem_county']
    ec_high = df['ec_high_county']
    ec_high_se = df['ec_high_se_county']
    child_high_ec = df['child_high_ec_county']
    child_high_ec_se = df['child_high_ec_se_county']
    ec_grp_mem_high = df['ec_grp_mem_high_county']
    exposure_grp_mem = df['exposure_grp_mem_county']
    exposure_grp_mem_high = df['exposure_grp_mem_high_county']
    child_exposure = df['child_exposure_county']
    child_exposure_high = df['child_high_exposure_county']
    bias_grp_mem = df['bias_grp_mem_county']
    bias_grp_mem_high = df['bias_grp_mem_high_county']
    child_bias = df['child_bias_county']
    child_bias_high = df['child_high_bias_county']

    ## county cohesiveness data ##
    clustering = df['clustering_county']
    support_ratio = df['support_ratio_county']

    ## civic engagement data ##
    volunteer_rate = df['volunteering_rate_county']
    civic_org = df['civic_organizations_county']

    ## upward mobility data ##
    upward_mobility = df['Household_Income_at_Age_35_rP_gP_p25']


    # first make hist of upward mobility
    plt.figure(figsize=(10, 6))
    plt.hist(upward_mobility, bins=100, edgecolor='black')
    plt.title(f'Upward Mobility, {len(upward_mobility)} Counties')
    # drop nans in weights
    # weights = weights.dropna()
    # drop upward mobility where weights are nan
    upward_mobility_orig = deepcopy(upward_mobility)
    upward_mobility = np.where(weights.isna(), np.nan, upward_mobility)
    weights_orig = deepcopy(weights)
    weights = np.where(weights.isna(), np.nan, weights)

    # drop nans in weights
    weights = weights[~np.isnan(weights)]
    upward_mobility = upward_mobility[~np.isnan(upward_mobility)]

    um_avg = np.average(upward_mobility, weights=weights)
    plt.vlines(um_avg, 0, 150, color='red', label=f'Weighted Average = {um_avg:.2f}')
    plt.legend()
    plt.xlabel('Income at Age 35 Given Parent Income in Lowest 5th')
    plt.ylabel('Frequency')
    plt.savefig('results/upward_mobility_hist.pdf')

    # make plots for each of the county connecectedness data
    # calculate weighted mean and standard error and save as df
    # 18 measures to plot
    fig, ax = plt.subplots(6, 3, figsize=(15, 20))
    ax = ax.ravel()
    connectedness_cols = ['ec_county', 'ec_se_county', 'child_ec_county', 'child_ec_se_county', 'ec_grp_mem_county', 'ec_high_county', 'ec_high_se_county', 'child_high_ec_county', 'child_high_ec_se_county', 'ec_grp_mem_high_county', 'exposure_grp_mem_county', 'exposure_grp_mem_high_county', 'child_exposure_county', 'child_high_exposure_county', 'bias_grp_mem_county', 'bias_grp_mem_high_county', 'child_bias_county', 'child_high_bias_county']
    for i, data in enumerate([ec, ec_se, child_ec, child_ec_se, ec_grp_mem, ec_high, ec_high_se, child_high_ec, child_high_ec_se, ec_grp_mem_high, exposure_grp_mem, exposure_grp_mem_high, child_exposure, child_exposure_high, bias_grp_mem, bias_grp_mem_high, child_bias, child_bias_high]):
        ax[i].hist(data, bins=100, edgecolor='black')
        ax[i].set_title(f'{connectedness_cols[i]}')

        # drop where weights are nan
        data = data[(~np.isnan(weights_orig)) &(~np.isnan(data))]
        weights_data = weights_orig[(~np.isnan(weights_orig)) &(~np.isnan(data))]

        # calculate weighted mean
        mean = np.average(data, weights=weights_data)
        ax[i].vlines(mean, 0, 100, color='red', label=f'Weighted Average = {mean:.2f}')

        ax[i].set_xlabel('Value')
        ax[i].set_ylabel('Frequency')
        ax[i].legend()
    plt.tight_layout()
    plt.savefig('results/connectedness_hist.pdf')

    # make plots for each of the county cohesiveness data
    # calculate weighted mean and standard error and save as df
    # 2 measures to plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax = ax.ravel()
    cohesiveness_cols = ['clustering_county', 'support_ratio_county']
    for i, data in enumerate([clustering, support_ratio]):
        ax[i].hist(data, bins=100, edgecolor='black')
        ax[i].set_title(f'{cohesiveness_cols[i]}')

        # drop where weights are nan
        data = data[(~np.isnan(weights_orig)) &(~np.isnan(data))]
        weights_data = weights_orig[(~np.isnan(weights_orig)) &(~np.isnan(data))]

        # calculate weighted mean
        mean = np.average(data, weights=weights_data)
        ax[i].vlines(mean, 0, 140, color='red', label=f'Weighted Average = {mean:.2f}')

        ax[i].set_xlabel('Value')
        ax[i].set_ylabel('Frequency')
        ax[i].legend()
    plt.tight_layout()
    plt.savefig('results/cohesiveness_hist.pdf')

    # make plots for each of the civic engagement data
    # calculate weighted mean and standard error and save as df
    # 2 measures to plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax = ax.ravel()
    civic_cols = ['volunteering_rate_county', 'civic_organizations_county']

    for i, data in enumerate([volunteer_rate, civic_org]):
        ax[i].hist(data, bins=100, edgecolor='black')
        ax[i].set_title(f'{civic_cols[i]}')

        # drop where weights are nan
        data = data[(~np.isnan(weights_orig)) &(~np.isnan(data))]
        weights_data = weights_orig[(~np.isnan(weights_orig)) &(~np.isnan(data))]

        # calculate weighted mean
        mean = np.average(data, weights=weights_data)
        ax[i].vlines(mean, 0, 160, color='red', label=f'Weighted Average = {mean:.2f}')

        ax[i].set_xlabel('Value')
        ax[i].set_ylabel('Frequency')
        ax[i].legend()
    plt.tight_layout()
    plt.savefig('results/civic_engagement_hist.pdf')

    plot_scatter()


## perform cluster analysis ##
def create_X_y(df_path='data/combined.csv'):
    '''create X and y for clustering analysis'''

    df = pd.read_csv(df_path)
    # randomly shuffle the data
    df = df.sample(frac=1)

    y = df['Household_Income_at_Age_35_rP_gP_p25']
    X = df.drop(columns=['county_name', 'county', 'Household_Income_at_Age_35_rP_gP_p25'])

    # normalize all columns of X 
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    # convert to numpy
    X = X.to_numpy()
    y = y.to_numpy()
    X = X.astype(float)
    y = y.astype(float)

    # save X and y
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)

def compute_similarity(X, index1, index2, dim, param):
    ''' compute similarity between two counties for a specific dimension'''
    # get data for each county
    x1 = X[index1, dim]
    x2 = X[index2, dim]
    sim = param * 1/(1 + np.abs(x1 - x2))
    if np.isnan(sim):
        sim = 0
    return sim

def get_adjacency_matrix(X, params):
    '''get the adjacency matrix for the combined dataset'''
    n = X.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):  # Start from i+1 to avoid self-loops and ensure we only fill the upper triangle
            for dim in range(X.shape[1]):
                adj[i, j] += compute_similarity(X, i, j, dim, params[dim])
            adj[j, i] = adj[i, j]  # Mirror the value to the lower triangle
    return adj

def cluster(X, y, params, show_scatter=False, show_graph=False, param_name=None):
    '''perform cluster analysis on the combined dataset

    Params:
        X (np.array): the combined dataset
        y (np.array): the upward mobility data
        params (np.array): the parameters for the similarity function
        plot_scatter (bool): whether to plot the scatter plot
        plot_graph (bool): whether to plot the graph (don't do this for large datasets)
    
    
    '''

    # get the adjacency matrix
    adj = get_adjacency_matrix(X, params)
    # adj = (adj + adj.T) / 2

    # create graph
    g = ig.Graph.Weighted_Adjacency(adj.tolist(), mode='undirected')

    # get the partition
    partition = la.find_partition(g, la.ModularityVertexPartition, weights='weight')

    modularity = partition.modularity

    # get the clusters
    clusters = partition.membership

    # get the cluster sizes
    cluster_sizes = np.bincount(clusters)

    
    # redo same cluster, but divide into percentiles based on the number of clusters
    # get the percentiles
    percentile = np.percentile(y, np.linspace(0, 100, len(cluster_sizes)+1))
    # which percentile does each county belong to
    # find the mean within each cluster
    y_means = np.zeros(len(cluster_sizes))
    for i in range(len(cluster_sizes)):
        y_means[i] = np.mean(y[clusters == i])
    # reassign the clusters based on the relative values. 0 is the lowest, len(cluster_sizes) is the highest
    y_means = np.argsort(y_means)
    # log where each index went
    # Create a mapping from old cluster labels to new ones based on sorted order
    new_labels_mapping = np.zeros_like(y_means)
    new_labels_mapping[np.arange(len(y_means))] = y_means # map the old indices to the new ones

    # Apply the new mapping to the clusters array
    clusters = new_labels_mapping[clusters]

    # decide based on the y value
    percentile_clusters = np.zeros(X.shape[0])
    for i in range(len(cluster_sizes)-1):
        percentile_clusters[(y >= percentile[i]) & (y <= percentile[i+1])] = i

    # compute overlap between clusters and percentiles
    # compare the number of the same class in each cluster
    overlap = np.zeros(len(X))
    for i in range(len(X)):
        overlap[i] = clusters[i] == percentile_clusters[i]

    # get the overlap percentage
    overlap_percentage = np.sum(overlap) / len(X)
    print(f'Overlap Percentage: {overlap_percentage}')
    print(f'Modularity: {modularity}')

    if show_scatter:
         # save the clusters
        savename = ''
        if param_name is not None:
            savename += param_name
        np.save(f'data/clusters_{savename}.npy', clusters)
    
        # replot the scatter plot with the clusters
        # assign each cluster a color
        colors = np.zeros(X.shape[0])
        # get list of colors
        colormap = plt.cm.get_cmap('tab10', len(cluster_sizes))

        # apply the colormap to the colors
        colors = colormap(clusters)

        # plot the scatter plot
        name = 'cluster'
        if param_name is not None:
            name += param_name
        plot_scatter(colors=colors, len_data_restrict=len(X), name=name)

        # get the colors for the percentiles
        percentile_colors = colormap(percentile_clusters)
        print(percentile_colors)

        # plot the scatter plot
        name = 'percentile'
        if param_name is not None:
            name += param_name

        plot_scatter(colors=percentile_colors, len_data_restrict=len(X), name=name)

    if show_graph:
        # display the clusters
        visual_style = {}
        visual_style["vertex_size"] = 20
        visual_style["bbox"] = (1000, 1000)
        visual_style["margin"] = 50
        visual_style["edge_width"] = 0.1  # Set edge width to a smaller value for thinner edges

        filename = 'results/cluster_plot'
        if param_name is not None:
            filename += param_name
        filename += '.pdf'
        ig.plot(partition, target=filename, **visual_style)
        plt.show()

    return overlap_percentage, clusters, modularity

## optimize the parameters ##
def random_params(shape):
    '''generate random parameters for the similarity function'''
    return np.random.uniform(0, 1, shape)

def optimize_params(X, y, n_iter=100):
    '''optimize the parameters for the similarity function'''

    random_func = partial(random_params, X.shape[1])
    cluster_func = partial(cluster, X, y)

    # get the initial parameters
    init_params = np.ones(X.shape[1])

    # get the initial overlap percentage
    overlap_percentage, clusters, modularity = cluster_func(init_params)

    # iterate to find the best parameters
    try:
        for i in trange(n_iter):
            print(f'Iteration: {i}')
            new_params = random_func()
            new_overlap_percentage, new_clusters, new_modularity = cluster_func(new_params)
            if new_overlap_percentage > overlap_percentage:
                overlap_percentage = new_overlap_percentage
                clusters = new_clusters
                modularity = new_modularity
                params = new_params
                print(f'New Best Overlap Percentage: {overlap_percentage}')
                print(f'New Best Modularity: {modularity}')
                print(f'New Best Parameters: {params}')
    except KeyboardInterrupt:
        print(f'Best Overlap Percentage: {overlap_percentage}')
        print(f'Best Modularity: {modularity}')
        print(f'Best Parameters: {params}')

        np.save('data/best_params.npy', params)
        np.save('data/best_clusters.npy', clusters)
        np.save('data/best_modularity.npy', modularity)
        return overlap_percentage, clusters, modularity, params
    
    np.save('data/best_params.npy', params)
    np.save('data/best_clusters.npy', clusters)
    np.save('data/best_modularity.npy', modularity)
    return overlap_percentage, clusters, modularity, params

if __name__ == '__main__':
    # get_combined_df()
    # plot_hist_sum()
    # create_X_y()
    X, y = np.load('data/X.npy', allow_pickle=True), np.load('data/y.npy', allow_pickle=True)
    optimize_params(X, y)


    # params = np.ones(X.shape[1])
    # # params = np.random.uniform(0, 1, X.shape[1])
    # X_mini = X[:400]
    # y_mini = y[:400]
    # clusters = cluster(X_mini, y_mini, params, show_scatter=True, show_graph=False, param_name=f'_{len(X_mini)}_equal')