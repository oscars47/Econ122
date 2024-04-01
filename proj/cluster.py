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
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# https://www.taxpolicycenter.org/statistics/household-income-quintiles
MEAN_SEC_FIFTH = 32631

## plotting and getting statistics ##
def get_combined_df(sc_data_path='data/social_capital_county.csv', um_data_path='data/cty_kfr_rP_gP_p25.csv', name=None):
    '''creates merged dataset with social capital and mobility data'''

    df_sc = pd.read_csv(sc_data_path)
    df_um = pd.read_csv(um_data_path)

    # rename last column to income
    df_um = df_um.rename(columns={df_um.columns[-1]: 'income'})

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
    if name is not None:
        df.to_csv(f'data/combined_{name}.csv', index=False)
    else:
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

    upward_mobility = df['income']
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

    connectedness_labels = ['Economic Connectedness', 'Economic Connectedness Standard Error', 'Child Economic Connectedness', 'Child Economic Connectedness Standard Error', 'Economic Connectedness by Group Membership', 'Economic Connectedness by High Income', 'Economic Connectedness by High Income Standard Error', 'Child Economic Connectedness by High Income', 'Child Economic Connectedness by High Income Standard Error', 'Economic Connectedness by Group Membership by High Income', 'Exposure Group Membership', 'Exposure by Group Membership High Income', 'Child Exposure', 'Child Exposure High Income', 'Bias Group Membership', 'Bias by Group Membership High Income', 'Child Bias', 'Child Bias by High Income']
    
    for i, data in enumerate([ec, ec_se, child_ec, child_ec_se, ec_grp_mem, ec_high, ec_high_se, child_high_ec, child_high_ec_se, ec_grp_mem_high, exposure_grp_mem, exposure_grp_mem_high, child_exposure, child_exposure_high, bias_grp_mem, bias_grp_mem_high, child_bias, child_bias_high]):
        ax[i].hist(data, bins=100, edgecolor='black')
        ax[i].set_title(f'{connectedness_labels[i]}')

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
    cohesiveness_labels = ['Clustering', 'Support Ratio']
    for i, data in enumerate([clustering, support_ratio]):
        ax[i].hist(data, bins=100, edgecolor='black')
        ax[i].set_title(f'{cohesiveness_labels[i]}')

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
    civic_labels = ['Volunteering Rate', 'Civic Organizations']

    for i, data in enumerate([volunteer_rate, civic_org]):
        ax[i].hist(data, bins=100, edgecolor='black')
        ax[i].set_title(f'{civic_labels[i]}')

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

def make_tables(combined_path='data/combined.csv'):
    df = pd.read_csv(combined_path)

    if 'income' in df.columns:
        upward_mobility = df['income']
    else:
        upward_mobility = df['Household_Income_at_Age_35_rP_gP_p25']
    weights = df['num_below_p50']

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

    # table 1:
    # calculate weighted mean and standard error and save as df
    connectedness_means = []
    connectedness_std = []
    num_data = []
    connectedness_mins = []
    connectedness_maxs = []
    for data in [upward_mobility, ec, ec_se, child_ec, child_ec_se, ec_grp_mem, ec_high, ec_high_se, child_high_ec, child_high_ec_se, ec_grp_mem_high, exposure_grp_mem, exposure_grp_mem_high, child_exposure, child_exposure_high, bias_grp_mem, bias_grp_mem_high, child_bias, child_bias_high]:
        data = data[(~np.isnan(weights)) &(~np.isnan(data))]
        weights_data = weights[(~np.isnan(weights)) &(~np.isnan(data))]
        num = len(data)

        # calculate weighted mean
        mean = np.average(data, weights=weights_data)
        std = np.sqrt(np.average((data - mean)**2, weights=weights_data))
        connectedness_means.append(mean)
        connectedness_std.append(std)
        num_data.append(num)
        connectedness_mins.append(np.min(data))
        connectedness_maxs.append(np.max(data))

    name = combined_path.split('/')[-1].split('.')[0]

    df_1 = pd.DataFrame()
    df_1['Measure'] = ['upward_mobility', 'ec_county', 'ec_se_county', 'child_ec_county', 'child_ec_se_county', 'ec_grp_mem_county', 'ec_high_county', 'ec_high_se_county', 'child_high_ec_county', 'child_high_ec_se_county', 'ec_grp_mem_high_county', 'exposure_grp_mem_county', 'exposure_grp_mem_high_county', 'child_exposure_county', 'child_high_exposure_county', 'bias_grp_mem_county', 'bias_grp_mem_high_county', 'child_bias_county', 'child_high_bias_county']
    
    
    df_1['Measure'] = ['Upward Mobility', 'Economic Connectedness', 'Economic Connectedness Standard Error', 'Child Economic Connectedness', 'Child Economic Connectedness Standard Error', 'Economic Connectedness by Group Membership', 'Economic Connectedness by High Income', 'Economic Connectedness by High Income Standard Error', 'Child Economic Connectedness by High Income', 'Child Economic Connectedness by High Income Standard Error', 'Economic Connectedness by Group Membership by High Income', 'Exposure Group Membership', 'Exposure by Group Membership High Income', 'Child Exposure', 'Child Exposure High Income', 'Bias Group Membership', 'Bias by Group Membership High Income', 'Child Bias', 'Child Bias by High Income']
    
    df_1['Weighted Mean'] = np.round(connectedness_means,4)
    df_1['Standard Error'] = np.round(connectedness_std,4)
    df_1['Min'] = connectedness_mins
    df_1['Max'] = connectedness_maxs
    df_1['Number of Data Points'] = num_data
    df_1.to_csv(f'results/connectedness_table_{name}.csv', index=False)

    ## county cohesiveness data ##
    clustering = df['clustering_county']
    support_ratio = df['support_ratio_county']

    cohesiveness_means = []
    cohesiveness_std = []
    cohesiveness_mins = []
    cohesiveness_maxs = []
    num_data = []
    for data in [clustering, support_ratio]:
        data = data[(~np.isnan(weights)) &(~np.isnan(data))]
        weights_data = weights[(~np.isnan(weights)) &(~np.isnan(data))]
        num = len(data)

        # calculate weighted mean
        mean = np.average(data, weights=weights_data)
        std = np.sqrt(np.average((data - mean)**2, weights=weights_data))
        cohesiveness_means.append(mean)
        cohesiveness_std.append(std)
        num_data.append(num)
        cohesiveness_mins.append(np.min(data))
        cohesiveness_maxs.append(np.max(data))

    df_2 = pd.DataFrame()
    # df_2['Measure'] = ['clustering_county', 'support_ratio_county']
    df_2['Measure'] = ['Clustering', 'Support Ratio']
    cohesiveness_means = np.round(cohesiveness_means, 4)
    cohesiveness_std = np.round(cohesiveness_std, 4)
    df_2['Weighted Mean'] = cohesiveness_means
    df_2['Min'] = cohesiveness_mins
    df_2['Max'] = cohesiveness_maxs
    df_2['Standard Error'] = cohesiveness_std
    df_2['Number of Data Points'] = num_data

    df_2.to_csv(f'results/cohesiveness_table_{name}.csv', index=False)

    ## civic engagement data ##
    volunteer_rate = df['volunteering_rate_county']
    civic_org = df['civic_organizations_county']

    civic_means = []
    civic_std = []
    num_data = []
    civic_mins = []
    civic_maxs = []
    for data in [volunteer_rate, civic_org]:
        data = data[(~np.isnan(weights)) &(~np.isnan(data))]
        weights_data = weights[(~np.isnan(weights)) &(~np.isnan(data))]
        num = len(data)

        # calculate weighted mean
        mean = np.average(data, weights=weights_data)
        std = np.sqrt(np.average((data - mean)**2, weights=weights_data))
        civic_means.append(mean)
        civic_std.append(std)
        num_data.append(num)
        civic_mins.append(np.min(data))
        civic_maxs.append(np.max(data))

    df_3 = pd.DataFrame()
    # df_3['Measure'] = ['volunteering_rate_county', 'civic_organizations_county']
    df_3['Measure'] = ['Volunteering Rate', 'Civic Organizations']
    civic_means = np.round(civic_means, 4)
    civic_std = np.round(civic_std, 4)
    df_3['Weighted Mean'] = civic_means
    df_3['Standard Error'] = civic_std
    df_3['Min'] = civic_mins
    df_3['Max'] = civic_maxs
    df_3['Number of Data Points'] = num_data
    

    df_3.to_csv(f'results/civic_engagement_table_{name}.csv', index=False)

    return df_1, df_2, df_3, upward_mobility, weights, ec, ec_se, child_ec, child_ec_se, ec_grp_mem, ec_high, ec_high_se, child_high_ec, child_high_ec_se, ec_grp_mem_high, exposure_grp_mem, exposure_grp_mem_high, child_exposure, child_exposure_high, bias_grp_mem, bias_grp_mem_high, child_bias, child_bias_high, clustering, support_ratio, volunteer_rate, civic_org

def plot_upward_mobility(plot_male_female=True):
    '''plot the histograms of the data, showing weighted mean for each. if plot_male_female, then plot the male and female data separately as a histogram.'''
    df_1, df_2, df_3, upward_mobility, weights, ec, ec_se, child_ec, child_ec_se, ec_grp_mem, ec_high, ec_high_se, child_high_ec, child_high_ec_se, ec_grp_mem_high, exposure_grp_mem, exposure_grp_mem_high, child_exposure, child_exposure_high, bias_grp_mem, bias_grp_mem_high, child_bias, child_bias_high, clustering, support_ratio, volunteer_rate, civic_org = make_tables()
    if plot_male_female:
        # df_1_male, df_2_male, df_3_male,  = make_tables('data/combined_male.csv')
        # df_1_female, df_2_female, df_3_female = make_tables('data/combined_female.csv')
        df_1_male, df_2_male, df_2_male, upward_mobility_male, weights_male, ec_male, ec_se_male, child_ec_male, child_ec_se_male, ec_grp_mem_male, ec_high_male, ec_high_se_male, child_high_ec_male, child_high_ec_se_male, ec_grp_mem_high_male, exposure_grp_mem_male, exposure_grp_mem_high_male, child_exposure_male, child_exposure_high_male, bias_grp_mem_male, bias_grp_mem_high_male, child_bias_male, child_bias_high_male, clustering_male, support_ratio_male, volunteer_rate_male, civic_org_male = make_tables('data/combined_male.csv')

        df_1_female, df_2_female, df_2_female, upward_mobility_female, weights_female, ec_female, ec_se_female, child_ec_female, child_ec_se_female, ec_grp_mem_female, ec_high_female, ec_high_se_female, child_high_ec_female, child_high_ec_se_female, ec_grp_mem_high_female, exposure_grp_mem_female, exposure_grp_mem_high_female, child_exposure_female, child_exposure_high_female, bias_grp_mem_female, bias_grp_mem_high_female, child_bias_female, child_bias_high_female, clustering_female, support_ratio_female, volunteer_rate_female, civic_org_female = make_tables('data/combined_female.csv')


    # first make hist of upward mobility
    plt.figure(figsize=(10, 6))
    if plot_male_female:
        plt.hist(upward_mobility, bins=100, edgecolor='black', alpha=0.5)
        plt.hist(upward_mobility_male, bins=100, edgecolor='black', label='Male', alpha=0.5)
        plt.hist(upward_mobility_female, bins=100, edgecolor='black', label='Female', alpha=0.5)
    else:
        plt.hist(upward_mobility, bins=100, edgecolor='black')

    if plot_male_female:
        plt.vlines(df_1_male['Weighted Mean'][0], 0, 150, color='orange', label=f'Male Weighted Average = {df_1_male["Weighted Mean"][0]}', linestyle='dashed')
        plt.vlines(df_1_female['Weighted Mean'][0], 0, 150, color='gold', label=f'Female Weighted Average = {df_1_female["Weighted Mean"][0]}', linestyle='dashed')
    plt.vlines(df_1['Weighted Mean'][0], 0, 150, color='red', label=f'Weighted Average = {df_1["Weighted Mean"][0]:.2f}')
    plt.vlines(MEAN_SEC_FIFTH, 0, 150, color='blue', label=f'Mean of 2nd Quintile = {MEAN_SEC_FIFTH}', linestyle='dotted')
    
    plt.title(f'Upward Mobility, {len(upward_mobility)} Counties')
    plt.xlabel('Income at Age 35 Given Parent Income in the Lowest 25%')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('results/upward_mobility_hist_correct.pdf')

## perform cluster analysis ##
def create_X_y(df_path='data/combined.csv', make_hist=False, save_data=True, name=None):
    '''create X and y for clustering analysis'''

    df = pd.read_csv(df_path)
    # randomly shuffle the data
    df = df.sample(frac=1, random_state=47)
    if name is not None:
        y = df['income']
    else:
        y = df['Household_Income_at_Age_35_rP_gP_p25']
    print('len y:', len(y))
    if name is not None:
        X = df.drop(columns=['county_name', 'county', 'income'])
    else:
        X = df.drop(columns=['county_name', 'county', 'Household_Income_at_Age_35_rP_gP_p25'])

    # drop all se columns
    # X = X.drop(columns=[col for col in X.columns if 'se' in col])

    
    print(X.columns, len(X.columns))
    print(X.columns)

    if make_hist:
        # make histograms of the data
        fig, ax = plt.subplots(6, 4, figsize=(15, 20))
        ax = ax.ravel()
        print(X.T.size)
        for i, col in enumerate(X.columns):
            ax[i].hist(X[col], bins=100, edgecolor='black')
            ax[i].set_title(f'{X.columns[i]}')
            ax[i].set_xlabel('Value')
            ax[i].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('results/combined_hist_notnormalized.pdf')

    # normalize all columns of X 
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    if make_hist:
        # make histograms of the data
        fig, ax = plt.subplots(6, 4, figsize=(15, 20))
        ax = ax.ravel()
        print(X.T.size)
        for i, col in enumerate(X.columns):
            ax[i].hist(X[col], bins=100, edgecolor='black')
            ax[i].set_title(f'{X.columns[i]}')
            ax[i].set_xlabel('Value')
            ax[i].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('results/combined_hist_normalized.pdf')

    print('X columns', X.columns, len(X.columns))

     # convert to numpy
    X = X.to_numpy()
    y = y.to_numpy()
    X = X.astype(float)
    y = y.astype(float)

    # save X and y
    if save_data:
        if name is not None:
            np.save(f'data/X_{name}.npy', X)
            np.save(f'data/y_{name}.npy', y)
        else:
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

    col_names = ['num_below_p50', 'pop2018', 'ec_county', 'ec_se_county',
       'child_ec_county', 'child_ec_se_county', 'ec_grp_mem_county',
       'ec_high_county', 'ec_high_se_county', 'child_high_ec_county',
       'child_high_ec_se_county', 'ec_grp_mem_high_county',
       'exposure_grp_mem_county', 'exposure_grp_mem_high_county',
       'child_exposure_county', 'child_high_exposure_county',
       'bias_grp_mem_county', 'bias_grp_mem_high_county', 'child_bias_county',
       'child_high_bias_county', 'clustering_county', 'support_ratio_county',
       'volunteering_rate_county', 'civic_organizations_county']

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
        print(f'Best Parameters: {list(params)}')
        
        # get top 5 parameters
        top_5 = np.argsort(params)[-5:]
        print(f'Top 5 Parameters: {params[top_5]}')
        # print out the corresponding column names
        print(f'Top 5 Column Names: {[col_names[i] for i in top_5]}')


        np.save(f'data/best_params_{overlap_percentage}.npy', params)
        np.save(f'data/best_clusters_{overlap_percentage}.npy', clusters)
        np.save(f'data/best_modularity_{overlap_percentage}.npy', modularity)
        return overlap_percentage, clusters, modularity, params
    
     # get top 5 parameters
    top_5 = np.argsort(params)[-5:]
    print(f'Top 5 Parameters: {params[top_5]}')
    # print out the corresponding column names
    print(f'Top 5 Column Names: {[col_names[i] for i in top_5]}')


    np.save(f'data/best_params_{overlap_percentage}.npy', params)
    np.save(f'data/best_clusters_{overlap_percentage}.npy', clusters)
    np.save(f'data/best_modularity_{overlap_percentage}.npy', modularity)
    return overlap_percentage, clusters, modularity, params

## XGBRegressor ##
def prep_data_XGB(X, y, p=0.7):
    # assign labels to X
    # labels = [ 'pop2018', 'ec', 'ec_se',
    # 'child_ec', 'child_ec_se', 'ec_grp_mem',
    # 'ec_high', 'ec_high_se', 'child_high_ec',
    # 'child_high_ec_se', 'ec_grp_mem_high',
    # 'exposure_grp_mem', 'exposure_grp_mem_high',
    # 'child_exposure', 'child_high_exposure',
    # 'bias_grp_mem', 'bias_grp_mem_high', 'child_bias',
    # 'child_high_bias', 'clustering', 'support_ratio',
    # 'volunteering_rate', 'civic_organizations']
    labels = ['Population in 2018', 'Economic Connectedness', 'Economic Connectedness Standard Error', 'Child Economic Connectedness', 'Child Economic Connectedness Standard Error', 'Economic Connectedness by Group Membership', 'Economic Connectedness by High Income', 'Economic Connectedness by High Income Standard Error', 'Child Economic Connectedness by High Income', 'Child Economic Connectedness by High Income Standard Error', 'Economic Connectedness by Group Membership by High Income', 'Exposure Group Membership', 'Exposure by Group Membership High Income', 'Child Exposure', 'Child Exposure High Income', 'Bias Group Membership', 'Bias by Group Membership High Income', 'Child Bias', 'Child Bias by High Income', 'Clustering', 'Support Ratio', 'Volunteering Rate', 'Civic Organizations']

    # remove _se columns
    # labels = [label for label in labels if 'se' not in label]

    
    # binary classification: did county get out of bottom quintile?
    # quintiles = [28007, 55000, 89744, 149131]
    y = np.where(y>MEAN_SEC_FIFTH, 1, 0)

    # y = np.digitize(y, quintiles)
    # print(np.unique(y, return_counts=True))

    # check wherever there is a nan in some column per row in X, remove that row
    # y = y[~np.isnan(X).any(axis=1)]
    # X = X[~np.isnan(X).any(axis=1)]

    # split so there is a consistent number of each class in the training and validation set
    # get the number of each class
    num_classes = np.unique(y, return_counts=True)[1]
    print(num_classes)
    # get the number of each class in the training and validation set

    # take log of support ratio, civic organizations
    # X[:, -1] = np.log(X[:, -1])
    # X[:, -2] = np.log(X[:, -2])
    
    if p < 1:
        X_tv, X_t, y_tv, y_t = train_test_split(
        X, 
        y, 
        test_size=1-p,  # or whatever size you want the test set to be
        stratify=y,     # This ensures stratification
        random_state=42 # for reproducibility
        )
    else:
        X_tv = X
        y_tv = y
        X_t = X
        y_t = y

    # get weights
    w_tv = X_tv[:, 0]
    w_t = X_t[:, 0]
    X_tv = X_tv[:, 1:]
    X_t = X_t[:, 1:]

    w_tv = np.where(w_tv > 0, w_tv, 0)
    w_t = np.where(w_t > 0, w_t, 0)

    # check number unique for y_tv and y_t
    print(np.unique(y_tv, return_counts=True))
    print(np.unique(y_t, return_counts=True))

    # create data matrix
    dtrain = xgb.DMatrix(X_tv, label=y_tv, weight=w_tv)
    dtrain.feature_names = labels
    dtest = xgb.DMatrix(X_t, label=y_t, weight=w_t)
    dtest.feature_names = labels

    return dtrain, dtest, y_t

def run_XGB(X, y, save_name=None, p=0.7):
    '''perform XGB regression
    
    Params:
        X_tv (np.array): the training and validation data
        y_tv (np.array): the training and validation labels
        X_t (np.array): the test data
        y_t (np.array): the test labels
        save_name (str): the name to save the model
    
    '''
    

    dtrain, dtest, y_t = prep_data_XGB(X, y, p=p)
   

    params = {
        'max_depth': 100,
        'eta': 0.4,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 2,
    }
    

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dtest, 'test')], early_stopping_rounds=100)
    

    # make predictions
    y_pred = model.predict(dtest)
    print(np.unique(y_pred, return_counts=True))
    acc = np.sum(y_pred == y_t) / len(y_t)

    cm = confusion_matrix(y_t, y_pred)

    if save_name is not None:
         # compute accuracy
        # plot confusion matrix: y_true vs y_pred
        print(cm)
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Confusion Matrix, Accuracy: {acc:.2f}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks([0, 1], ['Q1', 'Q2'])
        plt.yticks([0, 1], ['Q1', 'Q2'])
        # add text for percent values
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                print(np.sum(cm[:,j]))
                # jth col
                print(cm[:j])
                percent = f'{cm[i, j] / np.sum(cm[:,j]):.2f}'
                # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
                plt.text(j, i, percent, ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
        plt.savefig(f'results/xgb_{save_name}.pdf')
        model.save_model(f'models/xgb_{save_name}.json')

        # plot the importance matrix
        xgb.plot_importance(model, grid=False, importance_type='weight')
        plt.tight_layout()
        plt.savefig(f'results/xgb_importance_{save_name}.pdf')

    return model, acc, cm

def run_XGB_MF(X_male, y_male, X_female, y_female, p=0.7):
    '''performs the XGB classification and compares outputs'''
    
    model_male, acc_male, cm_male = run_XGB(X_male, y_male, p=p)
    model_female, acc_female, cm_female = run_XGB(X_female, y_female, p=p)

    # compare their CMs
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.ravel()
    cax0 = ax[0].imshow(cm_male, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax0, ax=ax[0])
    cax1 = ax[1].imshow(cm_female, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax1, ax=ax[1])

    thresh = cm_male.max() / 2
    for i in range(cm_male.shape[0]):
        for j in range(cm_male.shape[1]):
            percent = f'{cm_male[i, j] / np.sum(cm_male[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[0].text(j, i, percent, ha='center', va='center', color='white' if cm_male[i, j] > thresh else 'black')

    thresh = cm_female.max() / 2
    for i in range(cm_female.shape[0]):
        for j in range(cm_female.shape[1]):
            percent = f'{cm_female[i, j] / np.sum(cm_female[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[1].text(j, i, percent, ha='center', va='center', color='white' if cm_female[i, j] > thresh else 'black')

    ax[0].set_title(f'Confusion Matrix, Accuracy: {acc_male:.2f}')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_xticks([0, 1], ['Q1', 'Q2'])
    ax[0].set_yticks([0, 1], ['Q1', 'Q2'])


    ax[1].set_title(f'Confusion Matrix, Accuracy: {acc_female:.2f}')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')
    ax[1].set_xticks([0, 1], ['Q1', 'Q2'])
    ax[1].set_yticks([0, 1], ['Q1', 'Q2'])

    plt.tight_layout()
    plt.savefig(f'results/xgb_comparison.pdf')

    # Extract feature importance for both models
    importance_male = model_male.get_score(importance_type='weight')
    importance_female = model_female.get_score(importance_type='weight')

    # Convert to DataFrame for easier handling
    df_importance_male = pd.DataFrame({'Feature': list(importance_male.keys()), 'Importance male': list(importance_male.values())})

    df_importance_female = pd.DataFrame({'Feature': list(importance_female.keys()), 'Importance female': list(importance_female.values())})

    df_merged = pd.merge(df_importance_male, df_importance_female, on='Feature', how='outer').fillna(0)

    # sort by male importance
    df_merged = df_merged.sort_values(by='Importance male', ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set the positions and width for the bars
    positions = range(len(df_merged))
    width = 0.4

    # Plotting both importances
    plt.bar([p - width/2 for p in positions], df_merged['Importance male'], width, label='Male')
    plt.bar([p + width/2 for p in positions], df_merged['Importance female'], width, label='Female')

    # Adding labels, title, and legend
    plt.xticks(positions, df_merged['Feature'], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance (F-Score)')
    plt.title('Feature Importance Comparison')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('results/feature_importance_comparison.pdf')

def prep_data_reg(X, y, p):
    # remove nans
    # X = X[~np.isnan(X)]
    # y = y[~np.isnan(y)]
    y = np.where(y>MEAN_SEC_FIFTH, 1, 0)
    # y = np.digitize(y, quintiles)
    # print(np.unique(y, return_counts=True))

    # check wherever there is a nan in some column per row in X, remove that row
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]
    

    # split so there is a consistent number of each class in the training and validation set
    # get the number of each class
    num_classes = np.unique(y, return_counts=True)[1]
    print(num_classes)
    # get the number of each class in the training and validation set

    # take log of support ratio, civic organizations
    # X[:, -1] = np.log(X[:, -1])
    # X[:, -2] = np.log(X[:, -2])
    
    if p < 1:
        X_tv, X_t, y_tv, y_t = train_test_split(
        X, 
        y, 
        test_size=1-p,  # or whatever size you want the test set to be
        stratify=y,     # This ensures stratification
        random_state=42 # for reproducibility
        )
    else:
        X_tv = X
        y_tv = y
        X_t = X
        y_t = y

    # get weights
    w_tv = X_tv[:, 0]
    w_t = X_t[:, 0]
    X_tv = X_tv[:, 1:]
    X_t = X_t[:, 1:]

    w_tv = np.where(w_tv > 0, w_tv, 0)
    w_t = np.where(w_t > 0, w_t, 0)

    return X_tv, X_t, y_tv, y_t, w_tv, w_t

def run_regression(X, y, p= 0.7):
    '''run regression to determine relative importance of features'''
    
    
    X_tv, X_t, y_tv, y_t, w_tv, w_t = prep_data_reg(X, y, p)

    # create the model; 
    def fit_model_r2(X, y, w, X_t, y_t, w_t):
        X = deepcopy(X)
        X = sm.add_constant(X)

        # drop nans
        # X = X[~np.isnan(X)]
        # y = y[~np.isnan(X)]
        # w = w[~np.isnan(X)]

        X_t = sm.add_constant(X_t)
        # X_t = X_t[~np.isnan(X_t)]
        # y_t = y_t[~np.isnan(X_t)]
        # w_t = w_t[~np.isnan(X_t)]

        model = sm.WLS(y, X, weights=w).fit()

        # Using robust standard errors (e.g., HC1)
        robust_model = model.get_robustcov_results(cov_type='HC1')

        # get accuracy on test set
        pred = robust_model.predict(X_t)
        pred = np.where(pred > 0.5, 1, 0)
        acc = np.sum(pred == y_t) / len(y_t)

        # get confusion matrix
        cm = confusion_matrix(y_t, pred)

        # get R^2
        r2 = robust_model.rsquared
        return r2, acc, cm, robust_model
    
    r2_total, total_acc, total_cm, total_model = fit_model_r2(X_tv, y_tv, w_tv, X_t, y_t, w_t)

    # get performance
    
    # now start adding features one by one and see how the R^2 changes
    r2_features = []
    acc_features = []
    for i in range(X_tv.shape[1]):
        # add the ith feature
        r2, acc, _, _ = fit_model_r2(X_tv[:, i], y_tv, w_tv, X_t[:, i], y_t, w_t)
        r2_features.append(r2)
        acc_features.append(acc)

    # calculate fraction of R^2
    r2_frac = np.array(r2_features) / r2_total
    # create a zip with the names of the features
    # labels = [ 'pop2018', 'ec', 'ec_se',
    #    'child_ec', 'child_ec_se', 'ec_grp_mem',
    #    'ec_high', 'ec_high_se', 'child_high_ec',
    #    'child_high_ec_se', 'ec_grp_mem_high',
    #    'exposure_grp_mem', 'exposure_grp_mem_high',
    #    'child_exposure', 'child_high_exposure',
    #    'bias_grp_mem', 'bias_grp_mem_high', 'child_bias',
    #    'child_high_bias', 'clustering', 'support_ratio',
    #    'volunteering_rate', 'civic_organizations']
    labels = ['Population in 2018', 'Economic Connectedness', 'Economic Connectedness Standard Error', 'Child Economic Connectedness', 'Child Economic Connectedness Standard Error', 'Economic Connectedness by Group Membership', 'Economic Connectedness by High Income', 'Economic Connectedness by High Income Standard Error', 'Child Economic Connectedness by High Income', 'Child Economic Connectedness by High Income Standard Error', 'Economic Connectedness by Group Membership by High Income', 'Exposure Group Membership', 'Exposure by Group Membership High Income', 'Child Exposure', 'Child Exposure by High Income', 'Bias Group Membership', 'Bias by Group Membership High Income', 'Child Bias', 'Child Bias by High Income', 'Clustering', 'Support Ratio', 'Volunteering Rate', 'Civic Organizations']
    
    # labels = [label for label in labels if 'se' not in label]

    r2_frac_dict = dict(zip(labels, zip(r2_frac, acc_features)))
    return r2_frac_dict, total_acc, total_cm, total_model
   
def run_regression_MF(X_male, y_male, X_female, y_female, p=0.7):
    '''run regression to determine relative importance of features for males and females separately'''

    r2_frac_dict_male, acc_male, cm_male, model_male = run_regression(X_male, y_male)

    r2_frac_dict_female, acc_female, cm_female, model_female = run_regression(X_female, y_female)

    # plot the R^2 fractions. sort male by greatest to least. plot female in same order
    # sort male by greatest to least
    r2_frac_dict_male_sorted = {k: v for k, v in sorted(r2_frac_dict_male.items(), key=lambda item: item[1], reverse=True)}

    # use the same order as above for females
    r2_frac_dict_female_sorted = {k: r2_frac_dict_female[k] for k in r2_frac_dict_male_sorted.keys()}

    # Unpacking the items for plotting
    male_labels, male_values = zip(*r2_frac_dict_male_sorted.items())
    female_values = [r2_frac_dict_female_sorted[k] for k in male_labels]

    # r2 is first element, acc is second
    r2_frac_male = [v[0] for v in male_values]
    accs_male = [v[1] for v in male_values]
    r2_frac_female = [v[0] for v in female_values]
    accs_female = [v[1] for v in female_values]

    # plot
    fig, ax = plt.subplots()

       # Set the positions and width for the bars
    positions = range(len(r2_frac_female))
    width = 0.4

    # Plotting both importances
    plt.bar([p - width/2 for p in positions], r2_frac_male, width, label='Male')
    plt.bar([p + width/2 for p in positions], r2_frac_female, width, label='Female')

    plt.xticks(positions, male_labels, rotation=90)

    # Adding labels and title
    ax.set_ylabel('Feature Importance by $R^2$')
    ax.set_title('Fraction of $R^2$ for Each Feature by Gender')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/r2_fractions.pdf')

    # compare their CMs
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.ravel()
    cax0 = ax[0].imshow(cm_male, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax0, ax=ax[0])
    cax1 = ax[1].imshow(cm_female, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax1, ax=ax[1])

    thresh = cm_male.max() / 2
    for i in range(cm_male.shape[0]):
        for j in range(cm_male.shape[1]):
            percent = f'{cm_male[i, j] / np.sum(cm_male[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[0].text(j, i, percent, ha='center', va='center', color='white' if cm_male[i, j] > thresh else 'black')

    thresh = cm_female.max() / 2
    for i in range(cm_female.shape[0]):
        for j in range(cm_female.shape[1]):
            percent = f'{cm_female[i, j] / np.sum(cm_female[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[1].text(j, i, percent, ha='center', va='center', color='white' if cm_female[i, j] > thresh else 'black')

    ax[0].set_title(f'Confusion Matrix, Accuracy: {acc_male:.2f}')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_xticks([0, 1], ['Q1', 'Q2'])
    ax[0].set_yticks([0, 1], ['Q1', 'Q2'])


    ax[1].set_title(f'Confusion Matrix, Accuracy: {acc_female:.2f}')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')
    ax[1].set_xticks([0, 1], ['Q1', 'Q2'])
    ax[1].set_yticks([0, 1], ['Q1', 'Q2'])

    plt.tight_layout()
    plt.savefig(f'results/regression_comparison.pdf')

    # # plot the accuracy comparison
    # plt.figure(figsize=(10, 6))
    # plt.plot(accs_male, label='Male')
    # plt.plot(accs_female, label='Female')
    # plt.title('Accuracy Comparison')
    # plt.xlabel('Number of Features')
    # plt.ylabel('Accuracy')

def run_regression_XGB_MF(X_male, y_male, X_female, y_female, p=0.7):
    print(p)

    model_male, xgb_acc_male, xgb_cm_male = run_XGB(X_male, y_male, p=p)
    model_female, xgb_acc_female, xgb_cm_female = run_XGB(X_female, y_female, p=p)

    r2_frac_dict_male, reg_acc_male, reg_cm_male, male_model_reg = run_regression(X_male, y_male)
    r2_frac_dict_female, reg_acc_female, reg_cm_female, female_model_reg = run_regression(X_female, y_female)

    # run the oaxaca blinder: how accurate is predicting the males with female coefficients and vice versa
    dtrain_male, dtest_male, y_t_male = prep_data_XGB(X_male, y_male, p)
    dtrain_female, dtest_female, y_t_female = prep_data_XGB(X_female, y_female, p)

    male_XGB_pred_female = model_female.predict(dtest_male)
    male_XGB_female = np.sum(male_XGB_pred_female) / len(y_t_male)
    
    female_XGB_pred_male = model_male.predict(dtest_female)
    female_XGB_male= np.sum(female_XGB_pred_male) / len(y_t_female)

    print(f'female XGB predicting males: {male_XGB_female}, male XGB predicting females: {female_XGB_male}') 

    # female predicting female and male predicting male
    # male_XGB_pred_male = model_male.predict(dtest_male)
    # male_XGB_male = np.sum(male_XGB_pred_male) / len(y_t_male)
   
    print(f'male XGB predicting males: {np.sum(y_t_male) / len(y_t_male)}, female XGB predicting females: {np.sum(y_t_female) / len(y_t_female)}') 

    print(f'male diff: {np.sum(y_t_male) / len(y_t_male) - male_XGB_female}, female diff: {np.sum(y_t_female) / len(y_t_female) - female_XGB_male}')

    # compare their CMs
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()

    cax0 = ax[0].imshow(xgb_cm_male, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax0, ax=ax[0])
    cax1 = ax[1].imshow(xgb_cm_female, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax1, ax=ax[1])

    thresh = xgb_cm_male.max() / 2
    for i in range(xgb_cm_male.shape[0]):
        for j in range(xgb_cm_male.shape[1]):
            percent = f'{xgb_cm_male[i, j] / np.sum(xgb_cm_male[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[0].text(j, i, percent, ha='center', va='center', color='white' if xgb_cm_male[i, j] > thresh else 'black')

    thresh = xgb_cm_female.max() / 2
    for i in range(xgb_cm_female.shape[0]):
        for j in range(xgb_cm_female.shape[1]):
            percent = f'{xgb_cm_female[i, j] / np.sum(xgb_cm_female[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[1].text(j, i, percent, ha='center', va='center', color='white' if xgb_cm_female[i, j] > thresh else 'black')

    ax[0].set_title(f'XGB Male Accuracy: {xgb_acc_male:.2f}')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_xticks([0, 1], ['Below Threshold', 'Above Threshold'])
    ax[0].set_yticks([0, 1], ['Below Threshold', 'Above Threshold'])


    ax[1].set_title(f'XGB Female Accuracy: {xgb_acc_female:.2f}')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')
    ax[1].set_xticks([0, 1], ['Below Threshold', 'Above Threshold'])
    ax[1].set_yticks([0, 1], ['Below Threshold', 'Above Threshold'])

    # same for regression
    cax2 = ax[2].imshow(reg_cm_male, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax2, ax=ax[2])
    cax3 = ax[3].imshow(reg_cm_female, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax3, ax=ax[3])

    thresh = reg_cm_male.max() / 2
    for i in range(reg_cm_male.shape[0]):
        for j in range(reg_cm_male.shape[1]):
            percent = f'{reg_cm_male[i, j] / np.sum(reg_cm_male[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[2].text(j, i, percent, ha='center', va='center', color='white' if reg_cm_male[i, j] > thresh else 'black')

    thresh = reg_cm_female.max() / 2
    for i in range(reg_cm_female.shape[0]):
        for j in range(reg_cm_female.shape[1]):
            percent = f'{reg_cm_female[i, j] / np.sum(reg_cm_female[:,j]):.2f}'
            # percent = f'{cm[i, j] / np.sum(cm[i]):.2f}'
            ax[3].text(j, i, percent, ha='center', va='center', color='white' if reg_cm_female[i, j] > thresh else 'black')

    ax[2].set_title(f'WLS Male Accuracy: {reg_acc_male:.2f}')
    ax[2].set_xlabel('Predicted')
    ax[2].set_ylabel('True')
    ax[2].set_xticks([0, 1], ['Below Threshold', 'Above Threshold'])
    ax[2].set_yticks([0, 1], ['Below Threshold', 'Above Threshold'])


    ax[3].set_title(f'WLS Female Accuracy: {reg_acc_female:.2f}')
    ax[3].set_xlabel('Predicted')
    ax[3].set_ylabel('True')
    ax[3].set_xticks([0, 1], ['Below Threshold', 'Above Threshold'])
    ax[3].set_yticks([0, 1], ['Below Threshold', 'Above Threshold'])

    plt.tight_layout()
    plt.savefig(f'results/regression_xgb_comparison_{p}.pdf')

    # now make twin plots for the feature importance
    # Extract feature importance for both models
    importance_male = model_male.get_score(importance_type='weight')
    importance_female = model_female.get_score(importance_type='weight')

    # Convert to DataFrame for easier handling
    df_importance_male = pd.DataFrame({'Feature': list(importance_male.keys()), 'Importance male': list(importance_male.values())})

    df_importance_female = pd.DataFrame({'Feature': list(importance_female.keys()), 'Importance female': list(importance_female.values())})

    # add the r2 fractions

    # Unpacking the items for plotting
    male_labels, male_values = zip(*r2_frac_dict_male.items())
    female_labels, female_values = zip(*r2_frac_dict_female.items())

    # r2 is first element, acc is second
    r2_frac_male = [v[0] for v in male_values]
    accs_male = [v[1] for v in male_values]
    r2_frac_female = [v[0] for v in female_values]
    accs_female = [v[1] for v in female_values]

    df_xgb_merged = pd.merge(df_importance_male, df_importance_female, on='Feature', how='outer').fillna(0)

    df_r2_male = pd.DataFrame({'Feature': male_labels, 'R2 male': r2_frac_male})
    df_r2_female = pd.DataFrame({'Feature': female_labels, 'R2 female': r2_frac_female})

    df_r2_merged = pd.merge(df_r2_male, df_r2_female, on='Feature', how='outer').fillna(0)

    df_merged = pd.merge(df_xgb_merged, df_r2_merged, on='Feature', how='outer').fillna(0)

    # round columns to 4 decimal places
    df_merged = df_merged.round(4)

    # drop index
    df_merged.to_csv('results/feature_importance_comparison_wls_xgb.csv')

    # sort by male importance xgb
    df_merged = df_merged.sort_values(by='Importance male', ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set the positions and width for the bars
    positions = range(len(df_merged))
    width = 0.4

    # Plotting both importances
    ax.bar([p - width/2 for p in positions], df_merged['Importance male'], width, label='XGB Male', alpha=0.5, color='red')
    ax.bar([p + width/2 for p in positions], df_merged['Importance female'], width, label='XGB Female', alpha=0.5, color='blue')

    # ax2 = ax.twinx()
    # ax2.bar([p - width/2 for p in positions], df_merged['R2 male'], width, label='WLS Male', alpha=0.5, color='magenta')
    # ax2.bar([p + width/2 for p in positions], df_merged['R2 female'], width, label='WLS Female', alpha=0.5, color='cyan')

    # Adding labels, title, and legend
    ax.set_xticks(positions, df_merged['Feature'], rotation=90)
    ax.set_ylabel('Number of Times Feature Used Across Trees')
    # ax2.set_ylabel('Fraction of $R^2$')

    # plt.title('Feature Importance Comparison')
    ax.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # Show plot
    plt.tight_layout()
    plt.savefig(f'results/feature_importance_comparison_xgb_{p}.pdf')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar([p - width/2 for p in positions], df_merged['R2 male'], width, label='WLS Male', alpha=0.5, color='magenta')
    ax.bar([p + width/2 for p in positions], df_merged['R2 female'], width, label='WLS Female', alpha=0.5, color='cyan')

    ax.set_xticks(positions, df_merged['Feature'], rotation=90)
    ax.set_ylabel('Fraction of $R^2$')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'results/feature_importance_comparison_wls_{p}.pdf')


if __name__ == '__main__':
    get_combined_df(um_data_path='data/cty_kfr_rP_gF_p25.csv', name='female')
    get_combined_df(um_data_path='data/cty_kfr_rP_gM_p25.csv', name='male')

    # get X, y
    create_X_y(df_path='data/combined_female.csv', save_data=True, make_hist=False, name='female')
    create_X_y(df_path='data/combined_male.csv', save_data=True, make_hist=False, name='male')

    X_male, y_male = np.load('data/X_male.npy', allow_pickle=True), np.load('data/y_male.npy', allow_pickle=True)

    X_female, y_female = np.load('data/X_female.npy', allow_pickle=True), np.load('data/y_female.npy', allow_pickle=True)

    # run_XGB(X_male, y_male, save_name='male')
    # run_XGB(X_female, y_female, save_name='female')
    # run_XGB_MF(X_male, y_male, X_female, y_female)
    # run_regression_MF(X_male, y_male, X_female, y_female)
    
    # plot_upward_mobility(plot_male_female=True)
    run_regression_XGB_MF(X_male, y_male, X_female, y_female, p=1)

    # calculate percent of data above upper bound for lowest quintile
    # print(np.sum(y_male > MEAN_SEC_FIFTH) / len(y_male), np.sum(y_female > MEAN_SEC_FIFTH) / len(y_female))


    # get_combined_df()
    # plot_hist_sum()
    # make_tables()
    # create_X_y(save_data=False, make_hist=False)
    # X, y = np.load('data/X.npy', allow_pickle=True), np.load('data/y.npy', allow_pickle=True)
    # optimize_params(X, y, n_iter=1000)
    # run_XGB(X, y, save_name='all_0')


    # params = np.ones(X.shape[1])
    # # params = np.random.uniform(0, 1, X.shape[1])
    # X_mini = X[:400]
    # y_mini = y[:400]
    # clusters = cluster(X_mini, y_mini, params, show_scatter=True, show_graph=False, param_name=f'_{len(X_mini)}_equal')