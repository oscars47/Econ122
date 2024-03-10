# Readme file for econ 122 project.

## Reference paper: 
https://www.nature.com/articles/s41586-022-04996-4

## Updates

### 3/10/24
* results of 1000 random: can't beat 49%
* trying XGB: SUCCESS
    - on all individuals:
        - regression:
            - RMSE $3595
            - feature importance: num_below_p50 was most important --> need to do weighting
        - classification: above or below median y
            - ec se actually most important by weight
                - what to do with standard errors??
        - classification: 2021 income quintiles: https://www.taxpolicycenter.org/statistics/household-income-quintiles
            -  nobody ever makes it beyond 3rd quintile:
                490, 2561, 38 <-- 3rd quintile too small, so only doing whether stuck in bottom quintile or got above
            - accuracy = 90%! --> but problem is that it never predicts the 3rd quintile like mentioned above
            - what if we just do whether remained in bottom quintile or escaped
                total: 490 2599
                tv: (array([0, 1]), array([ 343, 1819]))
                t: (array([0, 1]), array([147, 780]))

                *got 87% accuracy using the splits above*

                ec_se is still most important factor!

        - trying male vs female xgb since more data available
            - thing is, still same inputs
            - interestingly definitely different feature importances             

### 3/6/24
* after 100 random runs, got 49%
* added find best 5 params and check which columns they correspond to
    - have to fihgure out how to deal with unequal distributions... or is this a problem? thing is we want to distinguish from importance and reweighting based on distribution. *what if we try weighting based on distribution?*
- features:
    - pop2018: population in 2018
    - num_below_p50: Number of children with below-national-median parental household income. 
    - ec_county: two times the share of high-SES friends among low-SES individuals, averaged over all low-SES individuals in the county. 
    - child ec county: same as ec_county, but for children
    - ec grp mem county: ec but restricted to friendships allocated to the group in which they were formed (see aupplemental B1)
    - ec high county: same as ec, but avgd over high income individuals
    - child high ec county: same as above, but for children
** network effects on black vs white (do the same things help them get out of poverty desite different counties) OR female vs male (same counties, but do differet things help them get out) **
    - choose based on data availability/size
** what about XGBoost? --> feature importance**
    - try based on all data, then on whichever other one
- make plots simpler/less overwhelming

### 3/5/24
- to do
    - define similarity metric; 
        - write partitioning code
        - start MCMC
* what i did:
    - made hist and scatter plots, saved to ```proj/results```.
    - implemented leiden: ```cluster``` is main function that does the clustering; also helper functions to create the pre-processed data, 
        - really fast! about 2 mins on all objects. however plotting the graph takes a long time....
    - coloring scatter plots based on cluster identity
        - plot pop18 and the weights as well
    - comparing based on #bins = percentiles -- these are target classes. then compare `accuracy`
        - using first 400, got ~51% overlap with all 1 params
        - using all of them, got 47% -> ~44% since have to take into account how to compare clusters to percentiles
    - implemented random search, protected against keyboard error; running on desktop
    - added X, y, combined.csv to ```proj/data/```
    - added mapping for old cluster labels to new based on an argsort of the means of y per old cluster



### 3/4/24
* started file cluster.py
    - to do: 
        - read in data: 
            - print basic stats / get histograms for overall distibriutions
                - weights are: num_below_p50 (number of children w below national median parental household income)
            - color hist by upward mobility, which is avg income in adulthood from low-income

        

    - what i did
        - using county data since would need to aggregate tract data into ZIP codes... plus then it should be faster since only 3089 counties with data.
            - upward mobility data: by county, household income for children at 35 given they were born into family in bottom 20%
