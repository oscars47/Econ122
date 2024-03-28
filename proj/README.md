# Readme file for econ 122 project.

## Reference paper: 
https://www.nature.com/articles/s41586-022-04996-4

## Updates

## 3/27/24
* intepreting male-female difference.
    - predict not accuracy but predicted percentage. hope is that goes a lot closer to females than male; difference in coefficients is why women are more likely to escape; and then do other way. if women acted like men, how much fewer of them
        ```
        female XGB predicting males: 0.679184202007122, male XGB predicting females: 0.5412754936872775
        male XGB predicting males: 0.5412754936872775, female XGB predicting females: 0.610877306571706
        male diff (true - predicted with reverse model): -0.198769828423438, female diff (true - predicted with reverse model): 0.06960181288442857
        ```
        - interpreting this: We see that if the male input values were treated like females, roughly 20% more should succeed than actually do. Moreover, the male model predicts about 7% fewer females should succeed than actually do. This analysis reveals that females are rewarded more for social capital than men are.

    - put the 0.7 in appendix (?) or just make a footnote about econ convention
    - need fixed labels.
    - add min/max for data tables

## 3/25/24
* wrote lit review, from that gathered that network effects should be more pronounced for men than women, contrary to my results
* steinberger fix data section:
    - rerunning plots for models as well as predicting with males with female model and vice versa. running p=.7 and p=1

        p=0.7
            - ```female XGB predicting males: 0.6828478964401294, male XGB predicting females: 0.7454153182308522
            male XGB predicting males: 0.7411003236245954, female XGB predicting females: 0.7874865156418555
            male diff: 0.058252427184465994, female diff: 0.04207119741100329```

        p=1
            - ```female XGB predicting males: 0.7325995467788928, male XGB predicting females: 0.710262220783425
            male XGB predicting males: 0.7584978957591454, female XGB predicting females: 0.8138556167044351
            male diff: 0.02589834898025256, female diff: 0.10359339592101002```

## 3/22/24
* meeting
    - analysis is on counties
        - how to see working women **
    - clarify the se is smaller for smaller
    - run females through the males to see how well prediction matches up
    - se makes sense because it is a measure of cliques.
    - what steinberger wants:
        - run on all data then report test separately
        - put WLS first so we establish what economists are familiar w
        - then show how XGB exceeds: in cm, note how male being linear really fails whereas for females linear does not as bad, comparing the diagonal percentages
        - make the feature importance 2 separate plots; discuss how this gives us insight
        
## 3/18/24
* got tables of data for paper
    - i forgot the social capital data is the same for males and females since it is by county and we can't get gender; what does change is the upward mobility
    - interestingly, females have a higher upward mobility than males

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
