# Readme file for econ 122 project.

## Reference paper: 
https://www.nature.com/articles/s41586-022-04996-4

## Updates

### 3/6/24
* after 100 random runs, got 49%
* added find best 5 params and check which columns they correspond to
    - have to fihgure out how to deal with unequal distributions... or is this a problem? thing is we want to distinguish from importance and reweighting based on distribution. *what if we try weighting based on distribution?*

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
