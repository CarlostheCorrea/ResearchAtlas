# A Comparison of Resampling and Recursive Partitioning Methods in Random Forest for Estimating the Asymptotic Variance Using the Infinitesimal Jackknife

## Response

The key findings of the research are as follows:

1. Using subsampling instead of bootstrap sampling significantly improves the accuracy of prediction variance estimation in random forests. This improvement is consistent across different sample sizes, tree types, and distributions.

2. Conditional inference (CI) trees consistently result in a lower mean absolute predictive bias (MAPB) compared to traditional classification and regression trees (CART), regardless of the sample size or resampling method used.

3. The resampling method has the largest impact on MAPB, with subsampling outperforming bootstrap sampling in all scenarios tested.

4. Increasing the number of variables tried at each split (mtry) generally increases MAPB for certain distributions (OR and AND), especially when using bootstrap resampling.

5. For SUM and SQ distributions, increasing the sample size decreases MAPB, while for OR and AND distributions, increasing the sample size increases MAPB, particularly with bootstrap resampling.

6. The study suggests that CI trees and subsampling should be preferred for more accurate variance estimation in random forests.

These findings extend previous work by Wager and Athey (2015) by exploring different distributions, mtry values, and including auxiliary noise variables in the training sets.
