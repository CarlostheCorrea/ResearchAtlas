# A Comparison of Resampling and Recursive Partitioning Methods in Random Forest for Estimating the Asymptotic Variance Using the Infinitesimal Jackknife

## Key Findings

1. **Resampling Method Impact**: The resampling method had the largest impact on the mean absolute predictive bias (MAPB). Subsampling resulted in a more accurate estimation of prediction variance compared to bootstrap sampling. This was consistent across different distributions and sample sizes.
2. **Tree Type**: Conditional inference (CI) trees consistently resulted in a lower MAPB than classification and regression trees (CART), regardless of the sample size, resampling method, or mtry used.
3. **Sample Size Effects**: For the SUM and SQ distributions, increasing the sample size decreased the MAPB. However, for the OR and AND distributions, increasing the sample size increased the MAPB, especially when using bootstrap resampling.
4. **mtry Effects**: Increasing mtry generally increased MAPB for the OR and AND distributions, particularly when using bootstrap resampling. The effect was less pronounced for the SUM and SQ distributions.
5. **Overall Performance**: Subsampling consistently outperformed bootstrap sampling, even in the worst-case scenarios for subsampling compared to the best-case scenarios for bootstrap sampling.
6. **Computational Considerations**: The study noted computational limitations when using CI trees with large sample sizes due to the lack of a C implementation for CI random forests. These findings suggest that using CI trees and subsampling can significantly improve the accuracy of prediction variance estimates in random forests.
