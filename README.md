## Stat Analysis

This repository is intended to document some statistical analysis of real-life data pertaining to an assortment of topics.

### Judge Voting

I found this article from FiveThirtyEight about [Judge Gorsuch's Record.](https://fivethirtyeight.com/features/for-a-trump-nominee-neil-gorsuchs-record-is-surprisingly-moderate-on-immigration/) This data set was chosen because of the binary nature of voting. Judges vote either conservatively or liberally.
This allows for use of classification methods to perform statistical analysis. Specifically, Logistical Regression will be invoked.

The following plot depicts the voting rate (conservative vote/total votes) of the six most conservative judges (per total votes) plus Gorsuch. 

<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/21417d364ad487e42bf6ea56e5f9ff82d9afdc56/Top6MostConservativeRates.png">
</p>

Next is the attempted Interquartile Range of the seven most liberal voting judges. Depocted through a boxplot.

<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/2b5e7162ec7918c679caed73fda079a86f1312d3/IQRLiberalJudges.png">
</p>
<!--
your comment goes here
and here
-->
<!---
Probabilities of five most liberal voting judges.

<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/2b5e7162ec7918c679caed73fda079a86f1312d3/LiberalVotesProb.png"
</p>
-->
### NBA Players (k-Means)

Obtained data from data.world pertaining to NBA players. Took averages of made 3PT field goals and blocks. Wanted to invole the ideals of data segmentation to see any relationship between average blocks and made three-pointers. Used the Elbow Method and Silhouette Analysis to determine optimal k.
Completed goal of visualizing clusters formed by k-Means algorithm. Included the centroids.
<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/4711fe3bb50591e0def3114cb6ddba756d78a40c/blocksmade3ptsclustering.png"
</p>

The following is a visual of the silhouette analysis to determine the optimal K for this dataset.
<p align="center">
https://github.com/jbravo87/Stat-Analysis/blob/2ac0fc66027af4cefa01720c0a14bc8ee0e47c48/SilhouetteAnalysis.png
</p>

## Planetary Data Analysis

Obtained data from NASA Exoplanet Archive. Initial idea was to find some real-world data to first run ditributional tests. After some cleaning this first plot shows the raw distribution of the orbital periods of these exoplanets.
<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/2ac0fc66027af4cefa01720c0a14bc8ee0e47c48/hist_orbitalperiod.png"
</p>

This next plot is the eccentricity column in a bootstrap 90% confidence interval.
<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/2ac0fc66027af4cefa01720c0a14bc8ee0e47c48/bootstrap_ci.png"
</p>

This last plot is scatterplot of eccentricity as a function of orbital period.
<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/2ac0fc66027af4cefa01720c0a14bc8ee0e47c48/scatter_orbper_ecce.png"
</p>