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

Probabilities of five most liberal voting judges.

<p align="center">
  <img src="https://github.com/jbravo87/Stat-Analysis/blob/2b5e7162ec7918c679caed73fda079a86f1312d3/LiberalVotesProb.png"
</p>
