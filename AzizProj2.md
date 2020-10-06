ST558 R Project II
================
Mana Azizsoltani
October 16, 2020

  - [Introduction](#introduction)
  - [Data](#data)
  - [Summarization](#summarization)
  - [Modeling](#modeling)

# Introduction

## Online News Data Set

The online news data set that we will be working with for this project
was found on the UCI Machine Learning Repository. The data itself comes
from the popular online news website, Mashable. Each observation of the
data set represents a single article released by Mashable during the
period from January 7, 2013 and January 7, 2015. Each of the 39644
articles has 60 attributes that describes the different aspects of each
article.

## Variable Descriptions

The variables that I will be using are the following (more on selection
later):

7.  num\_hrefs: Number of links  
8.  data\_channel\_is\_entertainment: Is data channel ‘Entertainment’?  
9.  data\_channel\_is\_socmed: Is data channel ‘Social Media’?  
10. data\_channel\_is\_tech: Is data channel ‘Tech’?  
11. data\_channel\_is\_world: Is data channel ‘World’?  
12. kw\_min\_min: Worst keyword (min. shares)  
13. kw\_min\_avg: Avg. keyword (min. shares)  
14. kw\_max\_avg: Avg. keyword (max. shares)  
15. kw\_avg\_avg: Avg. keyword (avg. shares)  
16. weekday\_is\_saturday: Was the article published on a Saturday?  
17. weekday\_is\_sunday: Was the article published on a Sunday?  
18. is\_weekend: Was the article published on the weekend?  
19. LDA\_00: Closeness to LDA topic 0  
20. LDA\_01: Closeness to LDA topic 1  
21. LDA\_02: Closeness to LDA topic 2  
22. LDA\_04: Closeness to LDA topic 4  
23. global\_subjectivity: Text subjectivity  
24. global\_sentiment\_polarity: Text sentiment polarity  
25. rate\_negative\_words: Rate of negative words among non-neutral
    tokens  
26. title\_subjectivity: Title subjectivity  
27. shares: Number of shares (target)

## Purpose and Methods

The purpose of this analysis is to predict the number of shares of a
particular article using tree-based machine learning techniques. In
particular, we will be using a normal (non-ensemble) regression tree and
a boosted tree model to attempt to predict the number of shares. This
process will be done for each day of the week.

# Data

## Reading in the Data

First things first, we need to read in the data and filter the data for
Monday.

``` r
# Read in data set
dat <- read_csv("OnlineNewsPopularity.csv") %>% dplyr::select(-url)

# Subset the data set by day
dayvar <- as.name(paste0("weekday_is_", params$day))
day.dat <- dat %>% filter((!!sym(dayvar)) == 1)
```

## Variable Selection

After reading the article attached to the UCI Machine Learning website
where we got the data, I decided to use the same technique as Dr. Ren
and Dr. Yang as discussed in [their
paper](http://cs229.stanford.edu/proj2015/328_report.pdf). They
calculated the Fisher score for each feature and selected the 20 with
the highest Fisher scores. The Fisher score for the \(j^{\text{th}}\)
feature is given by:  
\[F(j) =  \frac{(\bar{x}^1_j - \bar{x}^2_j)^2}{(s^1_j)^2 + (s^2_j)^2},\]  
where  
\[(s^k_j)^2 = \sum_{x \in X^k}(x_j - \bar{x}^k_j)^2\]

The top variables with the highest Fisher scores were the following:  
<img src="/Users/mana010/Downloads/fscores.png" width="570" /> All that
being said, I used these 20 variables as my predictors in my models.
This is also consistent with what we learned in the lectures about how
many predictors to use in a regression tree, which is
\(\frac{1}{3}(\text{# of predictors})\) = 20 in this case.

## Data Partitioning

Before creating the models, we must split the data into a training and
test data set in order to later evaluate the model’s prediction
accuracy. In this case we will be using a 70/30 split, training the data
on the 70% and testing the trained models on the 30%.

``` r
vars <- c(7, 14, 16:19, 25:27, 36:41, 43:45, 49, 56, 60)
index <- createDataPartition(day.dat$shares, p = .7, list = F) %>% as.vector()
train <- day.dat[index,vars]
test <- day.dat[-index,vars]
```

# Summarization

To get an idea of what we are working with when considering this data,
we want to get some summaries of the data. Before looking at the
training data, I wanted to see how much of the data pertained to each
day.  
![](AzizProj2_files/figure-gfm/plot1-1.png)<!-- -->

Next, I went ahead and looked at summaries of the four variables with
the highest Fisher Scores. The case of the quantitative variables, I
plotted them on a scatter plot against `shares`, and for the qualitative
variables, I just whipped up a bar plot to get a breakdown of the
training data.

    ## [1] 690400

![](AzizProj2_files/figure-gfm/sctrs-1.png)<!-- -->

Then I looked at the numeric summaries of some of the quantitative
variables as well as contingency tables of the qualitative variables.

    ##    num_hrefs        kw_avg_avg        LDA_02        global_subjectivity
    ##  Min.   :  0.00   Min.   :    0   Min.   :0.01819   Min.   :0.0000     
    ##  1st Qu.:  4.00   1st Qu.: 2359   1st Qu.:0.02857   1st Qu.:0.3941     
    ##  Median :  7.00   Median : 2838   Median :0.04000   Median :0.4509     
    ##  Mean   : 10.74   Mean   : 3066   Mean   :0.20766   Mean   :0.4408     
    ##  3rd Qu.: 13.00   3rd Qu.: 3531   3rd Qu.:0.31845   3rd Qu.:0.5045     
    ##  Max.   :162.00   Max.   :43568   Max.   :0.92000   Max.   :1.0000     
    ##  global_sentiment_polarity
    ##  Min.   :-0.38021         
    ##  1st Qu.: 0.05647         
    ##  Median : 0.11827         
    ##  Mean   : 0.11778         
    ##  3rd Qu.: 0.17562         
    ##  Max.   : 0.56667

|   | is ent. | is soc. med. | is tech | is world |
| :- | ------: | -----------: | ------: | -------: |
| 0 |    3731 |         4440 |    3793 |     3702 |
| 1 |     933 |          224 |     871 |      962 |

# Modeling

As mentioned above, I will be running a basic, non-ensemble regression
tree as well as a boosted tree in order to try and predict the number of
shares of a particular article. For both models, I used the `caret`
package in R to all the heavy lifting of cross-validation and tuning for
me.

## Regression Tree

Firstly, I went ahead and fit the basic, non-ensemble regression tree
model on the training data. The typical regression tree is split using
recursive binary splitting, meaning that for every possible value of
each predictor, it finds the SSR and tries to minimize it. After growing
a large tree (many splits), the tree is pruned back using
cost-complexity pruning, which is done to prevent overfitting the data.
I chose the Cp value (tuning parameter for pruning) using the default
tune grid from the `train()` function in `caret`. For this model, I used
Leave-One-Out Cross Validation (LOOCV) as instructed in the project
outline. I didn’t bother scaling the data since for a basic regression
tree it doesn’t really make a difference. After running the model, I
used it to predict the observations in the test data set and calculate
the Root Mean Square Error (RMSE).

``` r
# Specify CV method
trctrl <- trainControl(method = "LOOCV")

# Normal Regression Tree
treeFit <- train(shares ~., data = train, method = "rpart",
                 trControl=trctrl)
treePred <- predict(treeFit, newdata = test)
treeRMSE <- sqrt(mean((treePred-test$shares)^2))
```

The final regression tree model that I selected was the model with a Cp
value of 0.00302.

## Boosted Tree

After fitting the basic regression tree, we can go ahead and fit the
boosted tree model on the training data. Boosting is an ensemble
tree-based model in which the trees are slowly trained to avoid
overfitting. The trees are grown sequentially, so that each subsequent
tree is grown on a modified version of the original data. The
predictions are then updated using the residuals as the trees are grown.
I used a 5-fold cross validation to evaluate the boosted tree model.
Unlike the previous model, I went ahead and centered and scaled the data
in the pre-processing stage. After running the model, I used it to
predict the observations in the test data set and calculate the Root
Mean Square Error (RMSE).

``` r
# Boosted Regression Tree
trctrl2 <- trainControl(method = "cv", number = 5)
boostFit <- train(shares ~., data = train, method = "gbm",
                  trControl = trctrl2, preProcess = c("center", "scale"),
                  verbose = FALSE)
```

    ## Warning in preProcess.default(method = c("center", "scale"), x =
    ## structure(c(3, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

    ## Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19,
    ## uniqueCut = 10, : These variables have zero variances: weekday_is_saturday,
    ## weekday_is_sunday, is_weekend

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 10: weekday_is_saturday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 11: weekday_is_sunday has no variation.

    ## Warning in (function (x, y, offset = NULL, misc = NULL, distribution =
    ## "bernoulli", : variable 12: is_weekend has no variation.

``` r
boostPred <- predict(boostFit, newdata = test)
boostRMSE <- sqrt(mean((boostPred-test$shares)^2))
```

The final boosted tree model that I selected was the model with the
following tune of the parameters:

    ##   n.trees interaction.depth shrinkage n.minobsinnode
    ## 3     150                 1       0.1             10

## Model Comparison

|            |      RMSE |
| :--------- | --------: |
| Reg. Tree  | 10455.195 |
| Boost Tree |  7888.102 |

Comparison of Models’ RMSE

After evaluating the predictions of the test data set from each model,
the basic regression tree had an RMSE of 1.045519510^{4} and the boosted
tree model had an RMSE of 7888.1023407. I expected the boosted tree to
have a lower RMSE, since we learned that ensemble trees tend to
outperform single trees in terms of prediction.
