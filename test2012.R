library(tidyverse)
library(gridExtra)
library(grid)

#read 2012 data and delete CO
data2012 <- read.csv("gt_2012.csv")
data2012 <- data2012[-10]

#Find mean, median, percentile 1, percentile 99, standard deviation and range for each variable
meanVector <- vector("double", ncol(data2012))
medianVector <- vector("double", ncol(data2012))
perc1 <- vector("double", ncol(data2012))
perc99 <- vector("double", ncol(data2012))
stdVector <- vector("double", ncol(data2012))
rangeVector <- vector("double", ncol(data2012))
for (i in seq_along(data2012)){
  meanVector[[i]] <- mean(data2012[[i]])
  medianVector[[i]] <- median(data2012[[i]])
  perc1[[i]] <- quantile(data2012[[i]], probs = c(0.01))
  perc99[[i]] <- quantile(data2012[[i]], probs = c(0.99))
  stdVector[[i]] <- sd(data2012[[i]])
  rangeVector[[i]] <- max(data2012[[i]]) - min(data2012[[i]])
}

#correlation matrix using spearman rank correlation rounded
res <- cor(data2011, method = "spearman")
round(res, 2)

#format data
meanVector = format(meanVector, digits = 3)
medianVector = format(medianVector, digits = 3, nsmall = 2)
perc1 = format(perc1, digits = 3)
perc99 = format(perc99, digits = 3)
stdVector = format(stdVector, digits = 2)
rangeVector = format(rangeVector, digits = 3)

#create a table for the descriptive statistics
test2012 <- data.frame(meanVector, medianVector, perc1, perc99, stdVector, rangeVector)
colnames(test2012) <- c("Mean", "Median", "Percentile 1", "Percentile 99", "Standard Deviation", "Range")
rownames(test2012) <- c("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "NOX")


grid.table(test2012)