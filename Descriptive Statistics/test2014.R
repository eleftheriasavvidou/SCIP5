library(tidyverse)
library(gridExtra)
library(grid)
library(corrplot)

#read 2014 data and delete CO
data2014 <- read.csv("gt_2014.csv")
data2014 <- data2014[-10]

#Find mean, median, percentile 1, percentile 99, standard deviation and range for each variable
meanVector <- vector("double", ncol(data2014))
medianVector <- vector("double", ncol(data2014))
perc1 <- vector("double", ncol(data2014))
perc99 <- vector("double", ncol(data2014))
stdVector <- vector("double", ncol(data2014))
rangeVector <- vector("double", ncol(data2014))
for (i in seq_along(data2014)){
  meanVector[[i]] <- mean(data2014[[i]])
  medianVector[[i]] <- median(data2014[[i]])
  perc1[[i]] <- quantile(data2014[[i]], probs = c(0.01))
  perc99[[i]] <- quantile(data2014[[i]], probs = c(0.99))
  stdVector[[i]] <- sd(data2014[[i]])
  rangeVector[[i]] <- max(data2014[[i]]) - min(data2014[[i]])
}

#correlation matrix using spearman rank correlation rounded
res <- cor(data2014, method = "spearman")
round(res, 2)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(res, method = "color", col = col(200),  
         type = "full", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col = "darkblue", tl.srt = 45, #Text label color and rotation
         diag = TRUE
)

#format data
meanVector = format(meanVector, digits = 3)
medianVector = format(medianVector, digits = 3, nsmall = 2)
perc1 = format(perc1, digits = 3)
perc99 = format(perc99, digits = 3)
stdVector = format(stdVector, digits = 2)
rangeVector = format(rangeVector, digits = 3)

#create a table for the descriptive statistics
test2014 <- data.frame(meanVector, medianVector, perc1, perc99, stdVector, rangeVector)
colnames(test2014) <- c("Mean", "Median", "Percentile 1", "Percentile 99", "Standard Deviation", "Range")
rownames(test2014) <- c("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "NOX")


grid.table(test2014)