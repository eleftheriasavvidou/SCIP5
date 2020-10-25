#install.packages("tidyverse")
#install.packages("gridExtra")
#install.packages("corrplot")

#Load needed packages
library(tidyverse)
library(gridExtra)
library(grid)
library(corrplot)

#read 2011 data and delete CO
data2011 <- read.csv("gt_2011.csv")
data2011 <- data2011[-10]

#Find mean, median, percentile 1, percentile 99, standard deviation and range for each variable
meanVector <- vector("double", ncol(data2011))
medianVector <- vector("double", ncol(data2011))
perc1 <- vector("double", ncol(data2011))
perc99 <- vector("double", ncol(data2011))
stdVector <- vector("double", ncol(data2011))
rangeVector <- vector("double", ncol(data2011))
for (i in seq_along(data2011)){
  meanVector[[i]] <- mean(data2011[[i]])
  medianVector[[i]] <- median(data2011[[i]])
  perc1[[i]] <- quantile(data2011[[i]], probs = c(0.01))
  perc99[[i]] <- quantile(data2011[[i]], probs = c(0.99))
  stdVector[[i]] <- sd(data2011[[i]])
  rangeVector[[i]] <- max(data2011[[i]]) - min(data2011[[i]])
}

#correlation matrix using spearman rank correlation rounded
res <- cor(data2011, method = "spearman")
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
medianVector = format(medianVector, digits = 3)
perc1 = format(perc1, digits = 3)
perc99 = format(perc99, digits = 3)
stdVector = format(stdVector, digits = 2)
rangeVector = format(rangeVector, digits = 3)

#create a table for the descriptive statistics
test2011 <- data.frame(meanVector, medianVector, perc1, perc99, stdVector, rangeVector)
colnames(test2011) <- c("Mean", "Median", "Percentile 1", "Percentile 99", "Standard Deviation", "Range")
rownames(test2011) <- c("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "NOX")


grid.table(test2011)
