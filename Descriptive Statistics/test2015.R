library(tidyverse)
library(gridExtra)
library(grid)
library(corrplot)

#read 2015 data and delete CO
data2015 <- read.csv("gt_2015.csv")
data2015 <- data2015[-10]

#Find mean, median, percentile 1, percentile 99, standard deviation and range for each variable
meanVector <- vector("double", ncol(data2015))
medianVector <- vector("double", ncol(data2015))
perc1 <- vector("double", ncol(data2015))
perc99 <- vector("double", ncol(data2015))
stdVector <- vector("double", ncol(data2015))
rangeVector <- vector("double", ncol(data2015))
for (i in seq_along(data2015)){
  meanVector[[i]] <- mean(data2015[[i]])
  medianVector[[i]] <- median(data2015[[i]])
  perc1[[i]] <- quantile(data2015[[i]], probs = c(0.01))
  perc99[[i]] <- quantile(data2015[[i]], probs = c(0.99))
  stdVector[[i]] <- sd(data2015[[i]])
  rangeVector[[i]] <- max(data2015[[i]]) - min(data2015[[i]])
}

#correlation matrix using spearman rank correlation rounded
res <- cor(data2015, method = "spearman")
round(res, 2)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(res, method = "color", col = col(200),  
         type = "full", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col = "darkblue", tl.srt = 45, #Text label color and rotation
         diag = TRUE
)

#format data
meanVector = format(meanVector, digits = 3, nsmall = 2)
medianVector = format(medianVector, digits = 3, nsmall = 2)
perc1 = format(perc1, digits = 2)
perc99 = format(perc99, digits = 3, nsmall = 2)
stdVector = format(stdVector, digits = 2)
rangeVector = format(rangeVector, digits = 3)

#create a table for the descriptive statistics
test2015 <- data.frame(meanVector, medianVector, perc1, perc99, stdVector, rangeVector)
colnames(test2015) <- c("Mean", "Median", "Percentile 1", "Percentile 99", "Standard Deviation", "Range")
rownames(test2015) <- c("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "NOX")


grid.table(test2015)