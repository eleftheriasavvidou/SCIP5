library(tidyverse)
library(gridExtra)
library(grid)
library(corrplot)
library(dplyr)
library(readr)

#Read all datasets from path and add them in a data frame
df <- list.files(path="C:/Users/eleft/Documents", pattern = "*.csv", full.names = TRUE) %>% 
  lapply(read_csv) %>% 
  bind_rows 
df
df <- df[-10]

#Find mean, median, percentile 1, percentile 99, standard deviation and range for each variable
meanVector <- vector("double", ncol(df))
medianVector <- vector("double", ncol(df))
perc1 <- vector("double", ncol(df))
perc99 <- vector("double", ncol(df))
stdVector <- vector("double", ncol(df))
rangeVector <- vector("double", ncol(df))
for (i in seq_along(df)){
  meanVector[[i]] <- mean(df[[i]])
  medianVector[[i]] <- median(df[[i]])
  perc1[[i]] <- quantile(df[[i]], probs = c(0.01))
  perc99[[i]] <- quantile(df[[i]], probs = c(0.99))
  stdVector[[i]] <- sd(df[[i]])
  rangeVector[[i]] <- max(df[[i]]) - min(df[[i]])
}

#correlation matrix using spearman rank correlation rounded
res <- cor(df, method = "spearman")
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
perc1 = format(perc1, digits = 2, nsmall = 2)
perc99 = format(perc99, digits = 3, nsmall = 2)
stdVector = format(stdVector, digits = 2)
rangeVector = format(rangeVector, digits = 3)

#create a table for the descriptive statistics
testdf <- data.frame(meanVector, medianVector, perc1, perc99, stdVector, rangeVector)
colnames(testdf) <- c("Mean", "Median", "Percentile 1", "Percentile 99", "Standard Deviation", "Range")
rownames(testdf) <- c("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "NOX")


grid.table(testdf)
