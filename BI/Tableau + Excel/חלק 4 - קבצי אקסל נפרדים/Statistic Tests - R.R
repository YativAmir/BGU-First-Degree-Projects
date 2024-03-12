library(rlang)
library(MASS)
library(fitdistrplus)
library(magrittr)
library(dplyr)
library(lazyeval)
library(parallel)
library(e1071)
library(plotly)
library(ggplot2)
library(triangle)
library(sqldf)
library(simmer)
library(simmer.plot)
library(corrplot)
library(gridExtra)
library(strucchange)
library(RColorBrewer)
library(readxl)
library(RColorBrewer)


## set Variables for t test
Dataset <- read.csv(file.choose(),header = T)
Dataset$amountRe[is.na(Dataset$amountRe)] <- 0
Dataset$amountFS[is.na(Dataset$amountFS)] <- 0

# the column names are "AmountFS" and "AmountRe"
group1 <- Dataset$amountFS
group2 <- Dataset$amountRe

# Perform t-test
t_test_result <- t.test(group1, group2)

# Print the result
print(t_test_result)



#ANOVA test
Dataset <- read.csv(file.choose(),header = T)
anova(lm(Dataset$Amount~ Dataset$COUNTRY))
Dataset$amount[is.na(Dataset$amount)] <- 0

#defining colors
num_levels <- length(unique(Dataset$COUNTRY))
palette_size <- 9  
#orange <- rep(brewer.pal(9, "orange"), length.out = palette_size)
orange_palette <- brewer.pal(palette_size, "Oranges")
custom_palette <- colorRampPalette(c("orange", "darkorange"))(palette_size)

boxplot(Dataset$actual_cost ~ Dataset$Order.Month, 
        xlab = "Country", 
        ylab = "Amount",
        main = "Amount by country",
        col = orange
)
print (boxplot.matrix())

###

#Chi
# Read the CSV file
Dataset <- read.csv(file.choose(), header = TRUE)

# Convert Boolean column to factor with levels 0 and 1
Dataset$Boolean <- factor(Dataset$Boolean, levels = c(0, 1))

# Ensure that Amount has at least two levels (it should be categorical)
Dataset$Amount <- factor(Dataset$Amount)

# Perform chi-squared test
chi_squared_result <- chisq.test(Dataset$Amount, Dataset$Boolean, correct = FALSE)
print(chi_squared_result)

# Plot Amount by Boolean
plot(Dataset$Amount ~ Dataset$Boolean, pch = 1, 
     main = "Amount by Shipping Company", 
     xlab = "is Fedex", 
     ylab = "Amount", 
     col = c("light orange", "dark orange"))

#chi test
Dataset <- read.csv(file.choose(),header = T)

Dataset$Boolean <- factor(Dataset$Boolean, levels = c(0, 1))

chisq.test( Dataset$Amount, Dataset$Boolean, correct = FALSE)

plot(Dataset$Amount ~ Dataset$Boolean , pch=1, main = "amount by shipping company", xlab = "is Fedex", ylab = "Amount", col = c("light orange", "dark orange") )





