#packages
install.packages("ggpubr")
install.packages("dplyr")
remove.packages("vctrs")
install.packages("vctrs", dependencies = TRUE)

#loadLibrarys
library(rlang)
library(MASS)
library(fitdistrplus)
library(magrittr)
library(dplyr)
library(lazyeval)
library(parallel)
library(e1071)
library(ggplot2)
library(plotly)
library(triangle)
library(sqldf)
#library(readxl)
#library(knitr)
#library(rmarkdown)
library(simmer)
library(simmer.plot)
library(ggpubr)
library(dplyr)
library(lmtest)
library(strucchange)

##PART B

#get Data
dataset<-read.csv(file.choose(),header = T)

# change data to numeric
numericdata <- dataset
for(i in 1:length(dataset$Air.pollution.index)){
  if(dataset$Air.pollution.index[i]=="Minor pollution"){
    numericdata$Air.pollution.index[i] <- 0
  }
  else if(dataset$Air.pollution.index[i]=="Low pollution"){
    numericdata$Air.pollution.index[i] <- 1
  }
  else if(dataset$Air.pollution.index[i]=="Medium pollution"){
    numericdata$Air.pollution.index[i] <- 2
  }
  else  if(dataset$Air.pollution.index[i]=="High pollution"){
    numericdata$Air.pollution.index[i] <- 3
  }
  else if (dataset$Air.pollution.index[i]=="Extreme pollution")
    numericdata$Air.pollution.index[i]<-4
}


numericdata$Continent[numericdata$Continent=="Asia"]<-4
numericdata$Continent[numericdata$Continent=="Africa"]<-5
numericdata$Continent[numericdata$Continent=="North America"]<-2
numericdata$Continent[numericdata$Continent=="South America"]<-1
numericdata$Continent[numericdata$Continent=="Europe"]<-0
numericdata$Continent[numericdata$Continent=="Australia"]<-3
numericdatasorted <- numericdata %>% arrange("Depression rate (%)")




#2.1 remove unnececary parameters from the model--------------------------------------------------------------------
cor(numericdatasorted$Obesity.rate, numericdatasorted$Cancer.mortality.rate.... , method = c("pearson"))
cor(numericdatasorted$Smoke.rate, numericdatasorted$Cancer.mortality.rate.... , method = c("pearson"))
cor(numericdatasorted$State.Development.Index..0.1., numericdatasorted$Cancer.mortality.rate.... , method = c("pearson"))
cor(numericdatasorted$Median.age, numericdatasorted$Cancer.mortality.rate.... , method = c("pearson"))
cor(numericdatasorted$Average.temperature, numericdatasorted$Cancer.mortality.rate.... , method = c("pearson"))
cor(numericdatasorted$Percentage.of.residents.in.cities, numericdatasorted$Cancer.mortality.rate.... , method = c("pearson"))
cor(numericdatasorted$Depression.rate...., numericdatasorted$Cancer.mortality.rate.... , method = c("pearson"))

#preivios plots before change------------------------------------------
plot(numericdatasorted$State.Development.Index..0.1., numericdatasorted$Cancer.mortality.rate...., main="S.Development Index",
     xlab="State Development Index ", ylab="cancer mortality rate ", pch=19,col = "navy")

plot(numericdatasorted$Air.pollution.index, numericdatasorted$Cancer.mortality.rate...., main="Scatterplot AP",
     xlab="AP ", ylab="cancer mortality rate ", pch=19,col = "navy")


#2.2 - changing the DB

#Air pollution index

# state development variable----------------------------------------------------------
numericdatasorted$sd_new[numericdatasorted$State.Development.Index..0.1.>=0.8] <- 0
numericdatasorted$sd_new[and(numericdatasorted$State.Development.Index..0.1.>=0.700,numericdatasorted$State.Development.Index..0.1.<=0.799)] <- 1
numericdatasorted$sd_new[and(numericdatasorted$State.Development.Index..0.1.>=0.550,numericdatasorted$State.Development.Index..0.1.<=0.699)] <- 2
numericdatasorted$sd_new[numericdatasorted$State.Development.Index..0.1.<=0.549] <- 3
plot(numericdatasorted$sd_new, numericdatasorted$Cancer.mortality.rate...., main="New S. Development",
     xlab="State Development Index ", ylab="cancer mortality rate ", pch=19,col = "navy")

numericdatasorted$AP_new[numericdatasorted$Air.pollution.index == 0] <- 0
numericdatasorted$AP_new[numericdatasorted$Air.pollution.index == 1] <- 1
numericdatasorted$AP_new[numericdatasorted$Air.pollution.index == 2] <- 2
numericdatasorted$AP_new[numericdatasorted$Air.pollution.index == 3] <- 3
numericdatasorted$AP_new[numericdatasorted$Air.pollution.index == 4] <- 3
plot(numericdatasorted$AP_new, numericdatasorted$Cancer.mortality.rate...., main="Scatterplot AP New",
     xlab="AP_new ", ylab="cancer mortality rate ", pch=19,col = "navy")


# 2.3 - Create factors for the model---------------------------------------------------------------------
#create factor for new X4
x4Factor <- factor(numericdatasorted$sd_new)
#create factor for new X3
x3Factor <- factor(numericdatasorted$AP_new)
#create factor for new X9
x9Factor <- factor(numericdatasorted$Continent)


#2.4--------------------------------------------------------------------------------------------

# X1*X4factor
coloring<-ifelse(numericdatasorted$sd_new==0 , c("Purple"), ifelse(numericdatasorted$sd_new==1 , c("Green"), ifelse(numericdatasorted$sd_new==2 ,c("Navy"), c("orange"))))
plot(numericdatasorted$Obesity.rate, numericdatasorted$Cancer.mortality.rate...., xlab="Obesity Rate", ylab='Cancer mortality rate', col=coloring )
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Purple"),col="Purple")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Green"),col="Green")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Navy"),col="Navy")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="orange"),col="orange")


# X6*X9factor
coloring<-ifelse(numericdatasorted$Continent==0 , c("Purple"), ifelse(numericdatasorted$Continent==1 , c("Green"), ifelse(numericdatasorted$Continent==2 ,c("Navy"), ifelse(numericdatasorted$Continent==3,c("Maroon"),ifelse(numericdatasorted$Continent==4,c("Gold"), c("Magenta"))))))
plot(numericdatasorted$Average.temperature, numericdatasorted$Cancer.mortality.rate...., xlab="Average Temperature", ylab='Cancer mortality rate', col=coloring )
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Purple"),col="Purple")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Green"),col="Green")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Navy"),col="Navy")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Maroon"),col="Maroon")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Gold"),col="Gold")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Obesity.rate,data=numericdatasorted,subset=coloring=="Magenta"),col="Magenta")

# X5*X4factor
coloring<-ifelse(numericdatasorted$sd_new==0 , c("Purple"), ifelse(numericdatasorted$sd_new==1 , c("Green"), ifelse(numericdatasorted$sd_new==2 ,c("Navy"), c("orange"))))
plot(numericdatasorted$Median.age, numericdatasorted$Cancer.mortality.rate...., xlab="Median Age", ylab='Cancer mortality rate', col=coloring )
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Median.age,data=numericdatasorted,subset=coloring=="Purple"),col="Purple")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Median.age,data=numericdatasorted,subset=coloring=="Green"),col="Green")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Median.age,data=numericdatasorted,subset=coloring=="Navy"),col="Navy")
abline(lm(numericdatasorted$Cancer.mortality.rate....~numericdatasorted$Median.age,data=numericdatasorted,subset=coloring=="orange"),col="orange")

#3.1-------------------------------------------------------------------------------------------
y<-numericdatasorted$Cancer.mortality.rate....
x1<-numericdatasorted$Obesity.rate
x2<-numericdatasorted$Smoke.rate
x3<-numericdatasorted$AP_new
x4<-numericdatasorted$sd_new
x5<-numericdatasorted$Median.age
x6<-numericdatasorted$Average.temperature
x9<-numericdatasorted$Continent

p<- lm(formula = y~ x1 + x2 + x5+ x6 + x3Factor + x4Factor +x9Factor+
         x4Factor*x1+x9Factor*x6)
summary(p)
AIC<-extractAIC(p)
BIC<-extractAIC(p,k=log(117))
print(AIC)
print(BIC)


#AIC:

backwardAIC <- step(p, direction = "backward")
summary(backwardAIC)
AIC<-extractAIC(backwardAIC)
BIC<-extractAIC(backwardAIC,k=log(117))
print(AIC)
print(BIC)

nullModel <- lm(numericdatasorted$Cancer.mortality.rate....~1,data = numericdatasorted)
forwardAIC <- step(nullModel, direction = "forward", scope = formula(p))
summary(forwardAIC)
AIC<-extractAIC(forwardAIC)
BIC<-extractAIC(forwardAIC,k=log(117))
print(AIC)
print(BIC)

twoSidedAIC <- step(nullModel, direction = "both", scope = formula(p))
summary(twoSidedAIC)
AIC<-extractAIC(twoSidedAIC)
BIC<-extractAIC(twoSidedAIC,k=log(117))
print(AIC)
print(BIC)

#BIC:

backwardBIC <- step(p, direction = "backward", k = log(117))
summary(backwardBIC)
AIC<-extractAIC(backwardBIC)
BIC<-extractAIC(backwardBIC,k=log(117))
print(AIC)
print(BIC)

nullModel <- lm(numericdatasorted$Cancer.mortality.rate....~1, data = numericdatasorted)
forwardBIC <- step(nullModel,direction = "forward", scope = formula(p) , k = log(117))
summary(forwardBIC)
AIC<-extractAIC(forwardBIC)
BIC<-extractAIC(forwardBIC,k=log(117))
print(AIC)
print(BIC)

twoSidedBIC <- step(nullModel,direction = "both", scope = formula(p) , k = log(117))
summary(twoSidedBIC)
AIC<-extractAIC(twoSidedBIC)
BIC<-extractAIC(twoSidedBIC,k=log(117))
print(AIC)
print(BIC)

#the final regression model:-----------------------------------------------------------------
finalModel <-lm(formula = y~x5+ x6 + x3Factor + x4Factor +x9Factor+
                  +x9Factor*x6)


#3.2 Test the Assumptions of Regression:------------------------------------------------------
#Variance equality check
numericdatasorted$fitted <- fitted(finalModel)
numericdatasorted$residuals <- residuals(finalModel)
se <- sqrt(var(numericdatasorted$residuals))
numericdatasorted$stan_residuals <- (residuals(finalModel)/se)
plot(numericdatasorted$stan_residuals~ numericdatasorted$fitted, main="Residual Plot", xlab= "Fitted value" ,ylab= "Stan residuals")%>% abline(a=0, b=0, col="Red")


#f test for variance equality
gqtest(finalModel, alternative = "two.sided")

#Normal test
qqnorm(numericdatasorted$stan_residuals)
abline(a=0, b=1, col="Navy")
hist(numericdatasorted$stan_residuals,prob=TRUE, xlab ="Normalized error", main="Histogram of normalized error")
lines(density(numericdatasorted$stan_residuals),col="Navy",lwd=2)

#tests for normalityCheck
ks.test(x=numericdatasorted$stan_residuals,y="pnorm",alternative = "two.sided",exact=NULL)
shapiro.test(numericdatasorted$stan_residuals)

#linearty test
sctest(finalModel,type = "Chow")

#4---------------------------------------------------------------------------------------------
Sqrtp<- lm(formula = sqrt(y)~ (x1) + (x2) + (x5)+ (x6) + x3Factor + x4Factor +x9Factor+
             x4Factor*x1+x9Factor*x6)
summary(Sqrtp)
pSquared<- lm(formula = (y^2)~ (x1) + (x2) + (x5)+ (x6) + x3Factor + x4Factor +x9Factor+
                x4Factor*x1+x9Factor*x6)
summary(pSquared)
lnp<- lm(formula = log(y)~ (x1) + (x2) + (x5)+ (x6) + x3Factor + x4Factor +x9Factor+
           x4Factor*x1+x9Factor*x6)
summary(lnp)
AIC<-extractAIC(lnp)
BIC<-extractAIC(lnp,k=log(117))
print(AIC)
print(BIC)

#AIC:

backwardAIC <- step(lnp, direction = "backward")
summary(backwardAIC)
AIC<-extractAIC(backwardAIC)
BIC<-extractAIC(backwardAIC,k=log(117))
print(AIC)
print(BIC)

nullModel <- lm(numericdatasorted$Cancer.mortality.rate....~1,data = numericdatasorted)
forwardAIC <- step(nullModel, direction = "forward", scope = formula(lnp))
summary(forwardAIC)
AIC<-extractAIC(forwardAIC)
BIC<-extractAIC(forwardAIC,k=log(117))
print(AIC)
print(BIC)

twoSidedAIC <- step(nullModel, direction = "both", scope = formula(lnp))
summary(twoSidedAIC)
AIC<-extractAIC(twoSidedAIC)
BIC<-extractAIC(twoSidedAIC,k=log(117))
print(AIC)
print(BIC)

#BIC:

backwardBIC <- step(lnp, direction = "backward", k = log(117))
summary(backwardBIC)
AIC<-extractAIC(backwardBIC)
BIC<-extractAIC(backwardBIC,k=log(117))
print(AIC)
print(BIC)

nullModel <- lm(numericdatasorted$Cancer.mortality.rate....~1, data = numericdatasorted)
forwardBIC <- step(nullModel,direction = "forward", scope = formula(lnp) , k = log(117))
summary(forwardBIC)
AIC<-extractAIC(forwardBIC)
BIC<-extractAIC(forwardBIC,k=log(117))
print(AIC)
print(BIC)

twoSidedBIC <- step(nullModel,direction = "both", scope = formula(lnp) , k = log(117))
summary(twoSidedBIC)
AIC<-extractAIC(twoSidedBIC)
BIC<-extractAIC(twoSidedBIC,k=log(117))
print(AIC)
print(BIC)

#Anova
lnpUpdated <- lm(formula = log(y)~ (x5)+ (x6) + x3Factor + x4Factor +x9Factor+
                   x9Factor*x6)
newlnp <- lm(formula = log(y)~ (x1) +(x5)+ (x6) + x3Factor + x4Factor +x9Factor+
               x9Factor*x6)
anova(lnpUpdated,newlnp,test="F")