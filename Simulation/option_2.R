.libPaths("D:/soft/r/4.2")#this row only for labs
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
#library(readxl)
library(knitr)
library(rmarkdown)
library(simmer)
library(simmer.plot)
library(purrr)


##----------------------------------------- 1.  all functions ------------------------------------------------

addService<- function  (path,sname,attName){
  updatedPath <- seize(path, sname)%>%
    timeout(attName) %>%
    release(sname)
  
  return(updatedPath)
}


addServiceByAttribute<- function  (path,sname,attName){
  updatedPath <- seize(path, sname)%>%
    timeout_from_attribute(attName) %>%
    release(sname)
  
  return(updatedPath)
}
SizeGroup <-  function(){
  SizeGroup <-rdiscrete(1,c(0.33,0.4,0.27),c(3,4,5))
  return (c(SizeGroup))
}
avgQueue2 <- function(time, queueLength, simTime){
  Lavg = 0;
  L = queueLength[1];
  Tnow = time[1];
  Llast = time[1];
  TL = 0;
  Tmax = simTime;
  if (length(time) == length(queueLength)){
    for (i in 2:length(time)){
      if(queueLength[i] != queueLength[i-1]){
        Tnow = time[i];
        TL = TL+L*(Tnow-Llast);
        L = queueLength[i];
        Llast = Tnow;
      }#if
    }#for
  }#end if
  TL=TL+L*(Tmax-Llast);
  Lavg = TL/Tmax;
  return (Lavg);
}

choose_station<-  function(){
  # foodType <- foodType()
  foodType <- food_option()
  foodType[1] <-get_attribute(wedding,"banim")
  foodType[2] <-get_attribute(wedding,"fucacha")
  foodType[3] <-get_attribute(wedding,"vegie")
  foodType[4] <-get_attribute(wedding,"tortia")
  foodType[5] <-get_attribute(wedding,"sushi")
  
  Ok <- FALSE
  while(Ok==FALSE){
    station_num <- rdunif(1,1,5)
    if(station_num==1 & foodType[1]==0){
      return(1)
    }
    if(station_num==2 & foodType[2]==0){
      return(2)
    }
    if(station_num==3 & foodType[3]==0){
      return(3)
    }
    if(station_num==4 & foodType[4]==0){
      return(4)
    }
    if(station_num==5 & foodType[5]==0){
      return(5)
    }
    
    if(foodType[1]==1 & foodType[2]==1 & foodType[3]==1 & foodType[4]==1 & foodType[5]==1 ){
      # ok = TRUE
      break
    }
  }
  return(0)
}


y_value <- function(x){
  if((x>=1)&(x<2)){
    return(3*(x^2)/14 + 2*x/15 + 1/15)
  }
  if((x>=2)&(x<3)){
    return(2/3 - x/5)
  }
  if((x>=3)&(x<5)){
    return(1/6 - x/30)
  }
  else{
    return(0)
  }
}

z_value <- function(x){
  if((x>=1)&(x<=5)){
    return(25/12)
  }
  else{
    return(0)
  }
}

choosing_number <- function(){
  while(TRUE){
    u1 <- runif(1,0,1)
    satiety <- 4*u1+1
    z_function <- y_value(satiety)
    y_function <- z_value(satiety)
    
    u2 <- runif(1,0,1)
    if(u2 <= z_function/y_function){
      return(satiety)
    }
  }
}


#------------------------Function------------------------#

priority <- function(){
  satiety1 <- choosing_number()
  if(satiety1 < 4.5){
    return(c(5,5.1,FALSE))
  }
  return(c(4.5,5,FALSE))
}

singlescapacity <- function(){
  singles <- rdunif(1,100,150)
  return (singles)
}

outdoor_bar_choice <-  function(){
  familysize <- get_attribute(wedding,"familysize")
  if(familysize == 1){
    bar_path <-  rdiscrete (1, c(0.15,0.85),c(0,1))
    return(bar_path)
  }
  if(familysize == 2){
    bar_path <-rdiscrete (1, c(0.35,0.65),c(0,1))
    return(bar_path)
  }
  else
    bar_path <-rdiscrete (1, c(0.55,0.45),c(0,1))
  return(bar_path)
  
}


stations_vector <- function(){
  familysize <- get_attribute(wedding,"familysize")
  if(familysize==1){
    return(c(1,1,0,0,0))
  }
  
  else
    return(c(0,0,0,0,0))
}


CheckIfPassAllStations <-  function(){#Stop conditions for visiting the food stands
  c <- c()
  c[1] <- get_attribute(wedding,"banim")
  c[2] <- get_attribute(wedding,"fucacha")
  c[3] <- get_attribute(wedding,"vegie")
  c[4] <- get_attribute(wedding,"tortia")
  c[5] <- get_attribute(wedding,"sushi")
  
  if(now(wedding)>= canopystartingtime){
    return(FALSE)
  }
  if(c[1]==0 || c[2]==0 || c[3]==0 || c[4]==0 || c[5]==0){
    return(TRUE)
  }
  else
    return(FALSE)
}


food_option <-  function(){
  # x <- c(1,2,3,4,5)
  # menu <-sample(x, size=1, replace = FALSE, prob = NULL)
  menu <- rdunif(1,1,5)
  if(menu==1){
    y <-  rdiscrete (1, c(0.5,0.5),c(0,1))
    if(y==0  )
      return("sushi1")
    else
      return("sushi2")
  }
  if(menu==2){
    return("fucacha")
  }
  
  if(menu==3){
    return("tortia")
  }
  if(menu==4){
    return("banim")
  }
  else
    return("vegie")
}

total_canopy_time <-  function(){
  if(now(wedding)>= canopyfinishtime){
    return(0)
  }
  
  else
    return(canopyfinishtime-now(wedding))
  
}

single_capacity <- function(){
  Num <- rdunif(1,100,150)
  ArrivalTime <-  runif(1,60,75)
  At <- replicate(Num, ArrivalTime)
  
  return(At)
}

##----------------------------------------- 2.  all simulation parameters ------------------------------------------------

simulationTimewedding<-60*6


##----------------------------------------- 3.  Init Simulation and add all resources  ------------------------------------------------

canopystartingtime<-runif(1,140,160) # limited time for canopy
canopyfinishtime<-runif(1,canopystartingtime+20,canopystartingtime+35)
# start to count from the Distribution above

canopy_gate<-schedule(timetable = c(0, canopystartingtime,canopyfinishtime)
                      , values = c(0,Inf,0), period = Inf)
# allow the resource according to Distribution time
outdoorbar_gate<-schedule(timetable = c(0, canopystartingtime-10)
                          , values = c(3, 0), period = Inf)
# allow the resource according to canopy starting time

wedding <- simmer("wedding") %>%
  add_resource("parking", capacity = Inf, queue_size =Inf)%>%
  add_resource("reception1", capacity = 2, queue_size =Inf)%>%
  add_resource("reception2", capacity = 2, queue_size =Inf)%>%
  add_resource("fucacha", capacity = 3, queue_size =Inf)%>%
  add_resource("banim", capacity = 3, queue_size =Inf)%>%
  add_resource("sushi1", capacity = 2, queue_size =Inf)%>%
  add_resource("sushi2", capacity = 2, queue_size =Inf)%>%
  add_resource("vegie", capacity = 4, queue_size =Inf)%>%
  add_resource("tortia", capacity = 4, queue_size =Inf)%>%
  add_resource("indoorbar5", capacity = 5, queue_size =Inf)%>%
  add_resource("indoorbar7", capacity = 7, queue_size =Inf)%>%
  add_resource(name="canopy",capacity=canopy_gate,queue_size=Inf)%>%
  add_resource(name="outdoorbar",capacity=outdoorbar_gate,queue_size=Inf)%>%
  add_resource(name="tablenum",capacity=Inf,queue_size=Inf)%>%
  add_resource("dessert", capacity = 8, queue_size =15,preemptive=F)



##----------------------------------------- 4.  All trajectories, start from main trajectory and add sub-trajectories ABOVE IT it . ------------------------------------------------

home_traj <- trajectory("home_traj")%>% #just the distribution without resource
  set_attribute(keys=c("home"),values=1)

vegie_meal_traj <- trajectory("vegie_meal_traj")%>% #just the distribution without resource
  timeout(function()rnorm(1,7,4))


meat_meal_traj <- trajectory("meat_meal_traj")%>% #just the distribution without resource
  timeout(function()rnorm(1,12,3))

dancefloor_traj<-trajectory("dancefloor_traj") %>%
  # back to the dance floor and try again, 2 times
  timeout(function()rtriangle (1,5,10,7))

dessert_traj<-trajectory("dessert_traj")%>%
  
  timeout(function() runif(1,2.5,4)) %>%
  release("dessert", 1)

indoorbar_traj <- trajectory("indoorbar_traj")%>%
  simmer::select(resource = c("indoorbar5","indoorbar7"),policy = c("shortest-queue-available"),id=2)%>%
  simmer::seize_selected(id=2)%>%
  simmer::timeout(function()rexp(1,3.1104))%>%
  simmer::release_selected(id=2)

hall_traj <- trajectory("hall_traj")%>%  # after canopy everyone will come here
  
  addService("tablenum",function()rexp(1,3.5))%>% # distribution for finding table
  clone(n=function()get_attribute(wedding,"familysize"))%>% # eating alone
  branch(option=function(){rdiscrete (1, c(0.7,0.3),c(1,2))},continue=c(TRUE,TRUE),meat_meal_traj,vegie_meal_traj)%>%
  synchronize(wait=TRUE,mon_all=TRUE)%>%  # waiting for everyone
  set_prioritization(function()priority())%>%
  seize("dessert", 1, continue = c(FALSE,TRUE) ,post.seize=dessert_traj,reject =dancefloor_traj)%>%
  rollback(amount = 1, times = 1)%>% #try seize again 1 time
  seize("dessert", 1, continue = c(FALSE,FALSE) ,post.seize=dessert_traj,reject =dancefloor_traj)


canopy_traj <- trajectory("canopy_traj")%>%
  addService("canopy",function()total_canopy_time())%>%
  synchronize(wait=TRUE,mon_all=TRUE)


outdoorbar_traj <- trajectory("outdoorbar_traj")%>%
  addService("outdoorbar",function() rexp(1,2.666))

sushi_traj <- trajectory("sushi_traj")%>%
  simmer::select(resource = c("sushi1","sushi2"),policy = c("random"),id=4)%>%
  simmer::seize_selected(id=4)%>%
  simmer::timeout(function()rnorm(1,1.5,0.7))%>%
  simmer::release_selected(id=4)%>%
  timeout(function()rexp(1,0.8))%>%
  set_attribute(keys=c("sushi"),values=1)

fucacha_traj <- trajectory("fucacha_traj")%>%
  addService("fucacha",function()rnorm(1,1.5,0.7))%>%
  timeout(function()rexp(1,0.8))%>%
  set_attribute(keys=c("fucacha"),values=1)


tortia_traj <- trajectory("tortia_traj")%>%
  addService("tortia",function()rnorm(1,1.5,0.7))%>%
  timeout(function()rexp(1,0.8))%>%
  set_attribute(keys=c("tortia"),values=1)

banim_traj <- trajectory("banim_traj")%>%
  addService("banim",function()rnorm(1,1.5,0.7))%>%
  timeout(function()rexp(1,0.8))%>%
  set_attribute(keys=c("banim"),values=1)

vegie_traj <- trajectory("vegie_traj")%>%
  addService("vegie",function()rnorm(1,1.5,0.7))%>%
  timeout(function()rexp(1,0.8))%>%
  set_attribute(keys=c("vegie"),values=1)

reception_traj <- trajectory("reception_traj")%>%
  simmer::select(resource = c("reception1","reception2"),policy = c("shortest-queue-available"),id=3)%>%
  simmer::seize_selected(id=3)%>%
  simmer::timeout(function()0.3)%>%
  simmer::release_selected(id=3)%>%
  leave(0.07)%>%
  clone(n= function()get_attribute(wedding,"familysize")) %>%
  set_attribute(keys=c("banim","fucacha","vegie","tortia","sushi"),values=function()stations_vector())%>%
  branch(option=function()choose_station(), continue=c(TRUE,TRUE,TRUE,TRUE,TRUE),banim_traj,fucacha_traj,vegie_traj,tortia_traj,sushi_traj)%>%
  branch(option=function()outdoor_bar_choice(), continue=TRUE,outdoorbar_traj)%>%
  rollback(amount = 2, check = function() CheckIfPassAllStations() )

# this part divided into 3 different trajectories, one for each generator, and this is the main:
# getting table number
# finding parking spot
# select function for choosing reception
# 7% leaving
# vector for food stations
# sample function for choosing the food station
# branch for outdoor bar each time they finish eating

# ------------------------------------------------------------------------------------
couple_traj <- trajectory("couple_traj")%>%
  set_attribute(keys=c("tablenum"),values=function()rdunif(1,1,95))%>%
  set_attribute(keys=c("familysize"),values=2)%>%
  addService("parking",function()rtriangle(1,3,5,4))%>%
  join(reception_traj,canopy_traj,indoorbar_traj,hall_traj,home_traj)


# ------------------------------------------------------------------------------------
single_traj <- trajectory("single_traj")%>%
  set_attribute(keys=c("tablenum"),values=function()rdunif(1,1,95))%>%
  set_attribute(keys=c("familysize"),values=1)%>%
  join(reception_traj,canopy_traj,indoorbar_traj,hall_traj,home_traj)



# ------------------------------------------------------------------------------------

family_traj <- trajectory("family_traj")%>%
  set_attribute(keys="tablenum",values=function()rdunif(1,1,95))%>%
  set_attribute(keys=c("familysize"),values= function() SizeGroup())%>%
  addService("parking",function()rtriangle(1,3,5,4))%>%
  join(reception_traj,canopy_traj,hall_traj,home_traj)





##----------------------------------------- 5. All Generators, ALWAYS LAST. ------------------------------------------------
wedding %>%
  add_generator("single", single_traj,distribution=at(single_capacity()),mon = 2) %>%
  add_generator("couple", couple_traj,to(240,function() rexp(1,0.5469)),mon = 2) %>%
  add_generator("family", family_traj,to(240,function() rexp(1,0.797)),mon = 2)


##----------------------------------------- 6.  reset, run, plots, outputs ------------------------------------------------
set.seed(123)
reset(wedding)%>%
  run(simulationTimewedding)

weddingdata<-get_mon_resources(wedding)
weddingdata2<-get_mon_arrivals(wedding)

# #------------------------------------------------ מדד אורך תור ממוצע בבר החיצוני
# 
# n <- mclapply(1:63, function(i){
#   set.seed(((i+10)^2)*3-7)
#   reset(wedding)%>%run(until=simulationTimewedding)%>%
#     wrap()
# })
# 
# fullData<-get_mon_arrivals(n, ongoing = TRUE) # the full data of all replications
# fullData2<- get_mon_resources(n)
# fullData3<-get_mon_attributes(n)
# arrivalDataPerResource <- get_mon_arrivals(n, ongoing=TRUE,
#                                            per_resource=TRUE)
# resourceData <- get_mon_resources(n)
# avg_boutdorrbar_queue2 <- sqldf(
#   "select replication, avg(queue) as avgQueue2
#  from resourceData
#  where resource = 'outdoorbar'
#  group by replication")
# 
# #----------------------------------------------------------- חישוב כמות אנשים שעברו בכל העמדות
# avg_visits2 <-sqldf("select replication, count(*)/7 as visits2
# from arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
# or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
# or resource='tortia'
# group by replication    " ) %>% print
# mean_avg_visits<- mean(avg_visits2$visits2)
# sd_avg_visits<- sd(avg_visits2$visits2)
# 
# #---------------------------------------------------------------
# utility_total2<-sqldf("select replication, sum(activity_time)/(24*4*60)
# as utility2 from (select * from
# arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
# or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
# or resource='tortia' or resource='reception1' or resource='reception2')
# where activity_time<>'NA'group by replication") %>% print





