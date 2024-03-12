n <- mclapply(1:63, function(i){
  set.seed(((i+10)^2)*3-7)
  reset(wedding)%>%run(until=simulationTimewedding)%>%
    wrap()
})

fullData<-get_mon_arrivals(n, ongoing = TRUE) # the full data of all replications
fullData2<- get_mon_resources(n)
fullData3<-get_mon_attributes(n)
arrivalDataPerResource <- get_mon_arrivals(n, ongoing=TRUE,
                                           per_resource=TRUE)
resourceData <- get_mon_resources(n)
avg_boutdorrbar_queue <- sqldf(
  "select replication, avg(queue) as avgQueue
 from resourceData
 where resource = 'outdoorbar'
 group by replication")

#----------------------------------------------------------- חישוב כמות אנשים שעברו בכל העמדות
avg_visits <-sqldf("select replication, count(*)/7 as visits
from arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
or resource='tortia'
group by replication    " ) %>% print
mean_avg_visits<- mean(avg_visits$visits)
sd_avg_visits<- sd(avg_visits$visits)

#---------------------------------------------------------------
utility_total<-sqldf("select replication, sum(activity_time)/(24*4*60)
as utility from (select * from
arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
or resource='tortia' or resource='reception1' or resource='reception2')
where activity_time<>'NA'group by replication") %>% print

fullData<-get_mon_arrivals(n, ongoing = TRUE) # the full data of all replications
fullData2<- get_mon_resources(n)
fullData3<-get_mon_attributes(n)
arrivalDataPerResource <- get_mon_arrivals(n, ongoing=TRUE,
                                           per_resource=TRUE)
resourceData <- get_mon_resources(n)
avg_boutdorrbar_queue1 <- sqldf(
  "select replication, avg(queue) as avgQueue1
 from resourceData
 where resource = 'outdoorbar'
 group by replication")

#----------------------------------------------------------- חישוב כמות אנשים שעברו בכל העמדות
avg_visits1 <-sqldf("select replication, count(*)/8 as visits1
from arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
or resource='tortia1'or resource='tortia1'
group by replication    " ) %>% print
mean_avg_visits1<- mean(avg_visits1$visits1)
sd_avg_visits1<- sd(avg_visits1$visits1)

#---------------------------------------------------------------
utility_total1<-sqldf("select replication, sum(activity_time)/(28*4*60)
as utility1 from (select * from
arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
or resource='tortia1'or resource='tortia2' or resource='reception1' or resource='reception2')
where activity_time<>'NA'group by replication") %>% print

fullData<-get_mon_arrivals(n, ongoing = TRUE) # the full data of all replications
fullData2<- get_mon_resources(n)
fullData3<-get_mon_attributes(n)
arrivalDataPerResource <- get_mon_arrivals(n, ongoing=TRUE,
                                           per_resource=TRUE)
resourceData <- get_mon_resources(n)
avg_boutdorrbar_queue2 <- sqldf(
  "select replication, avg(queue) as avgQueue2
 from resourceData
 where resource = 'outdoorbar'
 group by replication")

#----------------------------------------------------------- חישוב כמות אנשים שעברו בכל העמדות
avg_visit2s <-sqldf("select replication, count(*)/7 as visits2
from arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
or resource='tortia'
group by replication    " ) %>% print
mean_avg_visits2<- mean(avg_visits2$visits2)
sd_avg_visits2<- sd(avg_visits2$visits2)

#---------------------------------------------------------------
utility_total2<-sqldf("select replication, sum(activity_time)/(24*4*60)
as utility2 from (select * from
arrivalDataPerResource where resource='outdoorbar' or resource='fucacha'
or resource='banim' or resource='sushi1' or resource='sushi2' or resource='vegie'
or resource='tortia' or resource='reception1' or resource='reception2')
where activity_time<>'NA'group by replication") %>% print


mean_utility_total<- mean(utility_total$utility)%>% print
sd_utility_total<- sd(utility_total$utility)%>% print
print (mean_avg_visits)
print (sd_avg_visits)
mean_outdoorbar_queue<- mean(avg_boutdorrbar_queue$avgQueue)%>% print
sd_outdoorbar_queue<- sd(avg_boutdorrbar_queue$avgQueue)%>% print

mean_utility_total1<- mean(utility_total1$utility1)%>% print
sd_utility_total1<- sd(utility_total1$utility1)%>% print
print (mean_avg_visits1)
print (sd_avg_visits1)
mean_outdoorbar_queue1<- mean(avg_boutdorrbar_queue1$avgQueue1)%>% print
sd_outdoorbar_queue1<- sd(avg_boutdorrbar_queue1$avgQueue1)%>% print

mean_utility_total2<- mean(utility_total2$utility2)%>% print
sd_utility_total2<- sd(utility_total2$utility2)%>% print
print (mean_avg_visits2)
print (sd_avg_visits2)
mean_outdoorbar_queue2<- mean(avg_boutdorrbar_queue2$avgQueue2)%>% print
sd_outdoorbar_queue2<- sd(avg_boutdorrbar_queue2$avgQueue2)%>% print

# -------------------------------------
pairdTest1<-
  t.test(x=utility_total$utility,y=utility_total1$utility1,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest1)

pairdTest2<-
  t.test(x=utility_total$utility,y=utility_total2$utility2,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest2)

pairdTest3<-
  t.test(x=utility_total1$utility1,y=utility_total2$utility2,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest3)
# -------------------------------------
pairdTest4<-
  t.test(x=avg_boutdorrbar_queue$avgQueue,y=avg_boutdorrbar_queue1$avgQueue1,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest4)

pairdTest5<-
  t.test(x=avg_boutdorrbar_queue$avgQueue,y=avg_boutdorrbar_queue2$avgQueue2,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest5)

pairdTest6<-
  t.test(x=avg_boutdorrbar_queue1$avgQueue1,y=avg_boutdorrbar_queue2$avgQueue2,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest6)

# -------------------------------------

pairdTest7<-
  t.test(x=avg_visits$visits,y=avg_visits1$visits1,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest7)

pairdTest8<-
  t.test(x=avg_visits1$visits1,y=avg_visits2$visits2,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest8)

pairdTest9<-
  t.test(x=avg_visits$visits,y=avg_visits2$visits2,
         alternative="two.sided",paired=TRUE,var.equal=TRUE,conf.level=0.99)
print(pairdTest9)







