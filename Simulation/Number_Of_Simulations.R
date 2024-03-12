n0 <- 20
gamma <- 0.06
alpha_total <- 0.09
alfa_i <- alpha_total/3
t <- qt(1-(alfa_i)/2,n0-1)
gamma_tag <- (gamma/(1+gamma)) %>% print
accuracy <- function(mean,sd){
  (t*sd/sert(n))/mean %>% print
}

test<- t.test(x= utility_total$utility,y=NULL, alternative="two.sided",conf.level=0.97)
print(test)
ci<-c(test$conf.int[1],test$conf.int[2])%>%print
meanFlow <- test$estimate%>%print
delta1<-test$conf.int[2]-test$estimate
diuk1<-delta1/test$estimate
print(diuk1)

test2<- t.test(x=avg_boutdorrbar_queue$avgQueue, y=NULL, alternative="two.sided",conf.level=0.97)
print(test2)
ci2<-c(test2$conf.int[1],test2$conf.int[2])%>%print
delta2<-test2$conf.int[2]-test2$estimate
diuk2<-delta2/test2$estimate
print(diuk2)

test3<- t.test(x=avg_visits$visits, y=NULL,
               alternative="two.sided",conf.level=0.97)
print(test3)
ci3<-c(test3$conf.int[1],test3$conf.int[2])%>%print
delta3<-test3$conf.int[2]-test3$estimate
diuk3<-delta3/test3$estimate
print(diuk3)

n1<-63*(delta1/(mean_utility_total*(gamma/(1+gamma))))^2
print(n1)
n2<-63*(delta2/(mean_outdoorbar_queue*(gamma/(1+gamma))))^2
print(n2)
n3<-63*(delta3/(mean_avg_visits*(gamma/(1+gamma))))^2
print(n3)