
--äåñôú èáìú çéôåù ìáñéñ äðúåðéí
Create Table Searches(
IP	        Varchar(15)   not null,
DT	        DateTime      not null, 
ItemID		  int         null,
Email 	    varchar(30),   
primary key(IP,DT),
constraint	FK_ItemID_Se
foreign key(ItemID)
references Furnitures(ItemID))

--úé÷åï éùåú ääæîðåú
create	table	Orders (
OrderID	          int	      not null,
OrderDate	      date        not null,
CardNumber      varchar(16)   not null,
Country	        Varchar(20)   not null,
Street	        varchar(20)   not null,
Number	          integer     not null,
ZipCode         varchar(20)	  not null,
ShippingType    varchar(20)   not null,
price             money       not null,

primary key(OrderID),
constraint	FK_CreditCard_Od
foreign key(CardNumber)
references CreditCards(CardNumber))

alter	table	Orders
add	constraint	Ck_price_Or
check	(price >= 0) -- (=) coz refunds




--äçáøä øåöä ìä÷èéï àú îáçø äøäéèéí áéùøàì åìëï äéà áåã÷ú îä äùìåùä äøäéèéí ùäëðéñå äëé îòè
select top 3 f.itemID, 'net value' = sum (m.price*c.quantity)
from furnitures as f join modifications as m on f.itemId=m.itemId
join [Contains] as c on f.itemID= C.itemId
join Orders as o on c.OrderID = o.OrderID
where o.country= 'israel'
group by f.itemID 
having count (*) < 20
order by 'net value' 

--äîãéðåú òí äëé äøáä àéñåôéí ëãé ìãòú àéôä àôùø ìöîöí àåìé àú äîòøê äìåâéñèé
select top 3 o.country, pickups = count (*)
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber
where o.shippingtype= 'pick up' 
group by o.country
ORDER BY pickups DESC


-- äùååàú áéï ùðé ÷åì÷ùðéí ëãé ìäñé÷ ðúåðé îëéøåú åäöìçéä áîãéðåú äùåðåú
 select b.country , bergamo , fermo, gap = bergamo - fermo
 from (
	select o.Country,fermo = sum(c.quantity)  
	from furnitures as f
join [Contains] as c on f.itemID= C.itemId
join Orders as o on c.OrderID = o.OrderID
where f.collection= 'fermo'
group by o.Country
)as a  join(
select o.Country, count(*) as bergamo
	from furnitures as f 
join [Contains] as c on f.itemID= C.itemId
join Orders as o on c.OrderID = o.OrderID
where f.collection= 'bergamo'
group by o.Country
)as b on a.Country=b.Country


--- îçæéø àú äîãéðåú ùîëøå ìîñôø ì÷åçåú ùåðéí îòì äîîåöò äòåìîé áùðú 2020 
select o.Country , count(distinct c.Email) as total
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber 
where year(o.OrderDate) = 2020
group by o.Country 
having (select av=AVG(a.total) 
from(
select o.Country , count(distinct c.Email) as total
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber 
where year(o.OrderDate) = 2020
group by o.Country 
)as a) < count(distinct c.Email)


-- äçáøä äçìéèä ìúú 10% îä÷ðéåú òã ëä åâí ìòúéã ìèåáú äæîðåú çãùåú
-- ALTER TABLE Customers DROP COLUMN Fcoins

alter table Customers add FCoins money 

update Customers set Fcoins = 
(select sum(o.price)*0.1 
 from orders as o join CreditCards as cc on o.CardNumber = cc.CardNumber
 where Customers.Email = cc.Email 
 group by cc.Email     
 )

 select *
 from Customers
 order by fcoins desc


 -- îëì äì÷åçåú ùùîøå ìôçåú ùðé ôøèéí áîåòãôéí àê òãéï ìà áéöòå äæîðä ëãé ìùìåç ÷åôåï
select *
from Customers as c 
where (select count(cu.Email)
       from Customers as cu join Favorites as f on cu.Email = f.Email
	   where cu.Email = c.Email
	   ) > 2 
except 
select c.Email, c.firstName, c.lastName , c.phone
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber
group by c.Email, c.firstName, c.lastName , c.phone


-- äçáøä øåöä ìäñúéø àú äëúåáú äîãåé÷ú ùì äì÷åçåú åâí àú äðúåðéí äîìàéí ùì ëøèéñ äàùøàé ìèåáú àáèçú îéãò
create view view_CustomerOrder as
select c.Email, c.firstName, c.lastName , c.phone,
       o.OrderID, o.OrderDate, o.CardNumber, o.Country , o.ShippingType,o.price
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber


select *
from view_CustomerOrder


-- áåã÷ ëîä ëì ì÷åç áæáæ áàúø ùìðå ìàåøê çéé äì÷åç
create function ordersCosts (@email varchar (30))
returns money
as begin
	declare @ordersCosts money
	select @ordersCosts= (sum(o.price))
	from Customers as c join creditcards as cc on c.email= cc.email
         join orders as o on cc.cardnumber= o.cardnumber 
	where c.Email = @email

	group by c.Email
	return @ordersCosts	
end

select distinct c.Email , [orders cost]= dbo.ordersCosts(c.Email)
from Customers as c join creditcards as cc on c.email= cc.email
	 join orders as o on cc.cardnumber= o.cardnumber 
order by [orders cost] desc


--áåã÷ ëîä îëéøåú äúáöòå áéï úàøéëéí ñôöéôéí áîãéðä îñåééîú ìôé ñåâ îùìåç
-- drop function function_CountrySellesByShippingType
create function function_CountrySellesByShippingType ( @Country varchar(20), @from date, @until date, @ShipingType varchar(20))
returns table as return
select c.ItemID ,count(distinct o.OrderID) as total_orders, sum(c.quantity) as total_amount
from Orders as o join [Contains] as c on o.OrderID = c.OrderID
where (o.OrderDate between @from and @until) and o.Country = @Country and o.ShippingType = @ShipingType
group by c.ItemID

select *
from dbo.function_CountrySellesByShippingType('denemrk','2021-12-12','2022-12-12','Pick Up')


--îçé÷ú øäéè ììà öåøê ìäúçùá áèáìàåú ùäåà îôúç æø áäí îåãéôé÷ùééï àæ ëùéù ôòåìú îçé÷ä äåà ÷åãí îåç÷ áèáìú îôúç äæø åàæ áèáìä äðçåöä
-- drop trigger Update_modifications
CREATE TRIGGER Update_modifications
ON furnitures instead of delete
as begin 
delete Modifications
where ItemID in (select d.ItemID from deleted as d) and ItemID in (select m.ItemID from Modifications as m)
delete Furnitures 
where ItemID in (select d.ItemID from deleted as d) and ItemID in (select f.ItemID from Furnitures as f)
end
  
select *
from Modifications as m
where m.ItemID = 108

delete from Furnitures where Furnitures.ItemID = 108


--äçáøä øåöä ìãòú ëîä øäéèéí ðîëøå ìôðé çåãùéí ëãé ìàôùø çéæåé äñô÷ä ìôé äöøëéí ùì ëì îãéðä
create procedure sp_furniture_supply (@country varchar (20))
	as begin 
			select c.itemid, total_amount=  sum(c.quantity)
			from orders as o join [Contains] as c on o.orderid= c.orderid
			where year (o.orderdate)= year (getdate()) and month(o.orderdate)= month(getdate())-2
			and o.country = @country
			group by c.itemid,c.quantity
			end

execute sp_furniture_supply 'israel'


--äúçìä ùì äãåç òñ÷é
-- drop view v_sales
CREATE VIEW V_Sales as
	select c.Email as [customer email], o.OrderID as [order ID], o.OrderDate as [order date], o.Country as [country], o.ShippingType as [Shipping Type], o.price as [order price], fu.ItemID as [furniture ID], co.quantity as [quantity] ,fu.collection as [furniture collection], fu.MainType as [furniture Main Type]
	from Customers as c  full join creditcards as cc on c.email= cc.email 
	full join orders as o on o.cardnumber=cc.cardnumber
	full join Favorites as f on f.Email = c.Email 
	full join [Contains] as co on co.OrderID = o.OrderID
	full join Furnitures as fu on fu.ItemID = co.ItemID
	group by c.Email, cc.CardNumber, o.OrderID, o.OrderDate, o.Country, o.ShippingType, o.price, fu.ItemID, fu.collection, fu.MainType, co.quantity

select *
from V_Sales


--drop view view_payments 
create view view_payments as
select c.ItemID, payment = sum(o.price), [counter] = count (*)
from orders as o join [contains] as c on o.OrderID = c.orderID
group by c.ItemID

select*
from view_payments


select itemID, profits = payment,
	NTILE (10) OVER (ORDER BY payment) as [Deciles], 
	[counter] as [Number of sold items], 
	(round (PERCENT_RANK() OVER (ORDER BY [counter]), 3))*100 as [Distribution by percentages of item sales]
from  view_payments
order by 2 desc


--drop view view_selledCollections 
--îøàä àú ääëðñä îëì àåñó àú äãéøåâ ùìå åàú äãéøåâ äéçñé
create view view_selledCollections as
SELECT [collection], payments = sum(o.price), 
       (select count(fu.collection)
	    from Furnitures as fu
		where f.collection = fu.collection) as [Number of items]	
from  Furnitures as f join [Contains] as c on f.ItemID = c.ItemID
join orders as o on o.OrderID = c.OrderID
group by [collection]


select [collection], [Avg income per item] = payments/[Number of items] ,profits = payments, 
	   rank() over (order by payments/[Number of items] desc) as [rank by ratio],  
	   round(cume_dist() over(order by payments/[Number of items] asc),3)*100  as [Distribution by percentages of collection sales],
	   rank() over (order by payments desc) as [rank by profits]
from view_selledCollections
order by [rank by ratio]

create view v_furnitures_rank as 
select c.ItemID, quantitiy = sum(c.quantity),RANK() over (order by sum(c.quantity)) rank
from  orders as o join [Contains] as c on o.orderid= c.orderid
join Furnitures as f on f.ItemID=c.ItemID
group by c.ItemID 

create function function_items_quan ( @num int)
returns table as return
select *
from v_furnitures_rank
where rank<= @num

-------drop procedure sp_make_disc
create procedure sp_make_disc  (@disc as  float , @num  as int)
as begin
update Modifications
set price = price*(1-@disc)
where Modifications.ItemID in (select ItemID from dbo.function_items_quan(@num))
end


with
-- ñëåí îëéøåú ùì ëåì îãéðä
countrySale (countryS ,[sum country Sales]) as
(select o.Country , sum(o.price)
from orders as o	
group by o.Country),

--ëîåú ì÷åçåú áëì îãéðä
countryCostomers (countryC, [coustemer amount]) as
(select o.country, count(distinct c.Email)
 from orders as o join CreditCards as cc on o.CardNumber = cc.CardNumber
	  join Customers as c on c.Email = cc.Email
 group by o.Country),

--ëìì äøååçéí ùì äçáøä
totalP as
(select sum(o.price) as [total profit]
 from orders as o),
 
--àçåæ øååç îëìì äøååçéí ùì äçáøä
countryProfitP (countryP, [profit percent]) as
(select cs.countryS, [profit precent] = cs.[sum country Sales]/[total profit]
 from countrySale as cs cross join totalP),

-- ëîåú äàéñåôéí
countryPickUps (countryPU, [Pick Ups amount]) as
(select o.country, pickups = count (*)
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber
where o.shippingtype= 'pick up' 
group by o.country),

-- ëîåú äîùìåçéí
countryShipping (countrySh, [Shipping amount]) as
(select o.country, Shipping = count (*)
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber
where o.shippingtype= 'Shipping' 
group by o.country)
 
select cs.countryS, cs.[sum country Sales], cc.[coustemer amount], cp.[profit percent], cpu.[Pick Ups amount], csh.[Shipping amount]
from countrySale as cs join countryCostomers as cc on cs.countryS = cc.countryC
     join countryProfitP as cp on cs.countryS = cp.countryP 
	 join countryPickUps as cpu on cs.countryS = cpu.countryPU
	 join countryShipping as csh on cs.countryS = csh.countrySh
