
--הוספת טבלת חיפוש לבסיס הנתונים
Create Table Searches(
IP	        Varchar(15)   not null,
DT	        DateTime      not null, 
ItemID		  int         null,
Email 	    varchar(30),   
primary key(IP,DT),
constraint	FK_ItemID_Se
foreign key(ItemID)
references Furnitures(ItemID))

--תיקון ישות ההזמנות
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




--החברה רוצה להקטין את מבחר הרהיטים בישראל ולכן היא בודקת מה השלושה הרהיטים שהכניסו הכי מעט
select top 3 f.itemID, 'net value' = sum (m.price*c.quantity)
from furnitures as f join modifications as m on f.itemId=m.itemId
join [Contains] as c on f.itemID= C.itemId
join Orders as o on c.OrderID = o.OrderID
where o.country= 'israel'
group by f.itemID 
having count (*) < 20
order by 'net value' 

--המדינות עם הכי הרבה איסופים כדי לדעת איפה אפשר לצמצם אולי את המערך הלוגיסטי
select top 3 o.country, pickups = count (*)
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber
where o.shippingtype= 'pick up' 
group by o.country
ORDER BY pickups DESC


-- השוואת בין שני קולקשנים כדי להסיק נתוני מכירות והצלחיה במדינות השונות
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


--- מחזיר את המדינות שמכרו למספר לקוחות שונים מעל הממוצע העולמי בשנת 2020 
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


-- החברה החליטה לתת 10% מהקניות עד כה וגם לעתיד לטובת הזמנות חדשות
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


 -- מכל הלקוחות ששמרו לפחות שני פרטים במועדפים אך עדין לא ביצעו הזמנה כדי לשלוח קופון
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


-- החברה רוצה להסתיר את הכתובת המדויקת של הלקוחות וגם את הנתונים המלאים של כרטיס האשראי לטובת אבטחת מידע
create view view_CustomerOrder as
select c.Email, c.firstName, c.lastName , c.phone,
       o.OrderID, o.OrderDate, o.CardNumber, o.Country , o.ShippingType,o.price
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber


select *
from view_CustomerOrder


-- בודק כמה כל לקוח בזבז באתר שלנו לאורך חיי הלקוח
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


--בודק כמה מכירות התבצעו בין תאריכים ספציפים במדינה מסויימת לפי סוג משלוח
-- drop function function_CountrySellesByShippingType
create function function_CountrySellesByShippingType ( @Country varchar(20), @from date, @until date, @ShipingType varchar(20))
returns table as return
select c.ItemID ,count(distinct o.OrderID) as total_orders, sum(c.quantity) as total_amount
from Orders as o join [Contains] as c on o.OrderID = c.OrderID
where (o.OrderDate between @from and @until) and o.Country = @Country and o.ShippingType = @ShipingType
group by c.ItemID

select *
from dbo.function_CountrySellesByShippingType('denemrk','2021-12-12','2022-12-12','Pick Up')


--מחיקת רהיט ללא צורך להתחשב בטבלאות שהוא מפתח זר בהם מודיפיקשיין אז כשיש פעולת מחיקה הוא קודם מוחק בטבלת מפתח הזר ואז בטבלה הנחוצה
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


--החברה רוצה לדעת כמה רהיטים נמכרו לפני חודשים כדי לאפשר חיזוי הספקה לפי הצרכים של כל מדינה
create procedure sp_furniture_supply (@country varchar (20))
	as begin 
			select c.itemid, total_amount=  sum(c.quantity)
			from orders as o join [Contains] as c on o.orderid= c.orderid
			where year (o.orderdate)= year (getdate()) and month(o.orderdate)= month(getdate())-2
			and o.country = @country
			group by c.itemid,c.quantity
			end

execute sp_furniture_supply 'israel'


--התחלה של הדוח עסקי
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
--מראה את ההכנסה מכל אוסף את הדירוג שלו ואת הדירוג היחסי
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
-- סכום מכירות של כול מדינה
countrySale (countryS ,[sum country Sales]) as
(select o.Country , sum(o.price)
from orders as o	
group by o.Country),

--כמות לקוחות בכל מדינה
countryCostomers (countryC, [coustemer amount]) as
(select o.country, count(distinct c.Email)
 from orders as o join CreditCards as cc on o.CardNumber = cc.CardNumber
	  join Customers as c on c.Email = cc.Email
 group by o.Country),

--כלל הרווחים של החברה
totalP as
(select sum(o.price) as [total profit]
 from orders as o),
 
--אחוז רווח מכלל הרווחים של החברה
countryProfitP (countryP, [profit percent]) as
(select cs.countryS, [profit precent] = cs.[sum country Sales]/[total profit]
 from countrySale as cs cross join totalP),

-- כמות האיסופים
countryPickUps (countryPU, [Pick Ups amount]) as
(select o.country, pickups = count (*)
from Customers as c join creditcards as cc on c.email= cc.email
join orders as o on cc.cardnumber= o.cardnumber
where o.shippingtype= 'pick up' 
group by o.country),

-- כמות המשלוחים
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
