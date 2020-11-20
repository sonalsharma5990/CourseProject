
## Stock price data
We tried getting American Airlines stock data from Yahoo Finance using old Nasdaq code AAMRQ  but it doesn't recognize AAMRQ as valid symbol. We could search by American Airlines new Nasdaq Code AAL but that was available from 2005 onwards.

Google Finance does show historical data for American Airlines, but thier API
is discontinued. On further search, we found that google finance formula can be used
in google sheets to fetch stock prices, we used below formulas to pull stock prices 
for Apple (AAPL) and American Airlines (AAMRQ) for period July-2000 to Dec-2001.

```
=GOOGLEFINANCE("NASDAQ:AAL", "price", DATE(2000,7,1), DATE(2001,12,31), "DAILY")
=GOOGLEFINANCE("NASDAQ:AAPL", "price", DATE(2000,7,1), DATE(2001,12,31), "DAILY")
```

## IEM 2000 Winner takes all data
The data from May-2000 to Nov-2000 was manually selected from IEM Website
[IEM 2000 U.S. Presidential Election: Winner-Takes-All Market](https://iemweb.biz.uiowa.edu/pricehistory/pricehistory_SelectContract.cfm?market_ID=29)

The data for each month was selected using dropdown and copied to a spreadsheet. After data for all months have been collected, the spreadsheet is saved as an CSV file.


