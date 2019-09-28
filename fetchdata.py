import os
from dotenv import load_dotenv
import requests
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import datetime

#api key
env_path = Path('config/')
load_dotenv(dotenv_path=env_path)
apikey = str(os.environ.get("ALPHAVANTAGE_API_KEY"))

#requesting ticker for the stock as JSON
while True:
    ticker = input("\nPlease enter the ticker of your stock of choice: ") 
    
    if ticker.isdigit():
        print("Invalid entry: a stock ticker only uses characters - integers or symbols are not permitted")
    else:
        #print(apiurl_daily_adjusted)
        pull = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=" + ticker + "&outputsize=compact&apikey=" + apikey)
        
        if "Error" in pull.text:
            print("Error: stock either cannot be found or is not listed on Alpha Vantage - please enter another stock ticker")
        else:
            break  
j=pull.json()

#variables created for the date to be used later
time = datetime.datetime.now()
a = time.strftime("%Y")
b = time.strftime("%m")
c = time.strftime("%d")    
d = time.strftime("%I")
e = time.strftime("%M")
f = time.strftime("%p")

#adds values pulled from Alpha Vantage to form DataFrame
t,opn,h,l,close,adj_close,vol,div = [],[],[],[],[],[],[],[]
for lx, value in j["Time Series (Daily)"].items():
    t.append(lx)
    opn.append(float(value["1. open"]))
    h.append(float(value["2. high"]))
    l.append(float(value["3. low"]))
    close.append(float(value["4. close"]))
    adj_close.append(float(value["5. adjusted close"]))
    vol.append(float(value["6. volume"]))
    div.append(float(value["7. dividend amount"]))

#data headers are formatted in order to be put into a CSV
print("\n --------------------------------------------------- \n")
print("Stock Ticker: " + ticker)
output = pd.DataFrame(
    {
        "Date":t, "Open":opn, "High": h, "Low":l, "Close":close, "Adjusted_Close":adj_close, "Volume": vol, "Dividends": div 
    }
)
print(output.head(), '\n')

#deletes a file if it is named in the same way (the data would essentially be the same)
while True:
    if os.path.isfile("data/" +ticker + "_" + a + b  + c + ".csv"):
        os.remove("data/" + ticker + "_" + a + b  + c + ".csv")
    else:
        break

#data is pushed into a CSV file
output.to_csv("data/" + ticker + "_" + a + b  + c + ".csv")

print("File saved as " + ticker + "_" + a + b  + c + ".csv in the 'data' folder \n")
print("--------------------------------------------------- \n")