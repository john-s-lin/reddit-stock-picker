# Reddit Stock Picker

Our Code Submission contains the following files:
|                         |                                              |
|-------------------------|----------------------------------------------|
| ./code/Scape.ipynb      | Scrapes stock data                           |    
| ./code/Stocks.ipynb     | Preprocesses the stock data                  |     
| ./code/TimeSeries.ipynb | Trains models with a timeseries split on data|         
| ./data/out.csv          | Data from just r/wallstreet                  |
| ./data/out2.csv         | Data from r/wallstreet and r/stocks          | 
| ./G17_report.pdf        | report in compiled pdf format                |  


## Running
- GPU is not required
- Training per 4 models may take half an hour
  - there are 12 available models
- the files needed to run TimeSeries are present
- output of Scrape files/input to Stocks are not included
- the scraped data can be found (here)[insert link]

### Package Requirements
- PRAW           
- sklearn        
- pandas         
- vaderSentiment 
- yfinance       


### Order to run notebook files
```
--> Scrape.ipynb
--> Stocks.ipynb
--> TimeSeries.ipynb
```