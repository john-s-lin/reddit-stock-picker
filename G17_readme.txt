# Reddit Stock Picker

Our Code Submission contains the following files:
./code/Scrape.ipynb       	<-- Scrapes stock data 
./code/Stocks.ipynb		<-- Preprocesses the stock data   
./code/TimeSeries.ipynb 		<-- Trains models with a timeseries split on data
./code/TrainTestSplit.ipynb   	<-- Trains models with a train test split on data
./data/out.csv           		<-- Data from r/wallstreetbets
./data/out2.csv           	<-- Data from r/Stocks
./G17_report.pdf          	<-- report in compiled pdf format

### Order to run notebook files
Training
--> TrainTestSplit.ipynb ~40min to run on one target set
--> TimeSeries.ipynb ~4h to run

Data Collection
--> Scrape.ipynb ~4h to run
--> Stocks.ipynb >24h to run


## Running
- GPU is not required
- Training per 4 models may take half an hour
  - there are 12 available model sets
- the files needed to run TimeSeries are present
- output of Scrape files/input to Stocks are not included

### Package Requirements
- praw
- sklearn
- pandas
- vaderSentiment
- yfinance
