# Reddit Stock Picker

## Overview
### Problem
In this day and age everyone probably owns a stock of some sort, and most probably bought it from the comfort of their home on an app, but why? Well the primary goal is to dabble or often make an eventual profit, but making a profit with stocks is risky. 

### Solution
Our solution to reducing the risk that is involved with the assistance of machine learning. And as a resource to train the model we utilized the resource of reddit's r/wallstreet's hive mind, scraping all posts for the last 3 years and then proceeding to filter only relevant comments from there.

## Setup
### `.py` files

#### cli
- `git clone github.com/<project_repo> <destination folder>`
- `cd <destination folder>/src`
- `python <filename>`


## Crawling
Our data that was scraped exceeded 800mbs while zipped so it has been excluded from the repository. But they may be generated with running the Scrape.ipynb notebook. Scraping should be noted to take a very long time if date range is not adjusted, 4+ hours.


## Running
### Order to run notebook files
```
--> Scrape.ipynb
--> Stocks.ipynb
--> TimeSeries.ipynb
```