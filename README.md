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

#### Intellij?
- `import remote git repo`
- `open up bash terminal`
- `cd src`
- `python <filename>`

### `Jupyter Notebook Files`
#### `locally`
- `install jupyter-notebook`
  - *varries based on operating system*
- `run it targeting the src directory`
  - *On emacs use `M-x ein:run` to have it start and manage your session with the `emacs-ipython-notebook` package installed
  - *follow your ide/editor's instructions based on which you use*

#### `browser`
- `go to `[jupyter](https://www.jupyter.org)
- `after logging in upload the file to you notebook`
- `run it`


## Crawling
Our data that was scraped exceeded 800mbs while zipped so it has been excluded from the repository. But they may be generated with running the Scrape.ipynb notebook. Scraping should be noted to take a very long time if date range is not adjusted, 4+ hours.


## Running
### Order to run notebook files
```
--> Scrape.ipynb
--> Stocks.ipynb
--> TimeSeries.ipynb
```