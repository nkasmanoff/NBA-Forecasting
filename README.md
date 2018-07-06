NBA Forecasting
===============

By Aaron Hunt, Sarang Yeola, Noah Kasmanoff, and Jordan Corbett-Frank



Introduction
============

Contained in this repository is the data scraping and machine learning routines used by our team in order to predict the outcome of an unplayed NBA game, using the results of many previous games. 

The program inputs the box score of a game, and using that assigns an output corresponding to the winner of the game([1,0] denoting a victory for the road team, [0,1] home team etc.), the winner of the spread of the game, or whether or not the designated points total over/under was met or not. This problem can be considered a classification problem in the machine learning domain. Using an artificial neural networks, training and testing can be done on existing NBA game box scores, and then prediction can be done on unplayed NBA games by using current team statistics as an indicator of how that team will perform in a future game as a prediction tool for helping forecast the winner, spread, and over/under outcomes of any given NBA game. We freely admit this is not a perfect way to go about predicting a spread winner as many more factors may come into play such as injuries, momentum, etc., but this technique provides some computational evidence to serve as a guidepost for predicting the outcome of NBA games. 

Among this there are other ML based methods included, such as a time series analysis of different NBA teams' seasons, and using different approaches in order to forecast that team's success over the next few seasons. 



NBA Game Classification Modules/Files
=====================================

Further documentation for each is provided within the functions themselves. This is a brief introduction of each. 

NNBA.py
-------
This module contains the neural network routine used to train and test on previous seasons of data in order to create an MLP classifier for future prediction. Along with the NN, the module also creates a normalization scale which is used to help optimize inputs and training.

prediction_stuff.py
-------------------
This module contains scraping and cleaning functions that pull the splits of a given NBA team, and organize it into the right shape so that it can be inputted into the NN for prediction. 

boxscore_extract.py
-------------------
This file scrapes the nba_py software package in order to obtain the box scores of available nba games to be used by the model. 

spread_extract.py
-----------------
This file scrapes oddsshark.com for the spreads of available NBA games to be used by the model. 

boxscore_clean_merge.py
-----------------------
This file cleans the boxscore and spread functions to be used as an input and output for the model. 



Future Modules
===============

This is where Aaron's stuff goes... 







and more!# NBA-Forecasting
