"""Run this file "python nbabot.py", it will run the model, tell you the accuracy,
and will determine the spread winner and tweet said prediction. 

"""

import numpy as np
import NNBA
import prediction_stuff
import matplotlib.pyplot as plt
from datetime import datetime
np.random.seed(11)



#Step 1: Create prediction models. 
spread_model,scaler = NNBA.make_network('NBADATA.csv',sklearn=False,keras=True,normalize=True,spread=True)


moneyline_model,scaler = NNBA.make_network('NBADATA.csv',sklearn=False,keras=True,normalize=True,moneyline=True)

#Step 2: Load in today's games
prediction_stuff.predict_todays_games()


#Step 3: Output results into a tweet, make sure it is all read in clearly!


#Step 4: Check to see if line has moved. 