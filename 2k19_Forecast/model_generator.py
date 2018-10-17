"""Contained in this file is the data wrangled, 
and then resulting model formed from it. 


The model is based off of the previous 
5 seasons of NBA data, only working within the regular season. 


This data consists of the rolling 5 game averages various team statistics of the playing NBA teams, 

and uses this to determine the forecasted winner, spread winner, and over under. 


"""


import pandas as pd
import numpy as np
FILENAME = 'NBADATA.csv'

def create_rolling_dataset(window,FILENAME,spread=False,winner=False):
    """Uses the dataframe to create rolling averages, in 
        an attempt to capture a team's performance over an N game window prior to the game. 
    """
    
    #retain relevant columns. 
    data = pd.read_csv(FILENAME) 
    data['3P%'] = np.divide(data['3P'].values,data['3PA'].values) 
    del data['3P'],data['3PA']
    data['FG%'] = np.divide(data['FG'].values,data['FGA'].values)
    del data['FG'],data['FGA']
    data['FT%'] = np.divide(data['FT'].values,data['FTA'].values)
    del data['Unnamed: 0'],data['PLUS_MINUS'],data['TOTAL']
    del data['FT'],data['FTA']
    del data['OU']
    data = data.loc[data['GAME_ID'].values < 41300001] #genius! No playoff games now :)   
    #del data['Team'] 
    #data = pd.get_dummies(data) #sometimes option to hot tcode team, but not yet. Seems like overfitting. 
    teams = data.Team.unique() #each nba team. 
#iterate over those teams, make a rolling window over n games. 
    N_GAMES = window  #different possible rolling window sizes. 
    nba_data = pd.DataFrame([])
    season_ids = []
    for i,val in enumerate(data['GAME_ID'].values):  #loop through every game
        season_ids.append(str(val)[1:3])

    data['Season_ID'] = season_ids #identify the unique seasons. 

    for team in teams:  #for each team
       # print(team)
    #get separate seasons here
        team_data = data.loc[data['Team'] == team]  #this contains the box score of every team game from 2013 to 2018.
        for season in data['Season_ID'].unique(): #this contains the box score of that team for that season. 
            #print(season)
            team_season = team_data.loc[team_data['Season_ID'] == season]
        
            stuff_to_turn_into_avgs = ['OR', 'DR', 'TOT', 'PF', 'ST', 'TO', 'BL', '3P%', 'FG%', 'FT%']
            for col in team_season.columns:
                if col in stuff_to_turn_into_avgs:
                        team_season['Rolling ' + col] = team_season[col].rolling(window=N_GAMES).mean().shift(1)

            #split each season up here, 
                    #if col != 'PTS':
                    #    team_season['Rolling ' + col] = team_season[col].rolling(window=N_GAMES).mean().shift(1)

                        del team_season[col]
            nba_data =  nba_data.append(team_season)

           # df = pd.concat([road_df,home_df],axis=1)
#reorganize the dataset. 
    nba_data_splits = nba_data.sort_values(by = ['GAME_ID', 'Home','Away'], ascending=[True, True,False])

    nba_data_splits.dropna(inplace=True)  #null values come with rolling means, drop those now. 
    del nba_data_splits['GAME_ID'],nba_data_splits['Date'],nba_data_splits['Home'],nba_data_splits['Away'],nba_data_splits['Team']
    del nba_data_splits['Season_ID']
 
    #delete columns no longer of use, ie team name etc. Can consider keeping team name and see if helps chances. 
    #now align the box scores so its one big one for each game, home team and road teams. 

    road_df = nba_data_splits.iloc[::2]
    home_df = nba_data_splits.iloc[1::2]
    for col in nba_data_splits.columns:
        road_df['road_' + col] = road_df[col]
        home_df['home_' + col] = home_df[col]
    
        del road_df[col],home_df[col]

    home_df.reset_index(inplace=True)
    road_df.reset_index(inplace=True)

#merged into a dataframe here. 
    df = pd.concat([road_df,home_df],axis=1)
    del df['index']

#create the dataset here. Can consider the spread, or winner. 
#at the moment only using a single classifier, that seems sufficient. A home team loss is synonymous with a road team win. 

    df['final_SPREAD'] = df['road_PTS'] - df['home_PTS']
    del df['road_PTS'], df['home_PTS'],df['home_SPREAD']
           # if openspread + endspread <0:
            #    y.append(np.array([0,1,0]))  #home team covered
            #elif openspread + endspread >0:
            #    y.append(np.array([1,0,0]))  #road covered
           # else: 
           #     y.append(np.array([0,0,1]))  #push!
    y = []

    if spread: 
        for i in range(len(df)):
            if df['road_SPREAD'].values[i] + df['final_SPREAD'].values[i] < 0:
                y.append(1) #home team covers
            else: # df['road_SPREAD'].values[i] + df['final_SPREAD'].values[i] > 0:
                y.append(0) #road team covers or push
    #else:
    #    y.append(np.array([0,1]))  #push! 
    
    if winner:
        for i in range(len(df)):
            if df['final_SPREAD'].values[i] < 0: #home team won. 
                y.append([0,1])
            else:
                y.append([1,0]) #road team won. 

    #del df['final_SPREAD']

#y_names = np.array(['road team win', 'home team win']) #for preprocessing/visualization. 
    X = df
    return X,y



X , y = create_rolling_dataset(window=5,FILENAME='NBADATA.csv',spread=True)






X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

model.score(X_train,y_train)

model.score(X_test,y_test)


