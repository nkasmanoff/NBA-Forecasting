#extract boxscores back 2013 and concatenate with date data

import nba_py
from nba_py import game
import pandas as pd
from datetime import date, timedelta
from time import sleep

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

all_boxscores = pd.DataFrame(columns = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS'])

game_ids = list()
game_days = []
start_date = date(2011, 3, 23) # October 29, 2013
end_date = date(2011, 3, 30)
for single_date in daterange(start_date, end_date):
    ids = list(nba_py.Scoreboard(day = single_date.day, month = single_date.month, year = single_date.year).available()['GAME_ID'].astype('str'))
    game_ids = game_ids + ids
    sleep(0.05)
    print(single_date)
    for game_id in ids:
        all_boxscores = all_boxscores.append(game.Boxscore(game_id).team_stats())
        # gamedays = gamedays + [single_date]
        sleep(0.05)
#writes to  csv. 
all_boxscores.to_csv('boxscore_data_1318.csv')

#adds box score. 
all_boxscores = pd.read_csv('boxscore_date_1318.csv')
game_date = pd.DataFrame()
game_date = [game_date.append(nba_py.game.BoxscoreSummary(game).game_info()['GAME_DATE']) for game in all_boxscores['GAME_ID']]
