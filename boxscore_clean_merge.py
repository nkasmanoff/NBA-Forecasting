# Clean up the boxscore data extracted from nba_py

import pandas as pd
from datetime import datetime

all_boxscores = pd.read_csv('boxscore_data_wdates_1318.csv', dtype = 'str')
spreads = pd.read_csv('spread_data.csv')

 # Make Date strings into datetime objects
def str2date(datestr):
    return datetime.strptime(datestr, '%A, %B %d, %Y').date()


all_boxscores['GAME_DATE'] = all_boxscores['GAME_DATE'].apply(str2date)

# remove all star and preseason games (game id = 001XXX,003XXX)
pattern = "(^001|^003)[0-9]{7}"
patterntf = all_boxscores['GAME_ID'].str.contains(pattern)
all_boxscores = all_boxscores[~patterntf]

# Spread data extracted during a game and game data after, just going to drop that game's data so they match.
all_boxscores = all_boxscores.drop(all_boxscores.index[len(all_boxscores)-3:len(all_boxscores)-1])

# To concatenate we need a column that will match the output from OddsShark.com
all_boxscores['team'] = all_boxscores['TEAM_CITY'] + " " + all_boxscores['TEAM_NAME']
# Some issues with team names not matching due to name changes between 2013 and 2015
all_boxscores['team'] = all_boxscores['team'].replace(['New Orleans Hornets','Charlotte Bobcats'], ['New Orleans Pelicans','Charlotte Hornets'])
all_boxscores['team'] = all_boxscores['team'].replace(['LA Clippers'],['Los Angeles Clippers'])

all_boxscores['date'] = all_boxscores['GAME_DATE']

# Just fix some of the spread data so we can merge.
def str2date2(datestr):
    return datetime.strptime(datestr, '%Y-%m-%d').date()

spreads['date'] = spreads['date'].apply(str2date2)

uncleaned = pd.merge(all_boxscores,spreads, on=['date','team'])
uncleaned.to_csv('boxscores_spreads_unclean.csv')



cleaned = uncleaned[['GAME_ID','date','team','home','away','FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
'OREB', 'DREB', 'AST', 'PF', 'STL', 'TO', 'BLK', 'PTS', 'spread', 'PLUS_MINUS', 'ou', 'total']]

# pd.concat([cleaned, uncleaned['ats']], axis= 1)[['PTS','spread','PLUS_MINUS', 'ats']]

cleaned.to_csv('boxscores_spreads_cleaned.csv')
