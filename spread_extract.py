# Get spread data from OddsShark.com
from bs4 import BeautifulSoup
import pandas as pd
import urllib2

data = pd.DataFrame(columns = ['Date', 'Opponent', 'Game', 'Result', 'Score', 'ATS', 'Spread', 'OU', 'Total', 'Preview', 'Recap', 'Team'])
for team in range(20722,20752): #20722 - 20752 Oddshark IDs for all the teams
    for year in range(2014, 2019): #Spread Data only goes back to 2013-2014 season
        url = str('https://www.oddsshark.com/stats/gamelog/basketball/nba/%d/%d' %  (team, year))
        page = urllib2.urlopen(url).read()
        soup = BeautifulSoup(page, "lxml")
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table))[0]
        df['Team'] = soup.find_all('h1')[0].text
        data = data.append(df)

data.to_csv('spread_data.csv')



