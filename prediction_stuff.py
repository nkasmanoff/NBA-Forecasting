#In here is how we use current NBA statistics to predict the outcome of future games. 
import pandas as pd
def get_splits(TEAMABR,ten=False,twenty=False,regseason=True):
    """returns the splits of a team over the past N days. Will consider changing this to a from - to thing for different dates. 
    
    Parameters
    ----------
    TEAMABR : str
    	Abbreviation of the desired team. Ex: Atlanta = ATL.  

    Returns 
    -------
    teamsplits : array	
    	Array of desired team's statistics over the previous N games. 

    """
    import numpy as np
    from nba_py import team
    teams = team.TeamList()
    teamids = teams.info()
    teamids = teamids[:-15]
    teamids = teamids[['TEAM_ID','ABBREVIATION']]

    teamids = teamids.rename(index=str, columns={"ABBREVIATION": "Team"})
    teamids = teamids.replace('BKN','BRK')
    teamids = teamids.sort_values('Team')

    TEAM_ID = teamids.loc[teamids['Team'] == TEAMABR].values[0,0]
    teamids = pd.get_dummies(teamids)
    teamarray = teamids.loc[teamids['TEAM_ID'] == TEAM_ID].values[0,1:]


    TEAM = team.TeamLastNGamesSplits(team_id=TEAM_ID,season='2017-18')
    if ten:
        df = TEAM.last10()
    if twenty:
        df = TEAM.last20()
    if regseason:
        TEAM = team.TeamGeneralSplits(team_id=TEAM_ID,season='2017-18')
        df = TEAM.overall()
   # if five:
        #df = TEAM
    
    df = df[['OREB','DREB','REB','PF','STL','TOV','BLK','FG3_PCT','FG_PCT','FT_PCT']].values
    
    teamsplits = np.concatenate((df[0],teamarray))
    return teamsplits

def OU_ML_game_maker(roadteam,hometeam,spread,scaler):
    import numpy as np

    """
        After creating a properly formated table, this concats the desired teams so they can be predicted. 
        Based on get team index # based on output of predictor, and make it the input for stats ie GSW are stats[0].
        and so on!
        
    Parameters 
    ----------
    roadteam : arr
        split array of the road team information coming from the get splits function. 

    hometeam : arr
        split array of the home team information coming from the get splits function. 
    
    scaler : sklearn.preprocessing.data.MinMaxScaler
        The normalization scale used by the NN model to fix the splits used above. 
    
    spread : float 
        The spread of the game. 
    Returns
    -------
    
    game : arr 
        The combined array of the inputs above. Used to "reflect" the potential result of an incoming game. 
    """
    game = np.concatenate((roadteam,hometeam))
    spread = np.array([spread])
    game = np.concatenate((roadteam,hometeam))
    game = np.concatenate((game,spread))
    game = [game]
    game = scaler.transform(game)
    return game


def spread_game_maker(roadteam,hometeam,spread,scaler):
    import numpy as np

    """
    
    The function used to create the spread prediction input used to predict the winner of the game. 
    Parameters
    ----------
    
    roadteam : arr 
        split array of the road team information coming from the get splits function. 
    hometeam : arr
        split array of the home team information coming from the get splits function. 

    spread : float
        The spread of the game. To avoid the +- confusion, this is the spread for the road team. 

    scaler : sklearn.preprocessing.data.MinMaxScaler
        The normalization scale used by the NN model to fix the splits used above. 
    
    Returns 
    -------
    
    game : arr 
        The combined array of the inputs above. Used to "reflect" the potential result of an incoming game. 
    """
    spread = np.array([spread])
    game = np.concatenate((roadteam,hometeam))
    game = np.concatenate((game,spread))
    game = [game]
    game = scaler.transform(game)
    return game

    