import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from pandas import get_dummies

def make_network(FILENAME,sklearn=False,keras=False,normalize=True,spread=False,moneyline=False,tpot = False):
    from pandas import read_csv,get_dummies
    import numpy as np
    from sklearn import cross_validation
    from sklearn.neural_network import MLPClassifier
    
    
    """
    Given the csv input of all the box scores, arrange it such that the home and away teams are lined up, 
    unnecessary columns removed, and hot encoding is done. Other stuff too probably. Such as normalization, but I 
    didn't do that!
    
    Note that this data has already been doctored from its original form, taking out most unnecessary columns but
    those could be useful later on.
    
    
    Parameters
    ----------
    FILENAME : file
        The csv of the data.
        
    sklearn : bool
        True or false designation for if you want the MLP to be based on an sklearn version 
    keras : bool
        True or false designation for if you want the MLP to be more manually designed using Keras. 

    normalize : bool
        True or false designation for if you want to set all relevant inputs onto the same scale. 
        
    spread : bool
        True or false designation for if you want to predict the spread. 

    
    moneyline : bool 
        True or false designation for if you want to predict the outright winner. 

    
    
        
    Returns
    -------
    
    
    model : NN
        The neural network fitted on the training and validation dataset so it can be applied to future data. 
    
    scaler : 
        The scale used to normalize the training data in order to impose this onto the prediction data. 
        
    """
    
    #Read in file, remove attempted and # and only account for % since that's more predictive in nature. 
    #*retrospectively that doesn't make sense, could be worth changing!
    data = read_csv(FILENAME) 
    data['3P%'] = np.divide(data['3P'].values,data['3PA'].values) 
    del data['3P'],data['3PA']
    data['FG%'] = np.divide(data['FG'].values,data['FGA'].values)
    del data['FG'],data['FGA']
    data['FT%'] = np.divide(data['FT'].values,data['FTA'].values)
    del data['Unnamed: 0'],data['GAME_ID'],data['Date'],data['Home'],data['Away'],data['PLUS_MINUS'],data['TOTAL']
    del data['FT'],data['FTA']
    data = get_dummies(data)

    #print(data)
    

    dat = []
    
    #reshape the dataset so now each colummn has roadstats and homestats concatenated into the same row, used for NN 
    
    for i in range(len(data.values)):
        data.values[i] = np.reshape(data.values[i],newshape=[1,len(data.values[i])])
    for p in range(int(len(data.values)/2)):
        fullboxgame = np.concatenate((data.values[2*p],data.values[(2*p)+1]))
        dat.append(fullboxgame)
    
    #convert list to array, now possible to array operations previously not possible
    dat = np.array(dat)   
    
    openingspreadS = dat[:,8] #what the predicted spread of ther game was. 
    roadpts = dat[:,7]       #column of all the points scored by road team 
    homepts = dat[:,52]
    endspreadS = roadpts-homepts  #all the final spreads of the game
            #concatenating all the arrays, looks messy but explanation doen in another nb. 
    x1 = dat[:,0:7] #road offensive rebounds to blocks
    x2 = dat[:,9:42] # road 3p% to team name (hot encoded)

    x3 = dat[:,45:52] #home offensive rebounds to blocks
    x4  =  dat[:,54:87] #home 3p% to hot encoded team name   
                      
    x5 = dat[:,8]              
    X1 = np.concatenate((x1,x2),axis=1)
    X2 = np.concatenate((x3,x4),axis=1)
    X3 = np.concatenate((X1,X2),axis=1)
    
    y = []
    
    if spread:
        #include initial spread of the game. 
        X = np.column_stack((X3,x5))

        for j in range(len(endspreadS)):  
            openspread = openingspreadS[j]
       # print("this is the spread of the road team " + str(openspread))
            endspread = endspreadS[j]
       # print("the road team won by  .. " + str(endspread))
       # if endspread>openspread:
        #    y.append(np.array([0,1,0]))  #OK, now make sure this is formateed properly!
            if openspread + endspread <0:
                y.append(np.array([0,1,0]))  #home team covered
            elif openspread + endspread >0:
                y.append(np.array([1,0,0]))  #road covered
            else: 
                y.append(np.array([0,0,1]))  #push!

    
    if moneyline:
        X = np.column_stack((X3,x5))
        #Spread is still a useful property for this type of bet. The spread implies the favorite! 
        for j in range(len(endspreadS)):  
            if endspreadS[j]<0:
                #means the home team had more points
                y.append(np.array([0,1]))
            else:
                y.append(np.array([1,0])) #alternatively, a road team victory. 
          

    #Now I iterated over all these, and hot encoded the labels of each to see whether or not the spread was covered
    #and by what team. 


        
    y = np.array(y)  #same explanation as above
                         
    #since everything got out of order I have to mash it together myself. 
    if normalize:
        
        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))

        scaler.fit(X)
        X = scaler.transform(X)
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.27)
    #print((X[0]))
    #print(np.shape(X[0]))

    #now to construct a model 
    if sklearn: 
        model = MLPClassifier()
        model.shuffle = True
        model.batch_size = 25
    #model.n_layers_ = 1000000
    #model.n_outputs_= 1000000
    #These don't do anything, have to adjust the layers in some different way! Keras is useful for this.
        model.fit(X_train,y_train)
        print(model.score(X_test,y_test))
    if keras:
        print("keras NN goes here")
        model=Sequential()
        model.add(Dense(22,input_dim=np.shape(X)[1],activation='relu'))
        model.add(Dense(30,activation='relu'))
        model.add(Dense(30,activation='relu'))
        model.add(Dense(22,activation='relu'))
        if spread:
            model.add(Dense(3,activation='softmax'))
        if moneyline: 
            model.add(Dense(2,activation='softmax'))  #different outputs for the 2 problems!

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(X_train,y_train,batch_size=40,epochs=20,validation_split=.2)
        scores = model.evaluate(X_test,y_test)
        print(scores[1])

    if tpot: 
        y2 = []
        for i in range(len(y_train)):
                if sum(y_train[i] == np.array([1, 0, 0])) == 3:
                    y2.append(0)
                else:
                    y2.append(1)
        
        from tpot import TPOTClassifier
        tpot = TPOTClassifier(generations = 5,population_size = 50,verbosity = 2, n_jobs = -1)
        tpot.fit(X_train,y2)
            #experimental. Trying to use genetic programming to identify optimal routine for game classification. 
    
        model = tpot #the returned model 
    return model,scaler
