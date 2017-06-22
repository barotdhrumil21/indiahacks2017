#from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
#import matplotlib.pyplot as pp


filename = 'aa.csv'
dataframe = pd.read_csv(filename)



def convert_to_int(dataframe):
    str = dataframe.loc[:, ['DetectedCamera', 'SignFacing (Target)']].as_matrix()
    for i in range(len(str)):
        if str[i, 0] == 'Front':
            str[i, 0] = 1
        elif str[i, 0] == 'Rear':
            str[i, 0] = 2
        elif str[i, 0] == 'Left':
            str[i, 0] = 3
        elif str[i, 0] == 'Right':
            str[i, 0] = 4
        try:
            if str[i, 1] == 'Front':
                str[i, 1] = 1
            elif str[i, 1] == 'Rear':
                str[i, 1] = 2
            elif str[i, 1] == 'Right':
                str[i, 1] = 4
            elif str[i, 1] == 'Left':
                str[i, 1] = 3
        except:
            pass
    dataframe.loc[:, 'DetectedCamera'] = str[:, 0]
    dataframe.loc[:, 'SignFacing (Target)'] = str[:, 1]

convert_to_int(dataframe)
target = dataframe.loc[:, str('SignFacing (Target)')]
dataframe = dataframe.drop(['SignWidth', 'SignHeight','Id','SignFacing (Target)'], axis=1)


test=pd.read_csv('test2.csv')
convert_to_int(test)
test_id=test['Id']
test=test.drop(['SignWidth', 'SignHeight','Id','SignFacing (Target)'],axis=1)


df=RandomForestClassifier()
#pp.plot(df)
df.fit(dataframe, list(target))
scores=cross_val_score(df,X= dataframe.values, y=list(target))
pred = df.predict_proba(test)

#print("pred : ",pred)
#pred=np.array(pred)
#np.set_printoptions(threshold='nan')
#print(pred)

columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("sub_rfih.csv", index=False) #99.8XXX
