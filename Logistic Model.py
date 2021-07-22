import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from numpy import nan
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def preprocess(name):
    data = pd.read_csv(name)


    df=data.drop(columns=['X1','X7'])
    values=df['X3'].values
    for i in range(values.size):
        if (values[i][0]=='L'or values[i][0]=='l'):
            values[i]='Low Fat'
        elif(values[i][0]=='R'or values[i][0]=='r'):
            values[i]='Regular'

    df['X3']=values

    df['X2'].fillna(method ='bfill', inplace = True)
    df['X4']=df['X4'].replace(0,nan)
    df['X4'].fillna(method='bfill',inplace=True)
    df['X9'].fillna(method='bfill',inplace=True)

    label_encoder=preprocessing.LabelEncoder()
    df['X3']=label_encoder.fit_transform(df['X3'])
    df['X5']=label_encoder.fit_transform(df['X5'])
    df['X9']=label_encoder.fit_transform(df['X9'])
    df['X10']=label_encoder.fit_transform(df['X10'])


    # print(df.corr())
    df2 = df.drop(columns=['X2','X3','X4','X5','X6'])
    X=df2[['X8','X9','X10']]
    # print(df.describe())
    norm=MinMaxScaler().fit(X)
    df2[['X8','X9','X10']]=norm.transform(df[['X8','X9','X10']])
    # print(df2.corr())

    return df2

df=preprocess('train.csv')
x = df[['X8','X9','X10']].values
y = df['Y'].values
df_test=preprocess('test.csv')
x_test = df_test.values
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)


logisticReg= LogisticRegression()
logisticReg.fit(X_train,Y_train)
logisticReg.predict_proba(X_test)
pred=logisticReg.predict(X_test)
score=logisticReg.score(X_test,Y_test)
print('Logistic Model: ',score*100)

logisticReg.predict_proba(x_test)
y_pred=logisticReg.predict(x_test)
# pd.DataFrame(y_pred,columns=['Y']).to_csv('LogisticPred.csv')


# plot=pd.DataFrame()
# plot['Target']=Y_test
# plot['Predictions']=pred
#
# sns.lmplot('Target','Predictions',data=plot,line_kws={'color':'green'},scatter_kws={'color':'blue'})
# plt.title('Logistic Model',size=10)
# plt.show()
