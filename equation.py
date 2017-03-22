#assumes that column name target is the target variable
from sklearn.linear_model import LogisticRegression
from treatments import conversion
import pandas as pd
def fit(data,sheetName='sheet1',variables=''):
    df = data
    X, y = df[:].loc[:, variables], df[:].loc[:, ['target']]
    OVR = LogisticRegression().fit(X, y)
    print OVR.coef_
    print 'score is' + str(OVR.score(X, y))
    return OVR
def predict(equation,df,output='output\\test.csv'):
    df1,variables=conversion(df)
    X = df1[:].loc[:, variables]
    fitted = pd.DataFrame(equation.predict_proba(X),columns=['high','medium','low'])
    fitted['listing_id']=df['listing_id']
    summit=pd.read_csv("input\\sample_submission.csv")
    summit=summit[['listing_id']]
    summit['key']=range(1, len(summit) + 1)
    result=fitted[['high','medium','low']]
    result['key']=range(1, len(result) + 1)
    final=summit.merge(result,on='key',how='left')
    final=final[['listing_id','high','medium','low']]
    final=final.set_index('listing_id')
    final.to_csv(output, sep=',')

