import json,csv
import pandas as pd

import statsmodels.api as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
raw=pd.read_json('D:\\vivek\\rentHop\\train.json')
test=pd.read_json('D:\\vivek\\rentHop\\test.json')
raw['target']=raw['interest_level'].map({'high':1,'low':2,'medium':3})
X,y=raw[:].loc[:,['bathrooms','bedrooms','price']], raw[:].loc[:,['target']]
tX=test[:].loc[:,['bathrooms','bedrooms','price']]
from sklearn.multiclass import OneVsRestClassifier
OVR = LogisticRegression().fit(X,y)
print OVR.score(X,y)
q=pd.DataFrame(OVR.predict_proba(tX),columns=['high','low','medium'])
q['key']=range(1, len(q) + 1)
q['listing_id']=test['listing_id']
tX['key']=range(1, len(tX) + 1)
final=pd.merge(q,tX,on='key')

result=final[['listing_id','high','low','medium']]

# tX['probabilities']=OVR.predict_proba(tX)
#print OVR.score(tX,ty)
print OVR.coef_

import csv

# with open('viv.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(probabilities)
# result.to_csv('D:\\vivek\\rentHop\\output\\test.csv')
# tX.to_csv('D:\\vivek\\rentHop\\output\\tX.csv')
# q.to_csv('D:\\vivek\\rentHop\\output\\q.csv')
writer = pd.ExcelWriter('D:\\vivek\\rentHop\\output\\training.xlsx', engine='xlsxwriter')
raw.to_excel(writer,'sheet1')


