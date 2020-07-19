import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
from sklearn.model_selection import train_test_split
df1=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df1.drop(['worst radius','worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'],axis=1,inplace=True)
df1.drop(['radius error','texture error','perimeter error','area error','smoothness error','compactness error','concavity error','concave points error','symmetry error','fractal dimension error'],axis=1,inplace=True)
X=df1
y=cancer['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
A=df1
B=cancer['target']
A_train , A_test , B_train , B_test = train_test_split(A,B, test_size = 0.25 , random_state = 0)
sc = StandardScaler()
A_train = sc.fit_transform(A_train)
A_test = sc.fit_transform(A_test)
log = LogisticRegression(random_state = 0)
log.fit(A_train,B_train)
tree = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
tree.fit(A_train, B_train)
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
forest.fit(A_train, B_train)
from sklearn.model_selection import GridSearchCV
param={'C':[0.1,1,10,100,100],'gamma':[1,0.1,0.01,0.001,0.0001]}
model=GridSearchCV(SVC(),param,verbose=3)
model.fit(X_train,y_train)
model.best_params_
model.best_estimator_
import pickle
model.predict(X_test)
pickle.dump(model,open('modelsvm.pkl','wb'))
pickle.dump(tree,open('modeltree.pkl','wb'))
pickle.dump(forest,open('modelrf.pkl','wb'))