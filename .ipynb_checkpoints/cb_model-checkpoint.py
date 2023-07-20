#%pip install catboost
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

modeling1 = pd.read_csv('data/catboost_multi.csv')

y = modeling1['Error_class']
X = modeling1.drop(['Error_class'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

cmodel = CatBoostClassifier()
cmodel.fit(X_train, y_train)

#Train/test Score
train_score = cmodel.score(X_train, y_train)
test_score = cmodel.score(X_test, y_test)

#y_pred
y_pred = cmodel.predict(X_test)

#report
report = classification_report(y_test, y_pred)

#importance
importance = cmodel.feature_importances_

#save --> model폴더에 저장하고 싶어 --> load --> predict excercise
savemodel = cmodel.save_model('model/model.dump')