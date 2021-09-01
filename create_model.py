import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('heart.csv')

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=True)


#random forest model

randforest = RandomForestClassifier(max_depth=10)
randforest = randforest.fit(X_train, y_train)
y_predict = randforest.predict(X_test)

print("accuracy of model to predict target:")
print(accuracy_score(y_test, y_predict))

filename = 'Random_Forest_model.sav'
pickle.dump(randforest, open(filename, 'wb'))
