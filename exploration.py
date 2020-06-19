# import pkgs
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# 1) explore the dataset
iris = datasets.load_iris() #loaded as numpy.ndarray
X = iris.data
Y = iris.target

# print("X=", X)
# print("Y=", Y)

# 2) input data processing
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
print("dataset=", df)

# 3) fitting
clf = RandomForestClassifier()
clf.fit(X, Y)

print(clf)

# 4) preditcting
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

print("prediction=", prediction)
print("prediction_proba=", prediction_proba)
