import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
import graphviz

st.title('Decision Trees - Hyper Parameter Tuning')
df = pd.read_csv('../heart_v2.csv')
y = df.pop('heart disease')
X = df

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42)
# st.write('### Shapes of Training & Test sets')
# st.write(X_train.shape, X_test.shape)

max_depth = st.sidebar.slider(
    'Max Depth', min_value=1, max_value=25, step=1, value=3)

max_leaf_nodes = st.sidebar.slider(
    'Max Leaves', min_value=2, max_value=100, step=1, value=100)

min_samples_split = st.sidebar.slider(
    'Min Samples Before Split', min_value=2, max_value=200, step=1, value=5)

min_samples_leaf = st.sidebar.slider(
    'Min Samples In Each Leaf', min_value=1, max_value=200, step=1, value=5)

criterion = st.sidebar.selectbox('Spliting Criterion', ['gini', 'entropy'])

# criterion = st.sidebar.selectbox('Spliting Criterion', ['gini', 'entropy'])


@st.cache
def classify(max_depth, max_leaf_nodes=None, min_samples_split=None, min_samples_leaf=None, criterion='gini'):
    dt = DecisionTreeClassifier(
        max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
    return dt.fit(X_train, y_train)


@st.cache
def get_dt_graph(dt_classifier):
    dot_data = StringIO()
    export_graphviz(dt_classifier, out_file=dot_data, filled=True, rounded=True,
                    feature_names=X.columns, class_names=['No Disease', 'Disease'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph

# model evaluation helper


def evaluate_model(dt_classifier):
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    st.write('### Train Set Performance')
    st.write('Accuracy : ', 100 *
             np.round(accuracy_score(y_train, y_train_pred), 3))
    st.write('#### Confusion Matrix')
    st.write(confusion_matrix(y_train, y_train_pred))
    st.write("-"*60)
    st.write('### Test Set Performance')
    st.write('Accuracy : ', 100*np.round(accuracy_score(y_test, y_test_pred), 3))
    st.write('#### Confusion Matrix')
    st.write(confusion_matrix(y_test, y_test_pred))


# st.write("-"*60)
dt = classify(max_depth, max_leaf_nodes, min_samples_split,
              min_samples_leaf, criterion)
graph = get_dt_graph(dt)
st.write('### Decision Tree')
st.image(graph.create_png(), width=800)
st.write("-"*60)
evaluate_model(dt)
