import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree #libraries and packages for training model and evaluating performance on test set
import seaborn as sns
import graphviz
#code to import funded.xlsx
df = pd.read_excel("/content/funded.xlsx")
#code to produce df heading
df.head()
#code to view class distribution of 'outcome' (hint: value counts)
df['outcome'].value_counts()
#code to compute apriori prediction accuracy
apriori = 417/len(df)
print('apriori accuracy =', apriori)
#select IVs to include in analysis, assign to object called 'x'
x = df.drop(['id','outcome'], axis=1)
# replace with code to dummy code categorical variables
x = pd.get_dummies(data = x, drop_first=False)
# assign your DV to an object called 'y'
y = df['outcome']
# create 4 dataframe objects x_train, x_test, y_train, y_test. Set test_size to 0.2, set random_state to 100
x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size = 0.2, random_state = 100)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# set parameters based on instructions for DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "gini", random_state= 100,
                              max_depth = 10, min_samples_leaf=8, min_impurity_decrease=0.002)
# fit model to x_train and y_train
model.fit(x_train, y_train)
#  generate tree plot of model
labels = y.value_counts()
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=x.columns,
                                class_names=labels.index.values,
                                filled=True) #plot model

# Draw graph
graph = graphviz.Source(dot_data, format="png")
graph
# predict onto test set using model
predictions = model.predict(x_test)
# create new dataframe from x_test, called df_pred
df_pred = x_test
# add predictions as a 'predicted_class' column to new dataframe
df_pred['predicted_class'] = predictions
# add y_test as a 'actual_class' column to new dataframe
df_pred['actual_class'] = y_test
df_pred
# fit confusion matrix to 'predicted_class' and 'actual_class'
conf = pd.DataFrame(df_pred, columns=['actual_class','predicted_class'])
confusion_matrix = pd.crosstab(conf['actual_class'], conf['predicted_class'], rownames=['Actual'], colnames=['Predicted'])
# visualize confusion matrix with a heatmap
sns.heatmap(confusion_matrix, annot=True, fmt='g')
sns.set(rc={'figure.figsize':(12,10)})
plt.show
#calculate overall accuracy from values in confusion matrix
acc = (52 + 17 + 46 + 17)/len(y_test)
print("overall accuracy = ", acc)
#specify DecisionTreeClassifier parameters
model_2 = DecisionTreeClassifier(criterion = "gini", random_state=100,
                               max_depth=12, min_samples_leaf=4, min_impurity_decrease=0.002)
#  fit model2 to training data
model_2.fit(x_train, y_train)
# 'predicted_class' and 'actual_class' columns from x_test
x_test = x_test.drop(columns = ['predicted_class','actual_class'])
#  fit model2 to x_test
predictions = model_2.predict(x_test)
#  print classification report
print(classification_report(y_test, predictions))
#specify DecisionTreeClassifier parameters
model_3 = DecisionTreeClassifier(criterion = "entropy", random_state=100, max_depth=5, min_samples_leaf=6, min_impurity_decrease=0.005)
#  fit model to training data
model_3.fit(x_train, y_train)
#  predict model3 onto x_test
predictions = model_3.predict(x_test)
#specify DecisionTreeClassifier parameters
model_4 = DecisionTreeClassifier(criterion = "log_loss", random_state=100, max_depth=15, min_samples_leaf=6,min_impurity_decrease=0.002)
#fit model to training data
model_4.fit(x_train, y_train)
# predict model4 onto x_test
predicitions = model_4.predict(x_test)
# plot tree of best performing model
labels = y.value_counts()
dot_data = tree.export_graphviz(model_3, out_file=None,
                                feature_names=x.columns,
                                class_names=labels.index.values,
                                filled=True)
graph2= graphviz.Source(dot_data, format="png"
)
graph2
