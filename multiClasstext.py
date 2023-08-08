# # Import necessary libraries
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, classification_report

# # Load the 20 newsgroups dataset
# newsgroups_train = fetch_20newsgroups(subset='train')
# print(newsgroups_train)
# newsgroups_test = fetch_20newsgroups(subset='test')
# print(newsgroups_test)
# # Define the text classification pipeline
# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('clf', MultinomialNB())
# ])

# # Train the classifier on the training set
# text_clf.fit(newsgroups_train.data, newsgroups_train.target)

# # Test the classifier on the test set
# predicted = text_clf.predict(newsgroups_test.data)

# # Evaluate the classifier's accuracy and other metrics
# print("Accuracy:", accuracy_score(newsgroups_test.target, predicted))
# print(classification_report(newsgroups_test.target, predicted))
# print(predicted)

# # import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split

# # load data into pandas dataframe
# df = pd.read_csv('text_classification_data.csv')

# # split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# # vectorize the text data
# vectorizer = CountVectorizer(stop_words='english')
# X_train_vect = vectorizer.fit_transform(X_train)
# X_test_vect = vectorizer.transform(X_test)

# # train the Multinomial Naive Bayes classifier
# clf = MultinomialNB()
# clf.fit(X_train_vect, y_train)

# # predict on the test data
# y_pred = clf.predict(X_test_vect)

# # evaluate the classifier
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
# print('Classification Report:\n', classification_report(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split

# load data from a CSV file
data = pd.read_csv("C:\\Users\\USER\\Downloads\\custom_email_dataset.csv")
print(data)

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(train_data)
print(test_data)

# extract the features and target variable from the training and testing data
X_train = train_data.drop('label', axis=1)
# print(X_train)
y_train = train_data['label']
# print(y_train.values)
X_test = test_data.drop('label', axis=1)
# print(X_test)
y_test = test_data['label']
# print(y_test.values)

# The given code loads a dataset from a CSV file and then splits it into training and testing sets using the train_test_split function from scikit-learn. The test_size parameter of this function specifies the proportion of the data that should be assigned to the test set, and in this case, it is set to 0.2, which means that 20% of the data is used for testing, while the remaining 80% is used for training.

# The code then extracts the features and target variable from the training and testing sets using the drop method of the Pandas DataFrame class. The axis parameter of this method is set to 1, which means that it drops the 'target' column along the column axis (i.e., axis=1), and returns a new DataFrame that contains all the columns except 'target'.

# Finally, the code prints the values of y_train and y_test using the .values attribute of the Pandas Series class. This returns an array of the actual values of the target variable, without the metadata that is displayed when you print the Series object directly.

