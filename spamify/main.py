import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reading the data into a pandas dataframe
df = pd.read_csv('mail_data.csv')

data = df.where((pd.notnull(df)), '')

# Converting the category column to integers
data.loc[data['category'] == 'spam', 'category',] = 0
data.loc[data['category'] == 'ham', 'category',] = 1

# Splitting the data into training and testing sets
X = data['message']
Y = data['category']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(
    min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Converting the labels to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Creating a logistic regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Predicting on training set
prediction_on_training_model = model.predict(X_train_features)
accuracy_on_training_model = accuracy_score(
    Y_train, prediction_on_training_model)

print('Accuracy on training set: {:.2f}'.format(accuracy_on_training_model))

# Predicting on testing set
prediction_on_testing_model = model.predict(X_test_features)
accuracy_on_testing_model = accuracy_score(
    Y_test, prediction_on_testing_model)

print('Accuracy on testing set: {:.2f}'.format(accuracy_on_testing_model))

# Add your own mail here & run the code to see if it is a spam mail or not!
input_your_mail = ['']  # Please enter your mails here
input_data_features = feature_extraction.transform(input_your_mail)

prediction = model.predict(input_data_features)

if (prediction[0] == 1):
    print('This mail is not a spam mail')
else:
    print('This mail is a spam mail')
