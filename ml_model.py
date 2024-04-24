import pandas as pd
import pickle

data = pd.read_csv('Social_Network_Ads.csv')

# Identify the target(y) and the feature(x) variables
x = data[['User ID','Age','EstimatedSalary']]
y = data['Purchased']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)

# Use Random Forest Classifiier Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=20, max_depth=20, min_samples_split=3, criterion='entropy')
model = rf.fit(x_train, y_train)

# Pickle the model
with open('model.pkl','wb') as pickle_model:
    pickle.dump(rf, pickle_model)