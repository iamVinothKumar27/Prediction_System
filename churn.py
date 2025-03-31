import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Tenure', 'NumOfProducts'], axis=1)

# Encode Gender
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Prepare features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = ExtraTreesClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("Customer_Churn_NoNumProducts.pkl", "wb") as f:
    pickle.dump(model, f)
