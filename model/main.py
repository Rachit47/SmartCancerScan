import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_and_evaluate_model(data):
    # Split the features and target variable
    features = data.drop(['diagnosis'], axis=1)
    target = data['diagnosis']
    
    # Normalize the feature set
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))
    
    return model, scaler
    
def load_and_prepare_data():
    data = pd.read_csv("static/data.csv")
    
    # Remove unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Map diagnosis labels to numerical values
    data['diagnosis'] = data['diagnosis'].map({'M' : 1, 'B': 0})
    
    return data

def main():
    data = load_and_prepare_data()
    
    # Train the model and get the scaler
    model, scaler = train_and_evaluate_model(data)
    
    # Save the trained model and scaler
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return data
    
if __name__ == '__main__':
    main()
