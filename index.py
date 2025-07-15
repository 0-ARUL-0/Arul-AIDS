import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Create spending class based on Spending Score
def spending_class(score):
    if score < 40:
        return 'Low'
    elif 40 <= score <= 70:
        return 'Average'
    else:
        return 'High'

df['Spending_Class'] = df['Spending Score (1-100)'].apply(spending_class)

# Step 3: Encode categorical data
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0
df['Spending_Class_Label'] = le.fit_transform(df['Spending_Class'])  # Low=0, Average=1, High=2

# Step 4: Define features (X) and target (y)
X = df[['Gender', 'Age', 'Annual Income (k$)']]
y = df['Spending_Class_Label']

# Step 5: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the decision tree model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 9: Visualize the decision tree
plt.figure(figsize=(16,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
plt.title("Customer Spending Classification Tree")
plt.show()
