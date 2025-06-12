# mood_vs_music.py

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
data = pd.read_csv("D:/GuviProject/data.csv")

# Sample dataset in case of no data
df = pd.DataFrame({
    'valence': np.random.rand(500),
    'energy': np.random.rand(500),
    'tempo': np.random.randint(60, 180, 500)
})

# 2. Define Mood Labels Based on Rules
def classify_mood(valence, energy, tempo):
    if valence > 0.6 and energy > 0.6:
        return 'Happy'
    elif valence < 0.4 and energy < 0.4:
        return 'Sad'
    elif valence >= 0.4 and valence <= 0.6 and energy < 0.5:
        return 'Calm'
    else:
        return 'Energetic'

df['mood'] = df.apply(lambda row: classify_mood(row['valence'], row['energy'], row['tempo']), axis=1)

# 3. Visualize Mood Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='mood', palette='Set2')
plt.title('Distribution of Moods')
plt.show()

# 4. Prepare Data for ML
X = df[['valence', 'energy', 'tempo']]
y = df['mood']

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Train Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predictions and Evaluation
y_pred = model.predict(X_test)

# Convert back to labels
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Evaluation
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred_labels)
sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Mood Classification Confusion Matrix")
plt.show()

# 8. Predict Mood for New Song
def predict_mood(valence, energy, tempo):
    sample = pd.DataFrame([[valence, energy, tempo]], columns=['valence', 'energy', 'tempo'])
    prediction = model.predict(sample)
    return le.inverse_transform(prediction)[0]

# Example usage
print("\nExample Prediction:")
example_mood = predict_mood(0.75, 0.8, 120)
print(f"The mood for valence=0.75, energy=0.8, tempo=120 is: {example_mood}")
