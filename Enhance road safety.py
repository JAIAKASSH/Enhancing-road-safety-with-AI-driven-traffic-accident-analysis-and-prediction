# Enhancing Road Safety with AI-Driven Traffic Accident Analysis and Prediction

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Simulate Dataset (Replace with real data if available)
np.random.seed(42)
data = {
    'Hour': np.random.choice(range(24), 1000),
    'Weather': np.random.choice(['Clear', 'Rainy', 'Foggy', 'Snowy'], 1000, p=[0.6, 0.2, 0.1, 0.1]),
    'Visibility': np.random.normal(10, 2, 1000),
    'Traffic_Density': np.random.randint(1, 5, 1000),
    'Accident_Severity': np.random.choice(['Low', 'Medium', 'High'], 1000, p=[0.5, 0.3, 0.2])
}
df = pd.DataFrame(data)

# 2. Preprocessing
df['Weather_Code'] = df['Weather'].map({'Clear': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3})
df['Severity_Code'] = df['Accident_Severity'].map({'Low': 0, 'Medium': 1, 'High': 2})

features = ['Hour', 'Weather_Code', 'Visibility', 'Traffic_Density']
target = 'Severity_Code'

X = df[features]
y = df[target]

# 3. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. AI Risk Prediction
df['AI_Predicted_Risk'] = model.predict_proba(X)[:, 2]  # Probability of 'High' severity

# 6. Visualization (save as images for reports)

# Phase 1: Accident Distribution by Hour
plt.figure(figsize=(10, 5))
sns.barplot(x=df['Hour'].value_counts().sort_index().index,
            y=df['Hour'].value_counts().sort_index().values,
            color='skyblue')
plt.title('Phase 1: Accident Distribution by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Accident Count')
plt.tight_layout()
plt.savefig('phase1_accident_distribution.png')
plt.close()

# Phase 2: Accident Severity vs Weather Conditions
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Weather', hue='Accident_Severity', palette='Set2')
plt.title('Phase 2: Accident Severity vs Weather Conditions')
plt.xlabel('Weather Condition')
plt.ylabel('Accident Count')
plt.legend(title='Severity')
plt.tight_layout()
plt.savefig('phase2_severity_weather.png')
plt.close()

# Phase 3: AI-Predicted Risk Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['AI_Predicted_Risk'], bins=20, kde=True, color='salmon')
plt.title('Phase 3: AI-Predicted Accident Risk Distribution')
plt.xlabel('Predicted Risk Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('phase3_risk_distribution.png')
plt.close()
