import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_patients = 500
data = {
    'Age': np.random.randint(30, 80, num_patients),
    'LengthOfStay': np.random.randint(1, 14, num_patients),
    'NumComorbidities': np.random.randint(0, 5, num_patients),
    'Readmitted': np.random.randint(0, 2, num_patients) # 0: No, 1: Yes
}
df = pd.DataFrame(data)
# Introduce some anomalies (simulating high-risk patients)
anomalies = np.random.choice(num_patients, size=int(num_patients * 0.1), replace=False)
df.loc[anomalies, 'LengthOfStay'] += np.random.randint(7, 21, size=len(anomalies))
df.loc[anomalies, 'NumComorbidities'] += np.random.randint(2, 5, size=len(anomalies))
df.loc[anomalies, 'Readmitted'] = 1
# --- 2. Anomaly Detection ---
# Use Isolation Forest for anomaly detection
model = IsolationForest(contamination='auto')
model.fit(df[['Age', 'LengthOfStay', 'NumComorbidities']])
df['AnomalyScore'] = model.decision_function(df[['Age', 'LengthOfStay', 'NumComorbidities']])
df['Anomaly'] = model.predict(df[['Age', 'LengthOfStay', 'NumComorbidities']])
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1}) # 0: Not an anomaly, 1: Anomaly
# --- 3. Analysis and Visualization ---
# Analyze the detected anomalies
anomaly_counts = df['Anomaly'].value_counts()
print("Number of anomalies detected:", anomaly_counts[1])
# Visualize the anomalies (scatter plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='LengthOfStay', y='NumComorbidities', hue='Anomaly', data=df, palette={0: 'blue', 1: 'red'})
plt.title('Anomaly Detection: Length of Stay vs. Number of Comorbidities')
plt.xlabel('Length of Stay (days)')
plt.ylabel('Number of Comorbidities')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'anomaly_detection_plot.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis (optional):  You could add more sophisticated analysis here, such as comparing the anomaly detection results with the actual 'Readmitted' status to evaluate the model's performance using metrics like precision, recall, F1-score etc.  This would involve creating a confusion matrix and calculating these metrics.