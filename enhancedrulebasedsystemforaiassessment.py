import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/eithanenaje/Downloads/Sleep_health_and_lifestyle_dataset.csv")

def predict_sleep_disorder_enhanced(row):

    if (row['BMI Category'] in ['Obese', 'Overweight'] and row['Daily Steps'] < 6000) \
        or (row['Blood Pressure'] in ['High', 'Elevated'] and row['Heart Rate'] >= 75) \
        or (row['Sleep Duration'] >= 8 and row['Quality of Sleep'] <= 6):
        return 'Sleep Apnea'
   
    if (row['Sleep Duration'] < 6.5 and row['Quality of Sleep'] <= 6) \
        or (row['Stress Level'] >= 7) \
        or (row['Heart Rate'] >= 80 and row['Blood Pressure'] in ['Normal', 'Elevated']):
        return 'Insomnia'
    

    if (6.5 <= row['Sleep Duration'] <= 8) and \
       (row['Quality of Sleep'] >= 7) and \
       (row['Daily Steps'] >= 6000) and \
       (row['Stress Level'] <= 6) and \
       (row['Heart Rate'] < 75) and \
       (row['Blood Pressure'] in ['Normal', 'Low']):
        return 'None'
    
    return 'None'


df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
df['Predicted_Enhanced'] = df.apply(predict_sleep_disorder_enhanced, axis=1)
acc = (df['Predicted_Enhanced'] == df['Sleep Disorder']).mean()
print(f"Enhanced Rule-Based System v1 Accuracy: {acc:.2%}")
print(df['Predicted_Enhanced'].value_counts())
print(df[['Sleep Disorder', 'Predicted_Enhanced']].head(20))


actual_counts = df['Sleep Disorder'].value_counts()
predicted_counts = df['Predicted_Enhanced'].value_counts()


comparison = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts}).fillna(0)


comparison.plot(kind='bar', figsize=(8,5))
plt.title('System 2: Actual vs Predicted Sleep Disorder Counts')
plt.xlabel('Sleep Disorder Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
