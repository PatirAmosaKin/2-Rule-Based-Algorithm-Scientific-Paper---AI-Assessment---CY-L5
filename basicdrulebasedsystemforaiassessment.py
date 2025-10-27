import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/eithanenaje/Downloads/Sleep_health_and_lifestyle_dataset.csv")


def predict_sleep_disorder_basic(row):
    if row['Daily Steps'] < 5500 and row['Sleep Duration'] >= 7:
        return 'Sleep Apnea'
    if row['Sleep Duration'] < 6.4 or row['Stress Level'] >= 6:
        return 'Insomnia'
    if 6.4 <= row['Sleep Duration'] <= 8 and row['Daily Steps'] >= 6000 and row['Stress Level'] <= 5:
        return 'None'
    return 'None'


df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
df['Predicted_Basic'] = df.apply(predict_sleep_disorder_basic, axis=1)


actual_counts = df['Sleep Disorder'].value_counts()
predicted_counts = df['Predicted_Basic'].value_counts()



comparison = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts}).fillna(0)


comparison.plot(kind='bar', figsize=(8, 5))
plt.title('System 1: Actual vs Predicted Sleep Disorder Distribution')
plt.xlabel('Sleep Disorder Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['Actual', 'Predicted'])
plt.tight_layout()
plt.show()


acc = (df['Predicted_Basic'] == df['Sleep Disorder']).mean()
print(f"3-Factor System Accuracy: {acc:.2%}")


print(df[['Sleep Duration', 'Daily Steps', 'Stress Level', 'Sleep Disorder', 'Predicted_Basic']].head(20))




