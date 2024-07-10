import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

file_path = 'absts.csv'  
df = pd.read_csv(file_path, header=None)
df.columns = ['language', 'abstract']

label_encoder = LabelEncoder()
df['language_encoded'] = label_encoder.fit_transform(df['language'])

X = df['abstract']
y = df['language_encoded']

print('------ Finish Prepared File ------')

pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

print('------ Finish Prepared Pipeline ------')

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')

print('------ Finish 10-Fold Cross-Validation ------')

mean_accuracy = cv_results.mean()

print(f'Mean Accuracy: {mean_accuracy}')
