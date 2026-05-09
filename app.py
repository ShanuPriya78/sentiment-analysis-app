import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_excel("train.xlsx")
df.columns = df.columns.str.strip().str.lower()

print(df.columns)
print(df['sentiment'].value_counts())


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['reviews'] = df['reviews'].apply(clean_text)


X = df['reviews']
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


word_vec = TfidfVectorizer(
    ngram_range=(1,2)   
)

char_vec = TfidfVectorizer(
    analyzer='char_wb', 
    ngram_range=(3,5)
)

vectorizer = FeatureUnion([
    ("word", word_vec),
    ("char", char_vec)
])

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(class_weight='balanced', max_iter=200)
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))


while True:
    user_input = input("Enter a sentence (type 'exit' to stop): ")

    if user_input.lower() == 'exit':
        print("Program ended.")
        break


    cleaned_input = clean_text(user_input)

    user_vec = vectorizer.transform([cleaned_input])
    prediction = model.predict(user_vec)

    print("Raw prediction:", prediction[0])  

    if prediction[0] == 'pos':
        print("Positive 😊")
    else:
        print("Negative 😡")