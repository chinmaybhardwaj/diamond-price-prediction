import pandas as pd
import sklearn
from sklearn import svm, preprocessing

df = pd.read_csv('./dataset/diamonds.csv', index_col=0)


# Convert String data into codes
#df['cut'].astype('category').cat.codes


cut_class_dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
clarity_dict = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10, "FL": 11}
color_dict = {"J": 1,"I": 2,"H": 3,"G": 4,"F": 5,"E": 6,"D": 7}

df['cut'] = df['cut'].map(cut_class_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)
print(df.head())

df = sklearn.utils.shuffle(df)

X = df.drop('price', axis=1).values
y = df['price'].values

X = preprocessing.scale(X)

test_size = 200

train_X = X[:-test_size]
train_y = y[:-test_size]

test_X = X[-test_size:]
test_y = y[-test_size:]


clf = svm.SVR(kernel='linear')
clf.fit(train_X, train_y)
score = clf.score(test_X, test_y)
print('Score (linear):', score)


clf = svm.SVR(kernel='rbf')
clf.fit(train_X, train_y)
score = clf.score(test_X, test_y)
print('Score (rbf):', score)

