import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# mencari dataset
dataset = pd.read_csv("dataset.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# encode data kategori (Atribut)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
x = ct.fit_transform(x)

# encode data kategori (class/label)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])

y = le.fit_transform(y)

# mengubah kolom pertama menjadi numerik
x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

# menghilangkan missing value (nan)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# membagi dataset kedalam training set dan test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 1:] = sc.fit_transform(x_train[:, 1:])
x_test[:, 1:] = sc.transform(x_test[:, 1:])

# keperluan print
print(x_train)
print(x_test)
print(y_train)
print(y_test)
print(x)
print(y)
