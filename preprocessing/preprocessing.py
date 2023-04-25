# TUGAS DATA MINING : MUHAMMAD FAIQ RAIHAN DHIYAULHAK - A11.2021.13833
# Kelas : A11.4610
# link gitHub : https://github.com/raihanfaiq72/DATA-MINING

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
x = np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x)
print(y)
print(x_train)
print(x_test)


# hitung entropy dan gain serta tentukan pohon keputusan yang terbentuk dari contoh kasus keputusan bermain tenis dibawah ini
# --------------------------------------------------------
# outlook     temprature      humadity    windy       play
# --------------------------------------------------------
# sunny       hot             high        no          Dont play
# sunny       hot             high        yes         Dont play
# cloudy      hot             high        no          play
# rainy       mild            high        no          play
# rainy       cool            normal      no          play
# rainy       cool            normal      yes         play
# cloudy      cool            normal      yes         play
# sunny       mild            high        no          Dont play
# sunny       cool            normal      no          play
# rainy       mild            normal      no          play
# sunny       mild            normal      yes         play
# cloudy      mild            high        yes         play
# cloudy      hot             normal      no          play
# rainy       mild            high        yes         Dont play


# --------------
# kerjakan latihan tahapan klasifikasi dengan decesion tree pada latihan sebelumnya , dataset bisa diganti kemudian simpan dalam decisiontree.py
