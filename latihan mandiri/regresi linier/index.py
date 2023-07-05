# import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# memanggil dataset
dataset=pd.read_csv('populasi.csv')
x=dataset.iloc[:,:1]
y=dataset.iloc[:,:1].values
dataku=pd.DataFrame(dataset)

# memanggil data yang akan diprediksi
dataUji=pd.read_csv('prediksitahun.csv')
X_uji=dataUji.iloc[:,:1]

# berikutnya adalah proses membagi data populasi menjadi data pelatihan dan data tes untuk validasi . 
# membagi/split data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.1,random_state=0)
# print(X_test)
# print(X_uji)
# pada proses pengujian diatas didapatkan jika pada tahun 2018 sementara yang akan diuji adalah dari 2021 hingga 2036

# proses pemodelan
# untuk memprediksi populasi 6 tahun kedepan kita akan menggunakan algoritma regresi linier 
# y = a+bx
# y adalah variable dependent (variable terikat)
# x adalah variable independent (variable bebas)

# fitting pada data training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# memprediksi hasil dengan data test 
Y_pred = regressor.predict(X_uji)
print(Y_pred)


# disimpan 
tahunuji=np.array(X_uji)
tahun=pd.DataFrame(tahunuji)
prediksi=pd.DataFrame(Y_pred)
hasil=pd.concat([tahun,prediksi],axis=1)
np.savetxt('hasil.csv',hasil,delimiter=',')
print(hasil)

# membuat grafik 

# visualisasi data 
# menggabungkan data training dan data hasil
absis=(np.concatenate([X_train,X_uji]))
ordinat=(np.concatenate([Y_train,Y_pred]))
plt.scatter(absis, ordinat , color='blue' )
plt.scatter(tahun, prediksi , color='blue' )
plt.xlabel("Tahun")
plt.ylabel("Populasi")
plt.title("Grafik Hasil Prediksi Populasi Pada Tahun Ke ")
plt.grid
plt.show()
