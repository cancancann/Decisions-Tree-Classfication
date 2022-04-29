# coding: utf-8

# # Decison Tree Classification

# İlk olarak verilerimizi yüklüyoruz ve kütüphaneleirmizi import ediyoruz.


import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")



print(df.head())


# 
# scikit-learn kütüphanesi decision tree'lerin düzgün çalışması için herşeyin rakamsal olmasını bekliyor bu nedenle veri setimizdeki tüm Y ve N değerlerini 0 ve 1 olarak düzeltiyoruz. 
# Aynı sebeple eğitim seviyesini de BS:0 MS:1 ve PhD:2 olarak güncelliyoruz. map() kullanarak boş hücreler veya geçersiz değer girilen hücreler NaN ile doldurulacaktır, 
# buna şuandaki veri setimizde ihtiyacımız yok ama sizin ilerde yoğun veri ile çalıştığınız zaman ihtiyacınız olacaktır.
# 


duzetme_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)
duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)
result = df.head()
print(result)

# Sonuc sütununu ayırıyoruz:


y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)

xresult = X.head()
print(xresult)


# Decision Tree'mizi oluşturuyoruz:

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# Prediction yapalım şimdi
# 5 yıl deneyimli, hazlihazırda bir yerde çalışan ve 3 eski şirkette çalışmış olan, eğitim seviyesi Lisans
# top-tier-school mezunu değil
print (clf.predict([[5, 1, 3, 0, 0, 0]]))


# Toplam 2 yıllık iş deneyimi, 7 kez iş değiştirmiş çok iyi bir okul mezunu şuan çalışmıyor
print (clf.predict([[2, 0, 7, 0, 1, 0]]))


# Toplam 2 yıllık iş deneyimi, 7 kez iş değiştirmiş çok iyi bir okul mezunu değil şuan çalışıyor
print (clf.predict([[2, 1, 7, 0, 0, 0]]))


# Toplam 20 yıllık iş deneyimi, 5 kez iş değiştirmiş iyi bir okul mezunu şuan çalışmıyor
print (clf.predict([[20, 0, 5, 1, 1, 1]]))


# ## Toplu Öğrenme: Random Forest

# 20 tane decision tree birleşiminden oluşan bir Random Forest kullanarak tahmin yapacağız:
# 


from sklearn.ensemble import RandomForestClassifier


rnd_fr_clf = RandomForestClassifier(n_estimators=20)
rnd_fr_clf = rnd_fr_clf.fit(X, y)

#Predict employment of an employed 10-year veteran
print (rnd_fr_clf.predict([[10, 1, 4, 0, 0, 0]]))

#...and an unemployed 10-year veteran
print (rnd_fr_clf.predict([[10, 0, 4, 0, 0, 0]]))







