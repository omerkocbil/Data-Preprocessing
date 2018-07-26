
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer   #machine learning kutuphanesi
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #biz ortalama deger stratejisini kullandik.
imputer = imputer.fit(X[:,1:3])    # nan olan verilerin yerine  

X[:, 1:3] = imputer.transform(X[:, 1:3]) # nan olan yerlerdeki degerleri kurala göre degistirdi ve tekrar aynı yere transform etti.
#X[:, 1:3] = imputer.fit_transform(X[:,1:3]) ayni sekilde direk degeri veriyor-kisa method-


#**********************************************

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#burda ulke adlarina rakam vererek kategoriledik

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#burda ise rakamlarin ulkelerin buyuklukleriyle alakasi olmadigini saglamak icin 2lik sistemde sayi degeri gibi deger almalarini sagladik




labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#burda son sutun olan yes-no sutununa rakam degerleri vererek kategoriledik.




# splitting dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #random state = 1 olursa random deger atar aksi halde ayni secim yapar



#feature scalling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
