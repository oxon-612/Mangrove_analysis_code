#Predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from xgboost import plot_importance
from xgboost import XGBRegressor


dataset = pd.read_excel("M9.xlsx")
X = dataset.drop(['Year','Mangrove','City'], axis=1)
y = dataset['Mangrove']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
print("X_train:",X_train)
print("X_test:",X_test)
print("y_train:",y_train)
print("y_test:",y_test)
dataset2 = pd.read_excel("m9-ac+0.19.xlsx")
Xpre = dataset2.drop(['Year','City'], axis=1)
#Metrics
def rms(test,predict):
    return np.sqrt(np.mean(np.power((test - predict), 2)))

def mae(test,predict):
    return np.mean(np.power((np.abs(test - predict)),1))

########### RF
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
cpts = pd.DataFrame(pca.transform(X_train))
x_axis = np.arange(1, pca.n_components_+1)
pca_scaled = PCA()
pca_scaled.fit(X_train_scaled)
cpts_scaled = pd.DataFrame(pca.transform(X_train_scaled))


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

rf_y_pred = rf.predict(Xpre)

plt.figure()
plt.subplot(1,1,1)
plt.plot(rf_y_pred)

plt.show()

"""data1 = pd.DataFrame(rf_y_pred)
writer = pd.ExcelWriter('M9result-ac+0.19.xlsx')
data1.to_excel(writer,'Page1',float_format='%.5f')
writer.save()
writer.close()"""
