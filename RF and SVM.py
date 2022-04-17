#The second one
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

#Metrics
def rms(test,predict):
    return np.sqrt(np.mean(np.power((test - predict), 2)))

def mae(test,predict):
    return np.mean(np.power((np.abs(test - predict)),1))

########### RF
# cross validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
folds = 10
k_choices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]

X_folds = []
y_folds = []

X_folds = np.vsplit(X, folds)
y_folds = np.hsplit(y, folds)

rms_of_k = {}
mae_of_k = {}
for k in k_choices:
    rms_of_k[k] = []
    mae_of_k[k] = []
# split the train sets and validation sets
for i in range(folds):
    X_train = np.vstack(X_folds[:i] + X_folds[i + 1:])
    X_val = X_folds[i]
    y_train = np.hstack(y_folds[:i] + y_folds[i + 1:])
    y_val = y_folds[i]
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    for k in k_choices:
        rf = RandomForestRegressor(n_estimators=k, oob_score=True, random_state=0)
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        test_score_rms = rms(y_val,y_val_pred)
        rms_of_k[k].append(test_score_rms)
        test_score_mae = mae(y_val, y_val_pred)
        mae_of_k[k].append(test_score_mae)

for k in sorted(k_choices):
    for rmse in rms_of_k[k]:
        print('k = %d,rms = %f' % (k, test_score_rms))

for k in sorted(k_choices):
    for maee in mae_of_k[k]:
        print('k = %d,mae = %f' % (k, test_score_mae))

# show the plot
import matplotlib.pyplot as plt
# show the accuracy
for k in k_choices:
    plt.scatter([k] * len(rms_of_k[k]), rms_of_k[k])
rms_mean = np.array([np.mean(v) for k, v in sorted(rms_of_k.items())])
rms_std = np.array([np.std(v) for k, v in sorted(rms_of_k.items())])
plt.errorbar(k_choices, rms_mean, yerr=rms_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation rms')
plt.show()

for k in k_choices:
    plt.scatter([k] * len(mae_of_k[k]), mae_of_k[k])
mae_mean = np.array([np.mean(v) for k, v in sorted(mae_of_k.items())])
mae_std = np.array([np.std(v) for k, v in sorted(mae_of_k.items())])
plt.errorbar(k_choices, mae_mean, yerr=mae_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation mae')
plt.show()

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

predicted_train = rf.predict(X_train)
rf_y_pred = rf.predict(X_test)

test_score = r2_score(y_test, rf_y_pred)
spearman = spearmanr(y_test, rf_y_pred)
pearson = pearsonr(y_test, rf_y_pred)
rf_importances = rf.feature_importances_


####### SVM
clf = svm.SVR()
clf.fit(X_train, y_train)
svm_y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, mean_squared_error

xgb = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=500, colsample_bytree=0.1)
xgb.fit(X_train,y_train)
xgb_y_pred = xgb.predict(X_test)
s = xgb.score(X_test, y_test)
xgb_importances = xgb.feature_importances_
print(mean_squared_error(y_test, xgb_y_pred))



print("===================RESULT================================")
print('RF_rms:',rms(y_test,rf_y_pred),'; RF_mae:',mae(y_test,rf_y_pred))
print('SVM_rms:',rms(y_test,svm_y_pred),'; SVM_mae:',mae(y_test,svm_y_pred))
print('XGB_rms:',rms(y_test,xgb_y_pred),'; XGB_mae:',mae(y_test,xgb_y_pred))
print("=========================================================")
print("rf_importances:",rf_importances)
print("xgb_importances:",xgb_importances)
print("=========================================================")


plt.figure()
plt.subplot(3,1,1)
plt.plot(y_test.values)
plt.plot(rf_y_pred)

plt.subplot(3,1,2)
plt.plot(y_test.values)
plt.plot(svm_y_pred)

plt.subplot(3,1,3)
plt.plot(y_test.values)
plt.plot(xgb_y_pred)

plt.show()
"""
data1 = pd.DataFrame(y_test.values)
data2 = pd.DataFrame(rf_y_pred)
data3 = pd.DataFrame(svm_y_pred)
data4 = pd.DataFrame(xgb_y_pred)
writer = pd.ExcelWriter('M9newresult.xlsx')
data1.to_excel(writer,'Page1',float_format='%.5f')
data2.to_excel(writer,'Page2',float_format='%.5f')
data3.to_excel(writer,'Page3',float_format='%.5f')
data4.to_excel(writer,'Page4',float_format='%.5f')
writer.save()
writer.close()
print("y_test:",y_test.values)
print("y_pred:",rf_y_pred)"""