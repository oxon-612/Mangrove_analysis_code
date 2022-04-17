#The first
#xgboost
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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=40)
print("X_train:",X_train)
print("X_test:",X_test)
print("y_train:",y_train)
print("y_test:",y_test)


#评价指标
def rms(test,predict):
    return np.sqrt(np.mean(np.power((test - predict), 2)))

def mae(test,predict):
    return np.mean(np.power((np.abs(test - predict)),1))

###### XGBoost
# cross validation
folds = 10
#k_choices = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1]
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
        rf = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=k, colsample_bytree=0.1)
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        test_score_rms = rms(y_val, y_val_pred)
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

import matplotlib.pyplot as plt
# show the accuracy
for k in k_choices:
    plt.scatter([k] * len(mae_of_k[k]), mae_of_k[k])
rms_mean = np.array([np.mean(v) for k, v in sorted(mae_of_k.items())])
rms_std = np.array([np.std(v) for k, v in sorted(mae_of_k.items())])
plt.errorbar(k_choices, rms_mean, yerr=rms_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation mae')
plt.show()

xgb = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=500, colsample_bytree=0.1)
xgb.fit(X_train,y_train)
xgb_y_pred = xgb.predict(X_test)
s = xgb.score(X_test, y_test)
xgb_importances = xgb.feature_importances_

"""plot_importance(xgb)
plt.show()"""


