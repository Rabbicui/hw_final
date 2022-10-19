#!/usr/bin/env python
# coding: utf-8


from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score, precision_score,recall_score, roc_auc_score,roc_curve,auc

def cross_pre(data, test, use_feature):
    """
    5-folds cross-validation
    """
    model = XGBClassifier(colsample_bytree=0.6, max_depth=6, reg_alpha=0.05, subsample=0.6,
                          objective='binary:logistic', n_job=-1, booster='gbtree', n_estimators=1000,
                          learning_rate=0.02,gamma=0.7,random_state=2022,enable_categorical=True)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    predictions = 0
    flag = 0
    train_X = data[use_feature]
    train_y = data['label']

    train_X.index = range(len(train_X))
    train_y.index = range(len(train_y))
    test_x = test[use_feature]
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_y)):
        print('Fold:', fold_+1)
        tr_x, tr_y = train_X.iloc[trn_idx, :], train_y[trn_idx]
        vl_x, vl_y = train_X.iloc[val_idx, :], train_y[val_idx]

        model.fit(tr_x, tr_y,
            eval_set=[(vl_x, vl_y)],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50)
        y_prob = model.predict_proba(vl_x)[:,1]
        if flag ==0:
            y_pre = model.predict_proba(test_x)[:,1]
        else:
            y_pre += model.predict_proba(test_x)[:,1]
        print(roc_auc_score(vl_y, y_prob))
        predictions += roc_auc_score(vl_y, y_prob)/5
        flag += 1
    print("最终模型auc:", predictions)
    print(model.feature_importances_)
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()
    return y_pre/5
