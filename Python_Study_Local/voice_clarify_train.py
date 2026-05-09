# 本项目来源于和鲸社区，使用转载需要标注来源
# 作者: HandsomeLuoyang
# 来源: https://www.heywhale.com/mw/project/5fac97c88ca2cf0030cb200e
import pandas as pd
import xgboost
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r"D:\new_document\Document\voice.csv")
data['label'] = LabelEncoder().fit_transform(data['label'])

xgb = xgboost.XGBClassifier(
    base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bytree=1,
    gamma=0,
    learning_rate=0.1,
    max_delta_step=0,
    max_depth=3,
    min_child_weight=1,
    n_estimators=100,
    n_jobs=1,
    nthread=None,
    objective='binary:logistic',
    random_state=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    seed=None,
    subsample=1,
)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

xgb.fit(X_train, y_train)

pred = xgb.predict(X_valid)
print(f"F1 Score: {f1_score(y_valid, pred):.4f}")