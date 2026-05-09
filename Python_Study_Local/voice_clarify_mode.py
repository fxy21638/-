# 模型保存代码（基于原代码扩展）
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 1. 加载数据并训练模型（同原代码）
data = pd.read_csv(r"D:\new_document\Document\voice\voice.csv")  # 改用用户上传的文件路径
data['label'] = LabelEncoder().fit_transform(data['label'])  # 0/1对应男/女（需记录映射关系：0=male？1=female？）
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

# 2. 训练模型（同原代码）
xgb_model = xgb.XGBClassifier(
    base_score=0.5, booster='gbtree', max_depth=3, 
    learning_rate=0.1, n_estimators=100, random_state=0
)
xgb_model.fit(X_train, y_train)

# 3. 保存模型和特征列表（关键：特征顺序必须与训练时一致）
joblib.dump(xgb_model, r"D:\new_document\Document\voice\voice_xgb_model.pkl")  # 保存模型
feature_names = X.columns.tolist()  # 保存特征列表（如['meanfreq', 'sd', 'median', ...]）
joblib.dump(feature_names, r"D:\new_document\Document\voice\voice_feature_names.pkl")

# 4. 记录标签映射（例如：0=male，1=female，需根据数据确认）
label_mapping = {0: "男性", 1: "女性"}
joblib.dump(label_mapping, r"D:\new_document\Document\voice\voice_label_mapping.pkl")

print("模型保存完成！特征列表：", feature_names[:5])  # 输出前5个特征确认