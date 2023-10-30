import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('/Users/wogns/refined_box_data_0912.csv', names=['image_width', 'image_height', 'box_count', 'box_size_avg', 'box_dist_avg', 'box_normalized_count', 'box_normalized_count_above_average', 'box_normalized_count_below_average', 'box_normalized_size_avg', 'box_normalized_dist_avg', 'normalization_warp_ratio', 'box_normalized_plus_dist_avg', 'd_box_norm_above_size', 'd_box_norm_below_size', 'd_box_normalized_points_far_distance', 'box_most_large_x', 'box_most_large_y', 'box_most_small_x', 'box_most_small_y', 'risk_level'])
#print(data.shape)

# count 500 이상 삭제
df = pd.DataFrame(data)
df = df[df['box_count'] < 1500]

# box_count 0값 삭제
df = pd.DataFrame(df)
variable_to_check = 'box_count'
df = df[df[variable_to_check] != 0]
df.reset_index(drop=True, inplace=True)

# 999999 값 삭제
column_to_check = 'box_dist_avg'  # 원하는 열 선택
value_to_remove = 999999
df = df[df[column_to_check] != value_to_remove]
#print(df.describe())

# 다중공선성 이후 피쳐값 삭제후 최종 상관관계
import seaborn as sns
import matplotlib.pyplot as plt

names = [ 'image_width', 'image_height', 'box_count', 'box_size_avg', 'box_dist_avg', 'box_normalized_count', 'box_normalized_size_avg', 'box_normalized_dist_avg', 'normalization_warp_ratio', 'box_normalized_plus_dist_avg', 'd_box_norm_above_size', 'd_box_norm_below_size', 'd_box_normalized_points_far_distance', 'box_most_large_x','risk_level']
cm = np.corrcoef(df[names].values.T)
sns.set(font_scale=0.8)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size':10},
yticklabels=names, xticklabels=names, cmap=plt.cm.Blues)
#plt.savefig('./data_1.png', dpi=300)

X = df[[ 'image_width', 'image_height', 'box_count', 'box_size_avg', 'box_dist_avg', 'box_normalized_count', 'box_normalized_size_avg', 'box_normalized_dist_avg', 'normalization_warp_ratio', 'box_normalized_plus_dist_avg', 'd_box_norm_above_size', 'd_box_norm_below_size', 'box_most_large_x']]
y = df['risk_level']

from statsmodels.stats.outliers_influence import variance_inflation_factor

Xs = sm.add_constant(X)

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])]
vif['predictor'] = Xs.columns
#print(vif)

from sklearn.model_selection import train_test_split
X = df[[ 'image_width', 'image_height', 'box_count', 'box_size_avg', 'box_dist_avg', 'box_normalized_count', 'box_normalized_size_avg', 'box_normalized_dist_avg', 'normalization_warp_ratio', 'box_normalized_plus_dist_avg', 'd_box_norm_above_size', 'd_box_norm_below_size', 'box_most_large_x']]
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)

# SVR이 SVC의 회귀 버전
from sklearn.svm import SVR

# SVR 모델 생성
model = SVR(kernel='rbf', C = 2000, epsilon=0.5)
model.fit(X_train, y_train)

# SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test) # 예측 라벨

# 성능 평가
mse = mean_squared_error(y_test, y_pred)  # MSE 계산
mae = mean_absolute_error(y_test, y_pred)  # MAE 계산
r2 = r2_score(y_test, y_pred)  # R-squared 계산

# 결과 출력
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# 시각화
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("test")
plt.ylabel("pred")
plt.title("SVR")
#plt.show()

comparison = pd.DataFrame({'actual': y_test, 'pred':y_pred})
print(comparison)

import joblib
joblib.dump(model, './sklearn_svr_model.pkl')