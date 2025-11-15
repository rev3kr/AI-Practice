import numpy as np # 숫자 데이터 다루는 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # 선형 회귀 모델
from sklearn.metrics import r2_score # 회귀 모델 평가 (R squared)
import matplotlib.pyplot as plt # 시각화 도구

# 1. 데이터 준비
# X : 공부한 시간, y : 시험 점수
X = np.array([1, 2, 3, 4, 5, 6, 8, 10])
y = np.array([20, 30, 50, 55, 65, 70, 80, 95])

# sklearn 모델에 넣기 위해 X를 2차원 배열로 변환
# (모델은 여러 특징이 들어올 것을 가정하기 때문)
X = X.reshape(-1, 1)

print("입력 데이터 (X): ")
print(X)
print("\n정답 데이터 (y): ")
print(y)

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 모델 선택
model = LinearRegression()

# 4. 모델 훈련
model.fit(X_train, y_train)
print("\n모델 훈련 완료!")

# 5. 모델 평가
y_pred = model.predict(X_test)

# 회귀 모델 평가: R-squared (R2) 점수
# 1에 가까울 수록 모델이 데이터를 잘 설명한다
r2 = r2_score(y_test, y_pred)
print(f"\n모델 평가 점수 (R2): {r2:.4f}")

# 6. 새로운 데이터로 예측
new_study_time = np.array([[7]])
predicted_score = model.predict(new_study_time)
print(f"\n7시간 공부했을 때 예상 점수: {predicted_score[0]:.2f}점")

# 7. 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Actual Data') # 실제 데이터 산포도
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line') # 모델 예측 선
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.title("Study Hours vs. Test Score")
plt.legend()
plt.grid(True)
plt.show() # 그래프 보여줌