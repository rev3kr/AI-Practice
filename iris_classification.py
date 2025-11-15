from sklearn.datasets import load_iris # 데이터셋
from sklearn.model_selection import train_test_split # 데이터 분할 도구
from sklearn.tree import DecisionTreeClassifier # 의사결정트리
from sklearn.metrics import accuracy_score # 정확도 평가 도구

# 1. 데이터 수집 및 준비
# scikit-learn에 내장된 붓꽃 데이터 로드
iris = load_iris()
x = iris.data   # 특징 (꽃의 정보)
y = iris.target # 정답 (꽃의 품종)

print("데이터 특징 (처음 5개): ")
print(x[:5])
print("\n데이터 정답 (처음 5개): ")
print(y[:5])

# 3. 데이터 분할
# 데이터를 훈련용(80%), 테스트용(20%)으로 나눔
# random_state=42 : 항상 같은 방식으로 섞기 위해 설정
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\n훈련 데이터 개수: {len(x_train)}개")
print(f"테스트 데이터 개수: {len(x_test)}개")

# 4. 모델 선택
model = DecisionTreeClassifier()

# 5. 모델 훈련
model.fit(x_train, y_train) # 모델에 훈련 데이터 학습
print("\n모델 훈련 완료!")

# 6. 모델 평가
y_pred = model.predict(x_test) # 테스트 데이터 정답 예측

accuracy = accuracy_score(y_test, y_pred) # 실제 정답과 모델 예측을 비교하여 정확도 계산
print(f"\n모델 정확도: {accuracy * 100:.2f}%")

# 7. 새로운 데이터로 예측
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)

predicted_species = iris.target_names[prediction][0]
print(f"\n새로운 꽃 예측 품종: {predicted_species}")
