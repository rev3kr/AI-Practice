from transformers import pipeline

# 1. 모델 준비
print("감성 분석 모델을 로드합니다...")
classifier = pipeline("sentiment-analysis")
print("모델 로드 완료!")

# 2. 데이터 준비
text1 = "This movie was absolutely amazing! I loved it."
text2 = "I'm not sure I like this product. It's quite confusing."

# 3. 모델 사용 (훈련 불필요)
results1 = classifier(text1)
results2 = classifier(text2)

# 4. 결과 확인
print(f"\n문장: '{text1}'")
print(f"결과: {results1}")

print(f"\n문장: '{text2}'")
print(f"결과: {results2}")