from transformers import T5ForConditionalGeneration, AutoTokenizer

# 저장된 경로에서 모델과 토크나이저 불러오기
model_path = "./New_Sample/finetuned_kot5_with_relationship"

# 모델과 토크나이저 로드
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model
# 테스트용 입력
input_text = """
generate response:
1 : 언니 명랑 핫도그 먹어 봤어?
2 : 웅 먹은 지 한참 됐어
1 : 나 오늘 먹었는데 진짜 존맛 ㅜ
2 : 나도 오랜만에 먹고 싶다
"""
"""
관계 정보:
화자 1번: 여성, 20대
화자 2번: 여성, 20대
"""

# 입력 텍스트를 토크나이징
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 모델로 텍스트 생성 (추론)
outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

'''
outputs = model.generate(
        input_ids,
        max_new_tokens=200,  # 출력될 새 토큰 수를 200으로 제한
        num_beams=5,
        no_repeat_ngram_size=2,  # 반복 방지 설정
        early_stopping=True,
        temperature=0.7,  # 텍스트의 다양성 조절
        top_p=0.9,  # 샘플링 범위 제한
        top_k=50  # 상위 k개의 후보 중 선택
    )
'''

# 생성된 텍스트 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 결과 출력
print("생성된 텍스트:", generated_text)


print("Output IDs:", outputs)
