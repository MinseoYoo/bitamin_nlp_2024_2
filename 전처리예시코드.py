# @title 데이터 변환
import json
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset


# 원본 JSON 파일을 불러옵니다
with open('./New_Sample/라벨링데이터/TL_02. FACEBOOK/FACEBOOK_101_01.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 변환된 데이터를 저장할 리스트
converted_data = []

# 데이터 변환 로직
for item in data['info']:
    conversation = []  
    speaker_relationships = {} 

    # 각 라인 별로 처리
    for line in item['annotations']['lines']:
        speaker_id = line['speaker']['id']

        # 화자별 관계를 저장
        if speaker_id not in speaker_relationships:
            speaker_relationships[speaker_id] = {
                'sex': line['speaker']['sex'],
                'age': line['speaker']['age']
            }

        # 대화 내용을 누적하여 저장
        conversation.append(line['text'])

        # 'speechAct'가 '질문하기'에 해당하는 경우
        if "(지시) 질문하기" in line['speechAct']:
            # 현재 질문 발화는 target으로 설정
            target = line['text']
            
            # conversation이 공백이 아닌 경우에만 추가
            if conversation[:-1]:  
                converted_data.append({
                    'input': {
                        'conversation': conversation[:-1].copy(),  
                        'relationship': speaker_relationships
                    },
                    'target': target 
                })

# 변환된 JSON 파일을 저장합니다
with open('./New_Sample/sample.json', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print("변환 완료!")

# 다시 할때는 여기서부터
import json

with open('./New_Sample/sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
def preprocess_data_with_relationship(data):
    input_texts = []
    target_texts = []

    for item in data:
        # 대화를 한 줄로 연결하여 conversation으로 넣음
        conversation_text = "\n".join(item['input']['conversation'])
        
        # 화자 관계 정보를 추가
        relationship_text = "\n".join([f"화자 {k}: {v['sex']}, {v['age']}" for k, v in item['input']['relationship'].items()])
        
        # 관계 정보를 포함한 최종 입력 텍스트
        input_text = f"{conversation_text}\n\n관계 정보:\n{relationship_text}"
        target_text = item['target']

        input_texts.append(input_text)
        target_texts.append(target_text)
    
    return input_texts, target_texts

# 전처리된 데이터 불러오기
input_texts, target_texts = preprocess_data_with_relationship(data)

# 각 input_text와 target_text 쌍 출력
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    print(f"Pair {i + 1}:")
    print("Input Text:")
    print(input_text)
    print("Target Text:")
    print(target_text)
    print("-" * 50)


model_name = "paust/pko-t5-base"  # KoT5 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)



dataset = Dataset.from_dict({"input_text": input_texts, "target_text": target_texts})

# 데이터 토큰화
def tokenize_data(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_data, batched=True)



# 학습 설정
training_args = TrainingArguments(
    output_dir="./New_Sample/results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.0001,
    logging_dir='./New_Sample/logs',
    logging_steps=10,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# 학습 시작
trainer.train()

# 학습 후 모델 저장
model.save_pretrained("./New_Sample/finetuned_kot5_with_relationship")
tokenizer.save_pretrained("./New_Sample/finetuned_kot5_with_relationship")

