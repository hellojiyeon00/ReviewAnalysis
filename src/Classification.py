from Preprocessing import preprocess_dataframe
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import torch
import os

def Classification(path, model_name):
    # 데이터셋 로드
    df = pd.read_csv(path, encoding="utf-8-sig")

    df.rename(columns={'review':'text'}, inplace=True)

    print("- 데이터셋을 로드했습니다.")

    # 데이터셋 전처리
    df = preprocess_dataframe(df)
    
    # star의 결측치를 해당 열의 평균값으로 채움
    df['star'] = df['star'].fillna(df['star'].mean())
    
    # 나머지 결측치는 삭제
    df.dropna(inplace=True)

    df.rename(columns={'text':'review'}, inplace=True)
    
    # review 텍스트의 줄바꿈 문자를 공백으로 변환
    df['review'] = df['review'].map(lambda x: x.replace("\n", " "))

    print("- 텍스트 전처리가 완료되었습니다.")

    # tokenizer 및 모델 초기화

    # Fine-tuned된 Re_ELECTRA_Finetuning 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(f"./model/{model_name}/finetuned")
    # num_labels=2 : 이진 분류
    model = AutoModelForSequenceClassification.from_pretrained(f"./model/{model_name}/finetuned", num_labels=2)

    print("- 토크나이저 및 모델이 초기화 되었습니다.")

    # 감성 분류
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"- {device.upper()} 사용하여 감성 분류를 시작합니다.")

    model.to(device)
    model.eval()

    # 토크나이저 최대 길이 제한
    tokenizer.model_max_length = 128

    # 배치 사이즈 조정(GPU 메모리에 맞게 설정 / 8, 16, 32 등)
    batch_size = 16   
    all_predict = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_reviews = df['review'].iloc[i:i+batch_size].tolist()
        encoding = tokenizer(batch_reviews, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**encoding).logits
            predict = logits.argmax(dim=-1)

        all_predict.extend(predict.cpu().numpy())

    df['label'] = all_predict
    df = df[['review', 'label', 'star']]

    print("- 리뷰 데이터셋 감성 분류가 완료되었습니다.")

    # 라벨링한 리뷰 데이터 파일 저장

    # 파일을 저장할 폴더 경로
    dir_path = f"./data/processed/review/{model_name}"

    # 해당 폴더가 없으면
    if not os.path.exists(dir_path):
        # 폴더 생성
        os.makedirs(dir_path)

    # csv로 저장
    file_name = os.path.basename(path)
    df.to_csv(f"{dir_path}/{model_name}_{file_name}", encoding="utf-8-sig", index=False)

    print(f"- {dir_path}에 파일이 저장되었습니다.")

if __name__ == "__main__":
    print("- 리뷰 데이터셋 감성 분류를 시작합니다.")

    path = "./data/raw/review"
    for file in os.listdir(path):
        # model_name : 감성 분류에 사용한 모델(ex. ReBERT, ReELECTRA, KcBERT, KcELECTRA)
        Classification(f"{path}/{file}", model_name="ReBERT")

    print("- 감성 분류를 종료합니다.")