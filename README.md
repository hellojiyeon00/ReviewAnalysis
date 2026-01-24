# 🛍️ 온라인 리뷰 특화 한국어 자연어 처리 모델: ReBERT, ReELECTRA
### **ReBERT, ReELECTRA: Domain-Adaptive Korean Language Models for Online Review Analysis**

---

## 🗓️ 프로젝트 기간
2025년 11월 03일 ~ 2025년 11월 14일

## 👥 팀명
**강지연과 아이들**

## 🧑‍💻 팀원
- 강지연 [@nouve53](https://github.com/nouve53)
- 곽동원 [@eee334223](https://github.com/eee334223)
- 안호용 [@hodol0213](https://github.com/hodol0213)
- 정수아 [@data-suah15](https://github.com/data-suah15)

---

## 1. 📘 프로젝트 개요

최근 전자상거래 시장이 급속히 성장함에 따라 **온라인 고객 리뷰(OCR, Online Customer Review)** 는 소비자의 구매 결정에 큰 영향을 끼치는 핵심 요인이 되었습니다.

본 프로젝트는 **패션 플랫폼 리뷰 데이터**를 기반으로 한국어 감성 분석 모델을 구축하는 것을 목표로 합니다.
이를 위해 범용 사전학습 언어모델인 **BERT**와 **ELECTRA**를 패션 플랫폼 리뷰 도메인에 최적화되도록 사전학습하여 **도메인 적응 언어 모델**을 새롭게 구현했습니다.

본 프로젝트에서 개발한 모델은 다음과 같습니다:
- **ReBERT (Review-BERT)**  
- **ReELECTRA (Review-ELECTRA)**

---

## 2. 🧩 모델 설명

### ReBERT

ReBERT는 Google의 BERT-Base를 기반으로, 리뷰 도메인 특성에 맞게 추가 사전학습(DAPT)을 수행한 모델입니다.

📌 MLM(Masked Language Model)

BERT의 MLM 방식은 입력 문장의 일부 단어를 가리고, 가려진 단어를 문맥을 기반으로 예측하도록 학습합니다.

학습 과정은 다음과 같습니다:

1) 입력 문장을 Subword 단위로 분리
2) 전체 토큰 중 15%를 선택해 마스킹
    - 80% → [MASK] 토큰으로 교체
    - 10% → 랜덤한 다른 단어로 교체
    - 10% → 원래 단어 유지
3) 모델은 양방향 문맥 정보를 활용해 마스킹된 위치의 원래 단어를 예측하도록 학습

이 과정을 통해 모델은 문장 내 단어를 더 잘 이해하고, 문맥을 파악하는 능력을 향상시킬 수 있습니다.

> 참고:
기존 BERT는 MLM과 NSP(Next Sentence Prediction) 방식을 함께 학습하지만, 리뷰 분석에서는 문장 간 관계 정보의 중요성이 낮고, NSP가 성능 향상에 기여하지 않는다는 연구 결과가 다수 존재합니다.
따라서 ReBERT는 NSP를 제외하고 MLM 기반으로만 사전학습을 진행했습니다.

### ReELECTRA

ReELECTRA는 Google의 ELECTRA-Small을 기반으로, 리뷰 도메인 특성에 맞게 추가 사전학습(DAPT)을 수행한 모델입니다.

📌 RTD(Replaced Token Detection)

ELECTRA의 RTD 방식은 입력 토큰이 진짜(original) 토큰인지 가짜(replaced) 토큰인지를 판별하는 이진 분류 문제로 학습합니다.

학습 과정은 다음과 같습니다:

1) Generator(생성기)와 Discriminator(판별기) 두개의 네트워크 사용
2) Generator
    - BERT와 동일한 MLM 방식 사용
    - 마스킹된 위치의 단어를 예측하여 가짜 토큰 생성
    - 생성된 가짜 토큰을 원래 문장에 삽입해 Discriminator의 입력으로 사용
3) Discriminator
    - 문장 내 각 토큰이 진짜 토큰인지 Generator가 만든 가짜 토큰인지를 판별하는 이진 분류(0 또는 1) 수행

ELECTRA는 문장 내 모든 토큰에 대해 학습 신호를 제공하므로, 일부 마스킹된 토큰만 예측하는 BERT보다 학습 속도가 빠르고 효율적입니다.

> [GAN](https://pseudo-lab.github.io/Tutorial-Book/chapters/GAN/Ch1-Introduction.html) vs ELECTRA:
ELECTRA는 BERT 기반 Transformer 구조 위에 GAN 아이디어를 적용한 언어 모델입니다.
주로 자연어 처리(NLP) 분야에서 사용되며, GAN과 달리 새로운 데이터(이미지, 텍스트 등)를 생성하는 것이 목적이 아니라, 문장 내 일부 토큰만 가짜로 만들어 판별하는 학습에 초점을 맞춘다는 점이 특징입니다.

---

## 3. 🗂️ 학습 데이터

### 1) Pre-training 데이터 (약 300만 문장)

- 📰 뉴스 기사 (AIHub)
- 🗣️ 구어체 발화 (AIHub)
- 🌐 웹사이트 문서 (AIHub)
- 🎬 영화/게임 리뷰 (GitHub)

### 2) DAPT 데이터 (22만 문장)

- 🛒 쇼핑몰 리뷰 (AIHub)

### 3) Fine-tuning 데이터 (20만 문장)

- 🛍️ 네이버 쇼핑 리뷰 (GitHub)
  - 긍정 리뷰: **60%**  
  - 부정 리뷰: **40%**

### 4) Crawling 데이터 (12만 문장) 

- **아우터**
   - 패딩  
   - 블레이저  
- **상의**
   - 맨투맨  
   - 후드티  
   - 반소매 티셔츠  
- **하의**
   - 트레이닝 팬츠  
   - 슬랙스  
   - 데님
- <details>
    <summary><b>📑 수집 항목</b></summary>
    <ul>
        <li>📝 리뷰 텍스트</li>
        <li>🙋 리뷰자 아이디</li>
        <li>⭐ 평점</li>
        <li>🛍️ 상품명</li>
        <li>💰 가격</li>
        <li>📈 누적 판매수</li>
        <li>👀 조회수</li>
        <li>🔖 할인율</li>
    </ul>
  </details>
- 출처: 👕 [무신사](https://www.musinsa.com/main/musinsa/recommend?gf=A) 리뷰

---

## 4. 🧹 텍스트 전처리 규칙

KcBERT 정제 규칙을 참고하여 다음을 적용:

### 🔤 문자 필터링
- 허용: **한글(자/모음), 영어, 특수문자, 유니코드 이모지, 숫자**
- 그 외 문자는 모두 제거

### ✂️ 중복 문자열 축약
- `soynlp` 활용  
- 예시: `ㅋㅋㅋㅋ` → `ㅋㅋ`

### 📏 짧은 문장 제거
- 글자 단위 **10글자 이하** 문장은 삭제

### 🗑️ 중복 문장 제거

---

## 5. 🛠️ 주요 라이브러리

### ✔ Modeling
- torch
- transformers
- tokenizers

### ✔ Preprocessing
- soynlp
- emoji

### ✔ Crawling
- selenium

### ✔ Data Analysis
- konlpy
- scikit-learn
- wordcloud
- matplotlib

> 전체 패키지 목록은 `requirements.txt` 참고.

---

## 6. 📁 프로젝트 구조

<details>
<summary>📂 data/</summary>

```
├── raw/                                 # 원본
│   ├── model/                           # 모델 학습용
│   │   ├── 📄 pretraining.txt
│   │   ├── 📄 dapt.txt
│   │   └── 📄 finetuning.txt
│   │
│   └── review/                          # 리뷰 데이터
│       └── 📄 musinsa_reviews_{goods_no}.csv
│
└── processed/                           # 텍스트 전처리
    ├── model/                           # 모델 학습용
    │   ├── 📄 pretraining_preprocessed.txt
    │   ├── 📄 dapt_preprocessed.txt
    │   └── 📄 finetuning_preprocessed.txt
    │
    └── review/                          # 감성 분류
        ├── KcBERT/
        │   └── 📄 labeled_reviews_{goods_no}.csv
        ├── KcELECTRA/
        │   └── 📄 labeled_reviews_{goods_no}.csv
        ├── ReBERT/
        │   └── 📄 labeled_reviews_{goods_no}.csv
        └── ReELECTRA/
            └── 📄 labeled_reviews_{goods_no}.csv
```

</details>

<details>
<summary>📂 model/</summary>

```
├── KcBERT/
│   └── finetuned/                  # 파인튜닝 모델
│       └── checkpoints/
│
├── KcELECTRA/
│   └── finetuned/                  # 파인튜닝 모델
│       └── checkpoints/
│
├── ReBERT/
│   ├── pretrained/                 # 사전학습 모델
│   │   └── checkpoints/
│   ├── DAPT/                       # DAPT 모델
│   │   └── checkpoints/
│   └── finetuned/                  # 파인튜닝 모델
│       └── checkpoints/
│
├── ReELECTRA/
│   ├── pretrained/                 # 사전학습 모델
│   │   └── checkpoints/
│   ├── DAPT/                       # DAPT 모델
│   │   └── checkpoints/
│   └── finetuned/                  # 파인튜닝 모델
│       └── checkpoints/
│
└── tokenizer/
```

</details>

<details>
<summary>📂 src/</summary>

```
├── KcBERT/
├── KcELECTRA/
├── ReBERT/
├── ReELECTRA/
├── Classification.py
├── Crawling.py
├── Preprocessing.py
└── Tokenizer.py
```

</details>

<details>
<summary>📄 README.md</summary>
</details>

<details>
<summary>📄 requirements.txt</summary>
</details>

---

## 7. 📊 성능 평가

|Model         |KcBERT        |ReBERT        |KcELECTRA     |ReELECTRA     |
|--------------|:------------:|:------------:|:------------:|:------------:|
|Size          |Base          |Small(tuning) |Small         |Small         |
|Accuracy      |93.41%        |89.53%        |91.70%        |88.98%        |
|F1 Score      |94.62%        |91.53%        |93.21%        |90.90%        |

---

## 8. ⚙️ 설치 방법

### 1) 저장소 클론
```python
git clone https://github.com/nouve53/team3.git
cd project
```

### 2) 가상환경 생성
```cmd
conda create -name <new_env> python=3.9
conda activate <new_env>
```

### 3) 패키지 설치
```python
pip install -r requirements.txt
```

---

## 9. 📌 향후 계획

- **도메인 일반화 성능 강화**  
  다양한 온라인 쇼핑 플랫폼의 리뷰 데이터를 추가 학습, 모델의 범용성 개선
- **효율성 최적화**  
  하이퍼파라미터 최적화 및 모델 경량화 기법을 적용, 실제 서비스 환경에서의 효율성 증대
- **사전학습 단계 도메인 특화 활용**  
  사전학습 단계에서 도메인 특화 데이터셋으로 구축된 단어 사전을 활용, 리뷰 도메인에서 자주 등장하는  
  신조어, 구어체, 상품 등 관련 용어를 더욱 정교하게 학습
  
