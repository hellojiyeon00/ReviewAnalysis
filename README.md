# 🛍️ 온라인 리뷰 특화 한국어 자연어 처리 모델: ReBERT, ReELECTRA
### **ReBERT, ReELECTRA: Domain-Adaptive Korean Language Models for Online Review Analysis**

---

## 🗓️ 프로젝트 기간
2025년 11월 03일 ~ 2025년 11월 14일

## 🧑‍💻 팀원
- 강지연 [@nouve53](https://github.com/nouve53)
- 곽동원 [@eee334223](https://github.com/eee334223)
- 안호용 [@hodol0213](https://github.com/hodol0213)
- 정수아 [@data-suah15](https://github.com/data-suah15)

---

## 1. 📘 프로젝트 개요

최근 전자상거래 시장이 급속히 성장함에 따라 **온라인 고객 리뷰(OCR, Online Customer Review)** 는 소비자의 구매 결정에 큰 영향을 끼치는 핵심 요인이 되었습니다.

본 프로젝트는 **온라인 패션 플랫폼 리뷰 데이터**를 기반으로 한국어 감성 분석 모델을 구축하는 것을 목표로 합니다.
이를 위해 범용 사전학습 언어모델인 **BERT**와 **ELECTRA**를 패션 플랫폼 리뷰 도메인에 최적화되도록 사전학습하여 **도메인 적응 언어 모델**을 새롭게 구현했습니다.

본 프로젝트에서 개발한 모델은 다음과 같습니다:
- **ReBERT (Review-BERT)**  
- **ReELECTRA (Review-ELECTRA)**

---

## 2. 🧩 모델 설명

### ReBERT
```
[모델 설명]
```

### ReELECTRA
```
[모델 설명]
```

---

## 3. 🛠️ 주요 라이브러리

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

## 4. 📁 프로젝트 구조

📂 project/

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
│       └── 📄 musinsa_review_{goods_no}.csv
│
└── processed/                           # 텍스트 전처리
    ├── model/                           # 모델 학습용
    │   ├── 📄 pretraining_preprocessed.txt
    │   ├── 📄 dapt_preprocessed.txt
    │   └── 📄 finetuning_preprocessed.txt
    │
    └── review/                          # 감성 분류
        ├── ELECTRA/
        │   └── 📄 labeled_review_{goods_no}.csv
        └── BERT/
            └── 📄 labeled_review_{goods_no}.csv
```

</details>

<details>
<summary>📂 model/</summary>

```
├── ReBERT/
│   ├── checkpoints/                # 체크포인트
│   ├── pretrained/                 # 사전학습 모델
│   ├── DAPT/                       # DAPT 모델
│   └── finetuned/                  # 파인튜닝 모델
│
├── ReELECTRA/
│   ├── checkpoints/                # 체크포인트
│   ├── pretrained/                 # 사전학습 모델
│   ├── DAPT/                       # DAPT 모델
│   └── finetuned/                  # 파인튜닝 모델
│
├── KcBERT/
│   ├── checkpoints/                # 체크포인트
│   └── finetuned/                  # 파인튜닝 모델
│
└── KcELECTRA/
    ├── checkpoints/                # 체크포인트
    └── finetuned/                  # 파인튜닝 모델
```

</details>

<details>
<summary>📂 src/</summary>

```
├── classification.py
├── crawling.py
├── KcBERT.py
├── KcELECTRA.py
├── preprocessing.py
├── tokenizer.py
│
├── ReBERT/
│   ├── pretraining.py
│   ├── DAPT.py
│   └── finetuning.py
│
└── ReELECTRA/
    ├── pretraining.py
    ├── DAPT.py
    └── finetuning.py
```

</details>

<details>
<summary>📄 requirements.txt</summary>
</details>

<details>
<summary>📄 README.md</summary>
</details>

---

## 5. ⚙️ 설치 방법

### 1) 저장소 클론
```python
git clone https://github.com/username/project.git
cd project
```

### 2) 환경 설정
```python
pip install -r requirements.txt
```

---

## 6. 📊 성능 평가

|Model         |KcBERT        |ReBERT        |KcELECTRA     |ReELECTRA     |
|--------------|:------------:|:------------:|:------------:|:------------:|
|Size          |Base          |Small(tuning) |Small         |Small         |
|Accuracy      |93.41%        |89.53%        |91.70%        |88.98%        |
|F1 Score      |94.62%        |91.53%        |93.21%        |90.90%        |

---

## 7. 📌 향후 계획

- 향후 계획 1
- 향후 계획 2
- 향후 계획 3
