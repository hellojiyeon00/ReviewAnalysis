import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from google import genai
import os
import re
import json

# Gemini API 설정
model = genai.Client(api_key=os.getenv("LLM_API_KEY"))

# 페이지 설정
st.set_page_config(page_title="무신사 리뷰 분석", layout="wide")

# 데이터셋 경로
path = "./data/processed/review/ReELECTRA"

# --- 1. Sidebar(설정) ---
with st.sidebar:
    st.title("🔍 Dashboard")
    
    # 경로 내 파일 유무 확인
    if os.path.exists(path):
        review_file_list = [f for f in os.listdir(path) if f.endswith('.csv')]
        selected_product = st.selectbox("상품 선택", review_file_list)
    else:
        st.error(f"경로를 찾을 수 없습니다: {path}")
        st.stop()
    
    st.divider()
    st.markdown("""
    **Model Info**
    - Analysis: `ReELECTRA`
    - Insight: `Gemini-2.5-Flash`
    """)

# 캐싱 함수 수정: 파일명이 바뀔 때마다 새로 로드하도록 인자 추가
@st.cache_data
def load_data(file_name):
    file_path = os.path.join(path, file_name)
    df = pd.read_csv(file_path, encoding="utf-8-sig", index_col=0)
    df['sentiment'] = df['label'].map({1: 'Positive', 0: 'Negative'})
    return df

df = load_data(selected_product)

# Sidebar 하단 데이터 범위 설정
with st.sidebar:
    st.title("📂 데이터 범위")
    st.write(f"전체 리뷰: **{len(df):,}개**")
    review_range = st.slider("분석 범위", 100, len(df), int(len(df)/2))
    target_df = df.iloc[:review_range]

# --- 2. Section 1: 통계 ---
st.header(f"📊 {selected_product.replace('.csv', '')} 분석 리포트")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("긍정/부정 비율")
    sentiment_counts = target_df['sentiment'].value_counts()
    fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     color=sentiment_counts.index,
                     color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B'},
                     hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("별점(Star) 분포")
    fig_star = px.histogram(target_df, x='star', nbins=5,
                            color='sentiment', barmode='group',
                            color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B'})
    st.plotly_chart(fig_star, use_container_width=True)

st.divider()

# --- 3. Section 2: Gemini 인사이트 ---
st.header("💡 Gemini AI 심층 요약")

def get_gemini_insight(data):
    # 효율적인 분석을 위해 긍정/부정 리뷰 샘플링
    pos_text = "\n".join(data[data['label'] == 1]['review'].head(15).tolist())
    neg_text = "\n".join(data[data['label'] == 0]['review'].head(15).tolist())
    
    prompt = f"""
    당신은 패션 이커머스 분석가입니다. 다음 리뷰 데이터를 바탕으로 상품 분석 보고서를 작성하세요.
    
    [긍정 리뷰]: {pos_text}
    [부정 리뷰]: {neg_text}
    
    보고서 형식:
    1. **이 상품의 핵심 포인트**: 3가지 (글머리 기호)
    2. **주요 불만 사항**: 3가지 (글머리 기호)
    3. **종합 추천 의견**: 한 줄 요약
    """
    response = model.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

if st.button("✨ Gemini 인사이트 도출하기"):
    with st.spinner('리뷰 원문을 분석 중입니다...'):
        insight_text = get_gemini_insight(target_df)
        st.markdown(insight_text)

st.divider()

# --- 4. Section 3: 키워드 및 워드클라우드 ---
st.header("🏷️ 핵심 키워드 추출")

# Gemini 키워드 추출
def get_keywords_from_gemini(data, sentiment_label):
    # 해당 감성(1:긍정, 0:부정) 리뷰 추출
    reviews = "\n".join(data[data['label'] == sentiment_label]['review'].head(50).tolist())
    
    sentiment_name = "긍정" if sentiment_label == 1 else "부정"
    
    prompt = f"""
    다음은 무신사 상품의 {sentiment_name} 리뷰들입니다.
    리뷰를 분석해서 핵심 키워드 10개를 뽑고, 각 키워드의 중요도(1~100)를 결정해주세요.
    반드시 다음 JSON 형식으로만 답하세요:
    {{"keywords": [{{"word": "단어", "score": 80}}, ...]}}
    
    리뷰 데이터:
    {reviews}
    """

    try:
        response = model.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
        )
        raw_text = response.text
        
        # 정규표현식으로 JSON 블록만 추출 (마크다운 기호 제거 등)
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_data = json.loads(json_match.group())
            # {단어: 점수} 딕셔너리 형태로 변환
            return {item['word']: item['score'] for item in json_data['keywords']}
        return None

    except Exception as e:
        st.error(f"Gemini 분석 중 오류 발생: {e}")
        return None

# 워드클라우드 시각화
def display_wordcloud(keywords, color_theme):
    # WordCloud 객체 생성 (점수 기반 빈도수 적용)
    wc = WordCloud(
        font_path="malgun.ttf",
        width=800, 
        height=400,
        background_color='white',
        colormap=color_theme,
        max_words=10
    ).generate_from_frequencies(keywords)

    # Matplotlib으로 그리기
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

c1, c2 = st.columns([1, 1])

with c1:
    if st.button("👍 긍정 리뷰 TOP 10"):
        with st.spinner("긍정 리뷰 분석 중..."):
                pos_data = get_keywords_from_gemini(target_df, 1)
                display_wordcloud(pos_data, 'summer')

with c2:
    if st.button("👎 부정 리뷰 TOP 10"):
        with st.spinner("부정 리뷰 분석 중..."):
                neg_data = get_keywords_from_gemini(target_df, 0)
                display_wordcloud(neg_data, 'autumn')