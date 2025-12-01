import streamlit as st
from streamlit_option_menu import option_menu
import webbrowser
import pandas as pd
import os
from Crawling import crawling
from Classification import Classification

menu = ["크롤링", "감성 분류", "파일 확인"]
model = ["ReBERT", "ReELECTRA"]

with st.sidebar:
    choice = option_menu("Menu", menu)
if choice == menu[0]:
    # 무신사 url 연결
    url = f"https://www.musinsa.com/main/musinsa/recommend?gf=A"
    st.markdown(f"# 🔎 [무신사]({url}) 상품 검색")
    # 상품 번호 및 가져올 리뷰수 입력
    goods_no = st.text_input("상품 번호를 입력하세요:", "")
    target_count = st.number_input("불러올 리뷰 갯수를 입력하세요:", min_value=1, max_value=10000)
    # 검색 버튼
    if st.button("검색"):
        st.write("크롤링을 시작합니다.")
        crawling(goods_no, target_count)
        st.write("크롤링을 종료합니다.")

elif choice == menu[1]:
    # 저장된 리뷰 파일 불러오기
    review_file_list = [""]
    review_file_list.extend(os.listdir("./data/raw/review"))
    # 파일 선택
    select_file = st.selectbox("📂 data/raw/review", review_file_list)
    # 분류 모델 선택
    model.insert(0, "")
    select_model = st.selectbox("모델을 선택하세요", model)
    # 파일 경로에서 상품 번호 추출
    goods_no = os.path.basename(select_file).split(sep="_")[-1].split(sep=".")[0]
    if st.button("선택"):
        st.write(f"{select_model} 모델을 사용하여 감성 분류를 시작합니다.")
        Classification(goods_no, model_name=select_model)
        st.write(f"감성 분류를 종료합니다.")
        with st.expander(f"{select_model}_labeled_reviews_{goods_no}.csv"):
            df = pd.read_csv(f"./data/processed/review/{select_model}/{select_model}_labeled_reviews_{goods_no}.csv", encoding="utf-8-sig")
            st.dataframe(df)

elif choice == menu[2]:
    folder_path = "./data/processed/review"
    #ReBERT로 분류한 파일 불러오기
    with st.expander(f"{model[0]}"):
        ReBERT_path = f"./data/processed/review/{model[0]}"
        ReBERT_list = os.listdir(ReBERT_path)
        ReBERT_list.insert(0, "")
        # 파일 선택
        select_file = st.selectbox(f"📂 {ReBERT_path}", ReBERT_list)
        if select_file:
            # 선택한 파일 확인
            with st.expander(select_file):
                df = pd.read_csv(f"{ReBERT_path}/{select_file}", encoding="utf-8-sig")
                st.dataframe(df)
    
    #ReELECTRA로 분류한 파일 불러오기
    with st.expander(f"{model[1]}"):
        ReELECTRA_path = f"./data/processed/review/{model[1]}"
        ReELECTRA_list = os.listdir(ReELECTRA_path)
        ReELECTRA_list.insert(0, "")
        # 파일 선택
        select_file = st.selectbox(f"📂 {ReELECTRA_path}", ReELECTRA_list)
        if select_file:
            # 선택한 파일 확인
            with st.expander(select_file):
                df = pd.read_csv(f"{ReELECTRA_path}/{select_file}", encoding="utf-8-sig")
                st.dataframe(df)