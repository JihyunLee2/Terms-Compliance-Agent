import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List # List 타입 힌트를 위해 추가
import utils


def run_batch_analysis(app, chunks, similarity_threshold, vectorstore):
    """
    여러 개의 조항(chunks)을 순회하며 일괄 분석합니다.
    (HITL이 없는 단순한 실행)
    """
    st.info(f"총 {len(chunks)}개 조항에 대한 분석을 시작합니다. (시간이 소요될 수 있습니다)")
    
    progress_bar = st.progress(0, text="분석 진행 중...")
    results = [] # 최종 결과 저장

    for i, chunk in enumerate(chunks):
        
        # 챗봇 모드와 동일하게 thread_id를 매번 새로 생성
        thread_id = f"batch_session_{datetime.now().timestamp()}_{i}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 챗봇 모드와 동일하게 initial_state 구성
        initial_state = {
            "clause": chunk,
            "iteration": 1,
            "session_id": thread_id,
            "validation_failed": False,
            "retrieved_cases_metadata": [],
            "retrieved_laws_metadata": [],
            "similarity_threshold": similarity_threshold
        }
        
        try:
            # 1. LangGraph app을 직접 호출 (노드 수동 호출이 아님)
            # app.invoke()는 '공정'/'불공정'을 알아서 판단하고 최종 결과(output)를 반환
            output = app.invoke(initial_state, config=config)
            
            # 2. 그래프 실행 결과(output) 저장           
            results.append({
                "original_clause": chunk,                               # 조항 원본  
                "fairness_label": output.get('fairness_label', 'N/A'),   #  판별
                "unfair_type": output.get('unfair_type', '—'),          # 불공정 유형
                "improvement_proposal": output.get('improvement_proposal', '—'),        # 개선 제안
                "related_cases_count": len(output.get('retrieved_cases_metadata', []))  # 참고 사례 수
            })

        except Exception as e:
            st.error(f"'조항 {i+1}' 분석 중 오류: {e}")
            results.append({
                "original_clause": chunk,
                "fairness_label": "오류",
                "unfair_type": f"분석 중 오류 발생: {e}",
                "improvement_proposal": "—",
                "related_cases_count": 0,
            })
            
        
        # 프로그레스 바 업데이트
        progress_bar.progress((i + 1) / len(chunks), text=f"분석 진행 중... ({i+1}/{len(chunks)})")

    progress_bar.empty()
    st.success("모든 조항 분석 완료!")
    
    # 4. 결과 리포트 표시 (새로 추가한 함수 호출)
    display_batch_results(results)
      
def display_batch_results(results: List[dict]):
    """
    일괄 분석 결과를 Streamlit UI에 리포트 형식으로 표시합니다.
    """
    # 1. (수정) 'unfair_type'이 아닌 'fairness_label'을 기준으로 필터링
    problematic_clauses = [
        r for r in results 
        if r['fairness_label'] == "불공정" # '불공정' 조항만 필터링
    ]
    
    st.header(f"검토 결과: 총 {len(results)}개 조항 중 {len(problematic_clauses)}개의 불공정 의심 조항 발견")
    st.divider()
    
    # 추가 11/16
    if not problematic_clauses:
        st.success("특별히 불공정으로 의심되는 조항이 발견되지 않았습니다.")
        return
    
    if problematic_clauses:
        st.subheader("불공정 의심 조항 상세")
        # 2. 영어 키('unfair_type', 'original_clause', 'improvement_proposal') 사용
        for i, res in enumerate(problematic_clauses):
            with st.expander(f"의심 조항 {i+1}: ({res['unfair_type']}) - {res['original_clause'][:50]}..."):
                
                # st.markdown()을 사용하여 Markdown 서식을 그대로 렌더링
                st.markdown(res['improvement_proposal'], unsafe_allow_html=True)
                
    
# --- 메인 실행 함수 ---
def run_pdf_batch_mode(app, vectorstore, current_threshold_value):
    st.header("PDF 약관 전체 검토")
    st.info("PDF 파일을 업로드하면 문서 전체를 분석하여 '불공정 의심 조항' 목록을 생성합니다.")

    uploaded_file = st.file_uploader(
        "📄 검토할 PDF 약관 파일을 업로드하세요.", 
        type="pdf",
        key="pdf_uploader" # key를 추가하여 탭 전환 시 파일이 유지되도록 함
    )
    
    if uploaded_file is not None:
        # 1. PDF 텍스트 추출
        pdf_text = utils.extract_text_from_pdf(uploaded_file)
        
        # 2. 텍스트 분할 (Chunking)
        chunks = utils.split_text_into_clauses(pdf_text)
        
        st.markdown(f"총 {len(chunks)}개의 조항(Chunk)이 감지되었습니다.")
        
        if st.button("전체 조항 분석 시작하기", type="primary", key="batch_start_btn"):
            # 3. vectorstore를 run_batch_analysis로 전달
            run_batch_analysis(app, chunks, current_threshold_value, vectorstore)
