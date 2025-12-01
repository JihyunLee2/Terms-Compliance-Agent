import streamlit as st
import graphviz
import pandas as pd

def draw_user_guide():
    st.header("사용 가이드 및 지원 범위")
    st.markdown("#### AI와 함께하는 약관 심사, 이렇게 진행하세요.")
    
    st.write("") # 여백

    # 1. 탭을 사용하여 정보 구조화
    tab1, tab2 = st.tabs(["이용 절차 (Workflow)", "지원 범위"])
    
    # --- [탭 1] 이용 절차 (시각화 중심) ---
    with tab1:
        st.markdown("#### 1. 업무 흐름도")
        st.caption("입력부터 최종 확정까지 4단계로 진행됩니다.")

        # Graphviz를 이용한 깔끔한 프로세스 다이어그램
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', size='12,4') # 가로 방향
        graph.attr('node', shape='rect', style='filled', 
                   fillcolor='#f0f2f6', fontname='Malgun Gothic', 
                   fontsize='14', margin='0.2')
        graph.attr('edge', fontsize='12')

        # 노드 정의
        graph.node('Input', '1. 조항 입력\n(분석 요청)', fillcolor='#e3f2fd') # 강조색
        graph.node('AI', '2. AI 정밀 진단\n(불공정/유사사례)')
        graph.node('Draft', '3. 개선안 제안\n(Redline)')
        graph.node('Human', '4. 전문가 검토\n(수락/수정/폐기)', fillcolor='#fff3e0') # 강조색

        # 엣지(화살표) 연결
        graph.edge('Input', 'AI')
        graph.edge('AI', 'Draft')
        graph.edge('Draft', 'Human', label=' 제안 확인')
        graph.edge('Human', 'Draft', label=' 💬 수정 요청\n(Feedback)', style='dashed', color='red')
        graph.edge('Human', 'Human', label=' ✅ 최종 확정', color='green')

        st.graphviz_chart(graph)
        
        st.write("") # 여백
        
        st.markdown("#### 2. 효과적인 사용을 위한 3단계 가이드")
        
        step1, step2, step3 = st.columns(3)
        
        with step1:
            with st.container(border=True):
                st.markdown("### 1️⃣ 입력")
                st.markdown("**'한 번에 하나씩'**")
                st.caption("""
                약관 전체를 붙여넣기보다,
                **분석이 필요한 '개별 조항' 단위**로
                입력해야 AI가 가장 정확하게 분석합니다.
                
                약관 전체를 넣고 싶다면, `PDF(전체 문서 분석)` 기능을 이용해주세요.
                """)
        
        with step2:
            with st.container(border=True):
                st.markdown("### 2️⃣ 검토")
                st.markdown("**'빨간 펜 선생님처럼'**")
                st.caption("""
                AI가 제안한 개선안에서
                **변경된 문구(Redline) 위주**로 확인하세요.
                법적 뉘앙스가 유지되었는지 체크합니다.
                """)
                st.write("")
                st.write("")
                st.write("")
                
        with step3:
            with st.container(border=True):
                st.markdown("### 3️⃣ 피드백")
                st.markdown("**'동료에게 말하듯이'**")
                st.caption("""
                결과가 아쉽다면 **구체적으로 명령**하세요.
                *"~부분은 빼줘", "좀 더 강한 어조로"* 처럼
                대화하듯 요청하면 즉시 수정합니다.
                """)
                st.write("")
                st.write("")
        
        st.divider()

        # 꿀팁 섹션
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **수정 요청(Feedback) 꿀팁**")
            st.markdown("""
            AI의 제안이 마음에 들지 않으면 채팅하듯 편하게 명령하세요.
            - *"2, 3번 참고 법령과 1번 유사 사례의 시정 취지를 반영하여, 입력된 조항의 불공정성을 해소하는 방향으로 다시 작성해줘"*
            - *"위약금을 5%로 낮춰줘"*
            - *"문구를 좀 더 부드럽게 다듬어줘"*
            - *"법적 근거를 더 보강해줘"*
            """)
            st.caption("법적 허용 범위 내에서 적용됩니다.")
        with col2:
            st.success("✨ **분석 완료 기준**")
            st.markdown("""
            - **[피드백]** 버튼을 눌러야 해당 조항 분석이 완료됩니다.
            - 완료된 내용은 자동으로 로그에 저장됩니다.
            """)

    # --- [탭 2] 분석 범위 (카드 UI 중심) ---
    with tab2:
        st.subheader("AI가 할 수 있는 것 vs 없는 것")
        st.markdown("본 시스템은 **'개별 조항(Clause)' 단위의 공정성 심사**에 특화되어 있습니다.")
        
        st.write("") 

        # 3단 컬럼으로 O/X 명확히 구분
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.container(border=True)
            st.success("✅ 문장/표현 레벨")
            st.markdown("""
            **[지원 기능]**
            - **오타 및 비문** 교정
            - **모호한 표현** 감지
            - **독소 조항** 키워드 식별
            """)
            
        with c2:
            st.container(border=True)
            st.info("✅ 법적 효력 레벨")
            st.markdown("""
            **[핵심 기능]**
            - **불공정 유형(8개)** 판별
            - **관련 법령** 매칭
            - **유사 심결례** 비교 검색
            """)
            
        with c3:
            st.container(border=True)
            st.warning("⚠️ 문서 구조 레벨")
            st.markdown("""
            **[지원 제한]**
            - **문서 서식/디자인**
            """)
            
        st.divider()
        st.caption("※ AI의 분석 결과는 법적 효력을 갖지 않으며, 최종 판단 책임은 담당자에게 있습니다.")
    pass

def draw_analysis_scope():
    st.header("데이터 구조 및 판단 기준")
    st.markdown("""
    본 서비스는 **공정거래위원회 심결례 보도자료**와 **현행 법령**을 기반으로 구축된 데이터셋을 근거로 약관의 공정성을 판단합니다.
    단순한 문장 유사도가 아닌, **법적 근거와 실제 시정 사례**에 기반한 분석 결과를 제공합니다.
    """)
    
    st.divider()

    # 탭을 사용하여 정보를 체계적으로 보여줍니다.
    tab1, tab2, tab3 = st.tabs(["데이터 출처 (법적 근거)", "공정/불공정 판단 기준", "데이터 신뢰성 검증"])

    # --- [탭 1] 데이터 출처 ---
    with tab1:
        st.subheader("1. 참고 법령 및 규정")
        st.caption("약관 심사에 필수적인 주요 법령 5종을 전수 학습하였습니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **약관의 규제에 관한 법률 (약관법)**: [직접 보러 가기](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%95%BD%EA%B4%80%EC%9D%98%EA%B7%9C%EC%A0%9C%EC%97%90%EA%B4%80%ED%95%9C%EB%B2%95%EB%A5%A0)
            - **약관심사지침 (공정위 예규)**: [직접 보러 가기](https://www.law.go.kr/%ED%96%89%EC%A0%95%EA%B7%9C%EC%B9%99/%EC%95%BD%EA%B4%80%EC%8B%AC%EC%82%AC%EC%A7%80%EC%B9%A8)
            - **전자금융거래법**: [직접 보러 가기](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%A0%84%EC%9E%90%EA%B8%88%EC%9C%B5%EA%B1%B0%EB%9E%98%EB%B2%95)
            """)
        with col2:
            st.markdown("""
            - **금융소비자 보호에 관한 법률 시행령**: [직접 보러 가기](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EA%B8%88%EC%9C%B5%EC%86%8C%EB%B9%84%EC%9E%90%EB%B3%B4%ED%98%B8%EC%97%90%EA%B4%80%ED%95%9C%EB%B2%95%EB%A5%A0%EC%8B%9C%ED%96%89%EB%A0%B9)
            - **금융소비자 보호에 관한 감독규정**: [직접 보러 가기](https://www.law.go.kr/%ED%96%89%EC%A0%95%EA%B7%9C%EC%B9%99/%EA%B8%88%EC%9C%B5%EC%86%8C%EB%B9%84%EC%9E%90%20%EB%B3%B4%ED%98%B8%EC%97%90%20%EA%B4%80%ED%95%9C%20%EA%B0%90%EB%8F%85%EA%B7%9C%EC%A0%95)
            """)
            
        st.write("")
        
        st.subheader("2. 불공정 약관 시정 사례")
        st.caption("공정거래위원회에서 실제 시정 조치한 보도자료를 기반으로 합니다.")
        st.info("""
        **총 115건의 불공정 심결례 데이터 보유**: [직접 보러 가기](https://huggingface.co/datasets/ensemble-2/kftc_unfair_terms_cases)
        - 출처: 공정거래위원회 보도자료 (2012년 ~ 2024년)
        - 내용: 위반 조항 원문, 시정 사유, 관련 법 조항, 시정 후 개선안 포함
        """)

    # --- [탭 2] 판단 기준 ---
    with tab2:
        st.subheader("1. 공정성 판단 로직")
        st.markdown("""
        - **🟢 공정:** 약관법 및 관련 법령, 신용카드 표준약관에 위배되지 않는 조항
        - **🔴 불공정:** 약관법, 약관심사지침을 위반하거나 공정위 과거 시정 사례와 유사한 조항
        """)
        
        st.write("")
        st.subheader("2. 불공정 유형 분류 (8대 유형)")
        st.caption("공정위의 불공정 약관 심사 기준을 8가지 유형으로 체계화하였습니다.")

        # 데이터프레임으로 깔끔하게 정리 (PDF 11페이지 참조)
        data = {
            "유형 (대분류)": [
                "1. 서비스 일방적 변경·중단",
                "2. 기한의 이익 상실",
                "3. 고객 권리 제한",
                "4. 통지·고지 부적절",
                "5. 계약 해지·변경 사유 포괄적",
                "6. 비용 과다 부과·환급 제한",
                "7. 면책·책임 전가",
                "8. 기타 불공정 약관"
            ],
            "상세 설명": [
                "사업자가 자의적으로 부가서비스·혜택을 중단/축소하거나 약관을 변경하는 경우",
                "연체 등 사유로 대출/서비스 기한이 즉시 소멸되어 전액 상환을 요구하는 경우",
                "항변권, 이의제기권 등 법적으로 보장된 고객의 권리를 제한하거나 포기하게 하는 경우",
                "중요한 변경 사항을 부적절한 방식(단순 게시 등)으로 알리거나 개별 통지를 누락하는 경우",
                "'기타 상당한 이유' 등 추상적이고 포괄적인 사유로 계약을 해지/변경하는 경우",
                "과도한 위약금/수수료를 부과하거나, 포인트/선불금 환급을 부당하게 제한하는 경우",
                "사업자의 귀책사유를 면책하거나, 책임을 고객에게 부당하게 떠넘기는 경우",
                "개인정보 남용, 특수상품(리스 등), 불리한 재판 관할 합의 등 기타 유형"
            ]
        }
        df = pd.DataFrame(data)
        st.table(df)

    # --- [탭 3] 데이터 신뢰성 검증 ---
    with tab3:
        st.subheader("데이터 품질 확보 프로세스")
        st.markdown("법률 전문가 및 LLM을 활용한 **3단계 교차 검증**을 통해 데이터의 정합성을 확보했습니다.")
        
        # 시각적인 단계 표현
        st.markdown("""
        1. **1단계: 휴먼 라벨링** - 법령 및 심사지침에 근거하여 '왜 불공정한지' 법적 사유 작성
        2. **2단계: LLM 자동 검수** - AI를 활용하여 라벨 오류 및 법령 불일치 1차 탐지
        3. **3단계: 교차 검수 (Cross-Validation)** - 4명의 검수자가 데이터를 교차 확인하여 주관적 편향 제거
        """)
        
        st.info("""
        **✨ 개선안 생성 원칙**
        단순한 문장 교정이 아니라, **공정위의 실제 시정 조치 문구** 또는 **표준약관**을 기반으로 가장 법적 리스크가 적은 개선안을 제안합니다.
        """)
    pass