from config2 import *
import streamlit as st
import re
import os
import json
from datetime import datetime
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt 
from langchain_core.tracers.context import tracing_v2_enabled
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prompts_app import ACTIVE_UNFAIR_TYPE_PROMPT, ACTIVE_IMPROVEMENT_PROMPT

load_dotenv()

# LLM 및 임베딩 초기화
embeddings = UpstageEmbeddings(model=EMBEDDING_MODEL)
llm = ChatUpstage(model=LLM_MODEL)

class ContractState(TypedDict):
    clause: str
    cleaned_text: str
    unfair_type: str
    related_cases: str
    improvement_proposal: str
    user_feedback: str
    modify_reason: str
    retry_action: str
    session_id: str
    iteration: int
    validation_failed: bool
    retrieved_cases_metadata: List[dict]
    retrieved_laws_metadata: List[dict]
    similarity_threshold: float

def load_vectordb():
    print("벡터 DB 로드 중...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name="contract_laws",
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("벡터 DB 로드 완료!\n")
    return vectorstore

def is_valid_contract_clause(clause: str) -> tuple[bool, str]:
    clause = clause.strip()
    
    if len(clause) < 20:
        return False, "입력이 너무 짧습니다 (최소 20자 필요)"
    
    contract_keywords = [
        '조항', '조건', '약관', '규정', '제', '항', '조', '자',
        '금지', '가능', '불가', '의무', '책임', '권리', '계약',
        '해지', '중단', '변경', '환불', '배상', '배제', '면책',
        '수수료', '이용료', '결제', '할인', '서비스', '제공',
        '개인정보', '보호', '이용', '관리', '통지', '동의',
        '유효', '기간', '상호', '시행', '효력', '청구', '위반',
        '손해배상', '면책조항', '이용자', '회사', '당사자'
    ]
    
    has_keyword = any(keyword in clause for keyword in contract_keywords)
    
    if not has_keyword:
        return False, "약관 관련 키워드 미검출 (예: 조항, 약관, 조건, 의무 등)"
    
    question_marks = ['?', '?']
    is_question = any(q in clause for q in question_marks)
    
    if is_question:
        return False, "질문 형식으로 보입니다. 약관 조항을 입력해주세요"
    
    return True, "검증 통과"

def clean_text_node(state: ContractState):
    print(f"\n[노드1] Rule-based 검증 + 텍스트 정제\n")
    
    is_valid, validation_msg = is_valid_contract_clause(state['clause'])
    print(f"[Rule-based 검증 결과] {validation_msg}")
    
    if not is_valid:
        print(f"-> API 호출 중단\n")
        return {
            "cleaned_text": "[룰 베이스 거부] 약관 조항이 아님",
            "validation_failed": True
        }
    
    print(f"-> 검증 통과\n")
    
    original_text = state['clause']
    cleaned = original_text
    
    cleaned = re.sub(r'^[\s•\-\*]+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', cleaned)
    cleaned = re.sub(r'\(\d+\)\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    print(f"[정제 전] {len(original_text)}자")
    print(f"{original_text}\n")
    print(f"[정제 후] {len(cleaned)}자")
    print(f"{cleaned}\n")
    
    return {
        "cleaned_text": cleaned,
        "validation_failed": False
    }

def classify_type_node(state: ContractState):
    print(f"[노드2] Solar API - 불공정 유형 분류\n")
    
    prompt = ACTIVE_UNFAIR_TYPE_PROMPT.format(
        clause=state['cleaned_text']
    )
    
    unfair_type = llm.invoke(prompt).content.strip()
    
    print(f"분류 결과: {unfair_type}\n")
    
    return {"unfair_type": unfair_type}

def retrieve_node(state: ContractState, vectorstore):
    
    current_threshold = state.get('similarity_threshold', SIMILARITY_THRESHOLD)
    print(f"[노드3] 검색 (임계값: {current_threshold:.0%})")
    
    search_query = f"{state['cleaned_text']}"
    
    # 1. 사례 검색 (유사도 점수 포함)
    # 이 함수는 (doc, similarity_score) 튜플을 반환합니다. (1.0이 100% 유사)
    results_cases_with_scores = vectorstore.similarity_search_with_relevance_scores(
        search_query, 
        k=SEARCH_TOP_K_CASES, 
        filter={"source_type": "case"}
    )
    
    filtered_cases_meta = []
    
    for i, (doc, similarity_score) in enumerate(results_cases_with_scores, 1):
        
        # similarity_score (예: 0.75)를 current_threshold (예: 0.70)와 직접 비교
        if similarity_score >= current_threshold:
            print(f"  ✓ 사례 통과 (유사도 {similarity_score:.1%})") # 디버깅용 로그
            filtered_cases_meta.append({
                "index": i,
                "similarity": similarity_score, # 계산이 아닌, 반환된 점수 그대로 사용
                "content": doc.page_content,
                "date": doc.metadata.get('date', 'N/A'),
                "case_type": doc.metadata.get('case_type', ''),
                "explanation": doc.metadata.get('explanation', ''),
                "conclusion": doc.metadata.get('conclusion', ''),
                "related_law": doc.metadata.get('related_law', '')
            })
        else:
             print(f"  ✗ 사례 필터됨 (유사도 {similarity_score:.1%})") # 디버깅용 로그

    # 표시 개수 제한
    final_cases_meta = filtered_cases_meta[:MAX_DISPLAY_CASES]
    
    # 2. 법령 검색
    law_query = " ".join([c['related_law'] for c in final_cases_meta if c['related_law']])
    if not law_query: law_query = search_query
        
    results_laws_with_scores = vectorstore.similarity_search_with_relevance_scores(
        law_query, 
        k=SEARCH_TOP_K_LAWS, 
        filter={"source_type": "law"}
    )
    
    final_laws_meta = []
    for i, (doc, similarity_score) in enumerate(results_laws_with_scores, 1):
        if similarity_score >= current_threshold:
            final_laws_meta.append({
                "index": i,
                "similarity": similarity_score,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
    final_laws_meta = final_laws_meta[:MAX_DISPLAY_LAWS]

    # 3. LLM 프롬프트용 텍스트 생성
    retrieved_text = f"[유사 시정 사례] ({len(final_cases_meta)}건, 임계점: {current_threshold:.0%})\n"
    
    for c in final_cases_meta:
        retrieved_text += f"\n- 사례{c['index']} (유사도 {c['similarity']:.1%}): {c['content']}\n"
        if c['related_law']:
            retrieved_text += f"  (관련법: {c['related_law']})\n"
    
    retrieved_text += f"\n[관련 법령] ({len(final_laws_meta)}건)\n"
    for l in final_laws_meta:
        retrieved_text += f"- 법령{l['index']} (유사도 {l['similarity']:.1%}): {l['content']}\n"

    return {
        "related_cases": retrieved_text,
        "retrieved_cases_metadata": final_cases_meta,
        "retrieved_laws_metadata": final_laws_meta
    }

def generate_proposal_node(state: ContractState):
    print(f"[노드4] Solar API - 개선안 생성 (반복: {state['iteration']}/{MAX_ITERATIONS})\n")
    
    # 파일 이름-법 이름 매핑 딕셔너리 추가
    LAW_FILENAME_MAP = {
        "1_약관법.pdf": "약관법",
        "1-2_약관심사지침.pdf": "약관심사지침",
        "2_금융소비자법시행령.pdf": "금융소비자법 시행령",
        "3_금융소비자보호에관한감독규정.pdf": "금융소비자보호 감독규정",
        "4_전자금융거래법.pdf": "전자금융거래법" 
        # (build_vectordb.py의 pdf_files 목록과 일치시켜야 함)
    }
    
    # 상태에서 필요한 정보 추출
    unfair_type = state['unfair_type']
    cases_meta = state.get('retrieved_cases_metadata', [])
    laws_meta = state.get('retrieved_laws_metadata', [])
    original_clause = state['cleaned_text']
    
    # 최종 출력 문자열(final_output) 조립 시작 ---
    
    # 0. 불공정 여부 판단
    final_output = "### 0. 불공정 여부 판단\n"
    if unfair_type == "공정":
        final_output += "✅ **공정**\n"
    else:
        # classify_type_node가 '불공정 (유형)'을 반환하므로 하이픈 제거
        final_output += f"❌ **{unfair_type}**\n"
        
    # 1. 쿼리와 유사한 사례
    final_output += "\n### 1. 유사 불공정 사례\n"
    if not cases_meta:
        final_output += "관련 사례를 찾지 못했습니다.\n"
    else:
        for case in cases_meta:
            case_summary = case['content'].split('약관:')[1].split('결론:')[0].strip()
            if len(case_summary) > 70:
                case_summary = case_summary[:70] + "..."
            final_output += f"* **`(유사도 {case['similarity']:.0%})`** {case_summary}\n"
            
    # 2. 쿼리와 유사한 법령
    final_output += "\n### 2. 참고 법령\n"
    if not laws_meta:
        final_output += "관련 법령을 찾지 못했습니다.\n"
    else:
        for law in laws_meta:
            similarity = law['similarity']
            content = law['content'].strip()
            metadata = law.get('metadata', {})
            
            # 메타데이터에서 파일 이름으로 법 이름 찾기
            source_file = metadata.get('source_file', '알 수 없는 법령')
            law_name = LAW_FILENAME_MAP.get(source_file, source_file)
            
            # 내용 요약 (70자)
            if len(content) > 70:
                content = content[:70] + "..."
            
            # [ㅇㅇ법] - [제ㅇ조...] 형식으로 조합
            final_output += f"* **`(유사도 {similarity:.0%})`** **`[{law_name}]`** - {content}\n"

    # --- 2. "공정"할 경우, 여기서 완료 ---
    if unfair_type == "공정":
        print("공정 조항으로 판단되어 개선안 생성 없이 완료.\n")
        return {"improvement_proposal": final_output}

    # --- 3. "불공정"할 경우, LLM을 호출하여 개선안 + 표 생성 ---
    
    print("불공정 조항으로 판단되어 LLM 개선안 생성 시작...\n")
    
    # 사용자 피드백 (재시도 시)
    feedback_context = ""
    if state.get('modify_reason'):
        feedback_context = f"\n[추가 사용자 피드백]\n{state['modify_reason']}\n위 의견을 반영해 다시 작성하세요.\n"

    # LLM에 전달할 프롬프트 (인라인 diff 및 테이블 생성 요청)
    prompt = ACTIVE_IMPROVEMENT_PROMPT.format(
        original_clause=original_clause,
        unfair_type=unfair_type,
        related_context=state['related_cases'], # prompts.py의 {related_context}와 매칭
        feedback_context=feedback_context
    )
    
    # LLM 호출
    llm_response = llm.invoke(prompt).content
    
    # LLM 응답을 최종 출력에 추가
    final_output += f"\n{llm_response}"
    
    print("LLM 개선안 생성 완료.\n")
    
    return {"improvement_proposal": final_output}

def ui_feedback_node(state: ContractState):
    print(f"\n[노드5] UI 피드백 대기 (반복: {state['iteration']}/{MAX_ITERATIONS})\n")
    print(f"개선안:\n{state['improvement_proposal']}\n")
    
    if state.get('retrieved_cases_metadata'):
        print(f"참고한 사례 수: {len(state['retrieved_cases_metadata'])}개")
        for case in state['retrieved_cases_metadata']:
            print(f"  - 사례 {case['index']}: 유사도 {case['similarity']:.2%}")
    
    return interrupt(state)

def process_feedback_node(state: ContractState):
    feedback = state['user_feedback']
    retry_action = state.get('retry_action', '')
    current_iteration = state.get('iteration', 1)
    
    if feedback == "approved":
        save_result(
            state=state,
            status="approved",
            iteration=current_iteration,
            total_iterations=current_iteration
        )
        print("[노드6] 결과 저장 완료 (수락)")
        print(f"총 {current_iteration}회 반복 후 완료\n")
        return {
            "user_feedback": "approved",
            "retry_action": ""
        }
    
    elif feedback == "rejected":
        if retry_action == "retry":
            new_iteration = current_iteration + 1
            save_result(
                state=state,
                status="rejected_retry",
                iteration=current_iteration
            )
            print(f"[노드6] 거절 기록 (재시도 예정)")
            print(f"-> 반복 {new_iteration}차 진행\n")
            return {
                "user_feedback": "rejected",
                "iteration": new_iteration,
                "retry_action": "retry"
            }
        else:
            save_result(
                state=state,
                status="rejected_discard",
                iteration=current_iteration,
                total_iterations=current_iteration
            )
            print(f"[노드6] 결과 저장 완료 (거절 및 폐기)\n")
            return {
                "user_feedback": "rejected",
                "retry_action": "discard"
            }
    
    elif feedback == "modify":
        if current_iteration >= MAX_ITERATIONS:
            save_result(
                state=state,
                status="max_iteration_reached",
                iteration=current_iteration,
                total_iterations=current_iteration,
                modify_reason="반복 횟수 제한 도달"
            )
            print(f"[노드6] 반복 횟수 제한 도달")
            print(f"총 {current_iteration}회 반복 (최대값)\n")
            return {
                "user_feedback": "approved", 
                "retry_action": ""
            }
        
        new_iteration = current_iteration + 1
        save_result(
            state=state,
            status="modify_request",
            iteration=current_iteration,
            modify_reason=state.get('modify_reason', '')
        )
        print(f"[노드6] 수정 요청 저장")
        print(f"-> 반복 {new_iteration}차 진행\n")
        return {
            "user_feedback": "modify",
            "iteration": new_iteration,
            "modify_reason": state.get('modify_reason', ''),
            "retry_action": ""
        }
    
    return {
        "user_feedback": feedback,
        "retry_action": ""
    }

def route_feedback(state: ContractState) -> str:
    if state.get('validation_failed', False):
        print("\n[라우팅 규칙 적용]")
        print(f"- 조건: validation_failed == True")
        print(f"- 액션: 그래프 즉시 종료")
        print(f"- 상태: 룰베이스 검증 실패\n")
        return "end"
    
    feedback = state.get('user_feedback', '').lower()
    retry_action = state.get('retry_action', '')
    current_iteration = state.get('iteration', 1)
    
    print(f"\n[라우팅 규칙 적용 - 반복 횟수: {current_iteration}/{MAX_ITERATIONS}]")
    
    if feedback == "approved":
        print(f"- 조건: user_feedback == 'approved'")
        print(f"- 액션: 그래프 종료 (결과 저장)")
        print(f"- 상태: 완료\n")
        return "end"
    
    elif feedback == "rejected" and retry_action == "retry":
        print(f"- 조건: user_feedback == 'rejected' AND retry_action == 'retry'")
        print(f"- 액션: generate 노드로 이동 (다른 개선안 생성)")
        print(f"- 상태: 재시도 (새로운 개선안)\n")
        return "generate"
    
    elif feedback == "rejected" and retry_action == "discard":
        print(f"- 조건: user_feedback == 'rejected' AND retry_action == 'discard'")
        print(f"- 액션: 그래프 종료 (폐기)")
        print(f"- 상태: 거절 및 폐기\n")
        return "end"
    
    elif feedback == "modify" and current_iteration < MAX_ITERATIONS:
        next_iteration = current_iteration + 1
        print(f"- 조건: user_feedback == 'modify' AND iteration({current_iteration}) < MAX({MAX_ITERATIONS})")
        print(f"- 액션: generate 노드로 이동 (피드백 반영)")
        print(f"- 상태: 반복 {next_iteration}차 진행\n")
        return "generate"
    
    elif feedback == "modify" and current_iteration >= MAX_ITERATIONS:
        print(f"- 조건: user_feedback == 'modify' AND iteration({current_iteration}) >= MAX({MAX_ITERATIONS})")
        print(f"- 반복 횟수 제한 도달!")
        print(f"- 액션: 그래프 종료 (강제)")
        print(f"- 상태: 반복 제한 도달\n")
        return "end"
    
    else:
        print(f"- 기타 조건")
        print(f"- 액션: 그래프 종료\n")
        return "end"

def save_result(state: ContractState, status: str, iteration: int,
                modify_reason: str = "", total_iterations: int = None):
    result = {
        "timestamp": datetime.now().isoformat(),
        "session_id": state['session_id'],
        "status": status,
        "iteration": iteration,
        "total_iterations": total_iterations or iteration,
        "original_clause": state['clause'],
        "cleaned_text": state['cleaned_text'],
        "unfair_type": state['unfair_type'],
        "improvement_proposal": state['improvement_proposal'],
        "modify_reason": modify_reason
    }
    
    filename = f"{status}_data.jsonl"
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

def extract_text_from_pdf(uploaded_file):
    """PDF 파일에서 텍스트를 추출합니다."""
    try:
        reader = pypdf.PdfReader(uploaded_file)
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n\n"
        return pdf_text
    except Exception as e:
        st.error(f"PDF 텍스트 추출 실패: {e}")
        return ""

def split_text_into_clauses(full_text: str) -> List[str]:
    """긴 텍스트를 의미 있는 조항(Chunk) 단위로 분할합니다."""
    # 법률 문서에 적합한 구분자 설정
    # 예: "제 1 조", "1.", "가.", "①" 등
    # RecursiveCharacterTextSplitter는 \n\n을 우선으로 자릅니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 조항 하나의 최대 길이 (조절 필요)
        chunk_overlap=100, # 조항간 겹침
        separators=[
            "\n\n", "\n", ". ", " "
        ],
        length_function=len,
    )
    
    chunks = text_splitter.split_text(full_text)
    
    # 너무 짧은 청크(예: 목차, 페이지 번호) 필터링
    # 기존 'is_valid_contract_clause'의 최소 길이 검사(20자) 활용
    valid_chunks = [
        chunk for chunk in chunks 
        if is_valid_contract_clause(chunk)[0] # 기존 룰베이스 검증 재활용
    ]
    
    return valid_chunks

def run_batch_analysis(app, chunks, similarity_threshold, vectorstore):
    """
    여러 개의 조항(chunks)을 순회하며 일괄 분석합니다.
    (HITL이 없는 단순한 실행)
    """
    try:
        pass
    except Exception as e:
        st.error(f"벡터 DB 접근 실패 (일괄 처리): {e}")
        return

    st.info(f"총 {len(chunks)}개 조항에 대한 분석을 시작합니다. (시간이 소요될 수 있습니다)")
    
    progress_bar = st.progress(0, text="분석 진행 중...")
    results = [] # 최종 결과 저장

    for i, chunk in enumerate(chunks):
        
        try:
            # 1. 초기 상태 정의 (피드백 노드가 없으므로 단순화)
            current_state = ContractState(
                clause=chunk,
                cleaned_text=chunk, # 스플리터가 이미 정제했다고 가정
                iteration=1,
                session_id=f"batch_{i}",
                similarity_threshold=similarity_threshold,
                validation_failed=False
            )

            # 2. 노드 순차 실행 (LangGraph 'app' 대신 직접 호출)
            
            # (Clean 노드는 split_text_into_clauses에서 처리)
            
            # [노드2] 유형 분류
            type_result = classify_type_node(current_state)
            current_state.update(type_result)
            
            # [노드3] 검색
            retrieve_result = retrieve_node(current_state, vectorstore)
            current_state.update(retrieve_result)
            
            # [노드4] 개선안 생성
            proposal_result = generate_proposal_node(current_state)
            current_state.update(proposal_result)
            
            # 3. 결과 저장
            results.append({
                "original_clause": chunk,
                "unfair_type": current_state['unfair_type'],
                "improvement_proposal": current_state['improvement_proposal'],
                "related_cases_count": len(current_state['retrieved_cases_metadata'])
            })

        except Exception as e:
            st.error(f"'조항 {i+1}' 분석 중 오류: {e}")
            results.append({
                "original_clause": chunk,
                "unfair_type": "오류",
                "improvement_proposal": f"분석 중 오류 발생: {e}",
                "related_cases_count": 0
            })
        
        # 프로그레스 바 업데이트
        progress_bar.progress((i + 1) / len(chunks), text=f"분석 진행 중... ({i+1}/{len(chunks)})")

    progress_bar.empty()
    st.success("모든 조항 분석 완료!")
    
    # 4. 결과 리포트 표시
    display_batch_results(results)

def display_batch_results(results: List[dict]):
    """
    일괄 분석 결과를 Streamlit UI에 리포트 형식으로 표시합니다.
    """
    
    # '8. 기타 불공정 약관' 대신 '공정'을 기준으로 필터링
    problematic_clauses = [
        r for r in results 
        if r['unfair_type'] not in ["공정", "오류"]
    ]
    
    st.header(f"검토 결과: 총 {len(results)}개 조항 중 {len(problematic_clauses)}개의 불공정 의심 조항 발견")
    
    for i, res in enumerate(problematic_clauses):
        with st.expander(f"의심 조항 {i+1}: ({res['unfair_type']}) - {res['original_clause'][:50]}..."):
            
            # st.markdown()을 사용하여 Markdown 서식을 그대로 렌더링
            st.markdown(res['improvement_proposal'], unsafe_allow_html=True)
            
@st.cache_resource
def get_app_and_vectorstore():
    vectorstore = load_vectordb()
    
    graph = StateGraph(ContractState)
    
    graph.add_node("clean", clean_text_node)
    graph.add_node("classify", classify_type_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, vectorstore))
    graph.add_node("generate", generate_proposal_node)
    graph.add_node("feedback", ui_feedback_node)
    graph.add_node("process_feedback", process_feedback_node)
    
    graph.set_entry_point("clean")
    
    def route_after_clean(state: ContractState) -> str:
        if state.get('validation_failed', False):
            return "end"
        return "classify"
    
    graph.add_conditional_edges(
        "clean",
        route_after_clean,
        {
            "end": END,
            "classify": "classify"
        }
    )
    
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "feedback")
    graph.add_edge("feedback", "process_feedback")
    
    graph.add_conditional_edges(
        "process_feedback",
        route_feedback,
        {
            "end": END,
            "generate": "generate"
        }
    )
    
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    
    return app, vectorstore

def main_chatbot_ui():
    st.set_page_config(page_title="법률 약관 검토 챗봇", layout="wide")
    st.title("법률 약관 검토 챗봇")
    st.caption("본 분석은 법적 효력을 가지지 않으며, 법률 자문을 대체하지 않습니다. 중대한 법적 판단은 반드시 자격 있는 법률 전문가와의 상담을 통해 이루어져야 합니다.")
    
    with st.sidebar:
        st.subheader("모델 정보")
        st.markdown(
            """
            * **모델:** Solar-Pro2
            * **버전:** 약관 분석 모듈 v1.0
            * **최근 업데이트:** 2025.11
            * **성능 범위:** 불공정 여부 판단, 유사 사례/법령 검색, 개선안 생성
            """
        )
        st.divider()
        
        st.header("검색 설정")
        similarity_threshold_percent = st.slider(
            "유사도 임계값 (%)",
            min_value=0,
            max_value=100,
            value=int(SIMILARITY_THRESHOLD * 100), # config 기본값 사용
            step=5,
            format="%d%%"
        )
        current_threshold_value = similarity_threshold_percent / 100.0
        st.caption(f"현재 설정: {similarity_threshold_percent}% 이상")
        st.divider()
        
    try:
        app, vectorstore = get_app_and_vectorstore()
    except Exception as e:
        st.error(f"애플리케이션 로드 실패: {e}")
        st.error("Chroma DB 파일('./chroma_db')이 올바르게 위치해 있는지 확인하세요.")
        return
        
    tab1, tab2 = st.tabs(["💬 챗봇 (단일 조항 검토)", "📄 PDF (전체 문서 검토)"])
    
    with tab1:
        run_chatbot_mode(app, current_threshold_value)
    
    with tab2:
        run_pdf_batch_mode(app, vectorstore, current_threshold_value)


def run_chatbot_mode(app, current_threshold_value):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "hitl_pending" not in st.session_state:
        st.session_state.hitl_pending = False
    if "current_state" not in st.session_state:
        st.session_state.current_state = {}
    if "pending_feedback" not in st.session_state:
        st.session_state.pending_feedback = None
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": """### 안녕하세요, 법률 약관 검토 챗봇입니다👋\n
새로운 약관 조항의 공정성 검토를 도와드리겠습니다. 분석을 원하는 **약관 조항**만 아래 채팅창에 입력해 주세요.
            
        [입력 예시]
        회원이 본 카드의 발급 목적과 다르게 이용한다고 카드사가 판단하거나, 
        기타 이에 준하는 중대한 사유가 발생하여 계약 유지가 곤란하다고 인정되는 경우, 카드사는 본 계약을 해지할 수 있습니다.
        """
        })

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.hitl_pending:
        current_iteration = st.session_state.current_state.get('iteration', 1)
        used_threshold = st.session_state.current_state.get('similarity_threshold', SIMILARITY_THRESHOLD)
        
        if SHOW_RETRIEVED_CASES:
            with st.expander("참고한 유사 사례 보기", expanded=False):
                cases = st.session_state.current_state.get('retrieved_cases_metadata', [])
                
                if cases:
                    st.caption(f"총 {len(cases)}개 사례 (유사도 {used_threshold:.0%} 이상)")
                    
                    for case in cases:
                        similarity = case['similarity']
                        
                        if similarity >= 0.7:
                            color = "🟢"
                        elif similarity >= 0.5:
                            color = "🟡"
                        else:
                            color = "🟠"
                        
                        st.markdown(f"### {color} 사례 {case['index']} - 유사도: {similarity:.1%}")
                        st.caption(f"📅 {case['date']} | 유형: {case['case_type']}")
                        
                        with st.container():
                            st.markdown("**약관 조항:**")
                            st.info(case['content'].split('결론:')[0].replace('약관: ', '').strip())
                            
                            if case['explanation']:
                                st.markdown("**시정 요청 사유:**")
                                st.warning(case['explanation'])
                            
                            if case['conclusion']:
                                st.markdown("**최종 결론:**")
                                st.success(case['conclusion'])
                            
                            if case['related_law']:
                                st.caption(f"🔗 관련법: {case['related_law']}")
                        
                        st.divider()
                else:
                    st.warning("검색된 사례가 없습니다.")
        
        st.info(f"개선안 (반복 {current_iteration}/{MAX_ITERATIONS})에 대한 피드백을 주세요.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("수정 요청 (Modify)")
            modify_reason = st.text_area("수정 요청 사유:", key="modify_reason_input")
            
            if current_iteration >= MAX_ITERATIONS:
                st.warning(f"반복 횟수 제한({MAX_ITERATIONS}회)에 도달하여 더 이상 수정 요청을 할 수 없습니다.")
                if st.button("현재 개선안 수락 (Approve)", use_container_width=True, type="primary"):
                    st.session_state.pending_feedback = {
                        "user_feedback": "approved",
                        "modify_reason": "반복 횟수 제한 도달",
                        "retry_action": ""
                    }
                    st.session_state.hitl_pending = False
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "[피드백] 반복 초과로 현재 개선안을 수락합니다."
                    })
                    st.rerun()
            else:
                if st.button("수정 요청 제출 (Modify)", key="modify_btn", use_container_width=True):
                    if not modify_reason.strip():
                        st.error("수정 요청 사유를 반드시 입력해야 합니다.")
                    else:
                        st.session_state.pending_feedback = {
                            "user_feedback": "modify",
                            "modify_reason": modify_reason.strip(),
                            "retry_action": ""
                        }
                        st.session_state.hitl_pending = False
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": f"[피드백] 수정 요청:\n{modify_reason.strip()}"
                        })
                        st.rerun()

        with col2:
            st.subheader("수락 또는 거절 (Approve / Reject)")
            if st.button("개선안 수락 (Approve)", key="approve_btn", use_container_width=True):
                st.session_state.pending_feedback = {
                    "user_feedback": "approved",
                    "modify_reason": "",
                    "retry_action": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "[피드백] 개선안을 수락합니다 (완료)."
                })
                st.rerun()

            if st.button("다른 개선안 생성 (Reject + Retry)", key="retry_btn", use_container_width=True):
                st.session_state.pending_feedback = {
                    "user_feedback": "rejected",
                    "retry_action": "retry",
                    "modify_reason": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "[피드백] 거절 (다른 개선안 재시도)."
                })
                st.rerun()

            if st.button("폐기 (Reject + Discard)", key="discard_btn", use_container_width=True):
                st.session_state.pending_feedback = {
                    "user_feedback": "rejected",
                    "retry_action": "discard",
                    "modify_reason": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "[피드백] 거절 (검토 폐기)."
                })
                st.rerun()
        
        st.chat_input("피드백을 먼저 완료해주세요.", disabled=True)

    else:
        if st.session_state.pending_feedback is not None:
            feedback_input = st.session_state.pending_feedback
            st.session_state.pending_feedback = None
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.chat_message("assistant"):
                with st.spinner("피드백을 반영하여 처리 중..."):
                    try:
                        output = app.invoke(feedback_input, config=config)
                        st.session_state.current_state = output
                        
                        last_feedback = output.get('user_feedback', '')
                        last_retry = output.get('retry_action', '')

                        if last_feedback == "approved" or (last_feedback == "rejected" and last_retry == "discard"):
                            st.markdown("### 검토 완료\n검토가 최종 완료되었습니다.")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "검토가 완료되었습니다."
                            })
                            st.rerun()
                        else:
                            st.markdown(f"### 🔄 새로운 개선안 (반복 {output.get('iteration', '?')}/{MAX_ITERATIONS})")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            st.session_state.hitl_pending = True
                            st.rerun()

                    except Exception as e:
                        st.error(f"피드백 처리 중 오류 발생: {e}")
                        st.session_state.hitl_pending = False
                        st.session_state.thread_id = None
                        st.session_state.current_state = {}

        elif prompt := st.chat_input("검토할 약관 조항을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("약관 조항을 분석 중입니다..."):
                    try:
                        st.session_state.thread_id = f"session_{datetime.now().timestamp()}"
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        initial_state = {
                            "clause": prompt,
                            "iteration": 1,
                            "session_id": st.session_state.thread_id,
                            "validation_failed": False,
                            "retrieved_cases_metadata": [],
                            "retrieved_laws_metadata": [],
                            "similarity_threshold": current_threshold_value
                        }
                        
                        with tracing_v2_enabled():
                            output = app.invoke(initial_state, config=config)
                        
                        st.session_state.current_state = output
                        
                        if output.get('validation_failed', False):
                            error_msg = f"입력 오류: {output.get('cleaned_text', '알 수 없는 오류')}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            st.session_state.thread_id = None
                        else:
                            st.markdown("### 제안 (첫 번째 개선안)")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            st.session_state.hitl_pending = True
                            st.rerun()

                    except Exception as e:
                        st.error(f"약관 검토 중 오류 발생: {e}")
                        import traceback
                        st.exception(traceback.format_exc())
                        st.session_state.thread_id = None
                        st.session_state.hitl_pending = False
                        st.session_state.current_state = {}

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
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        # 2. 텍스트 분할 (Chunking)
        chunks = split_text_into_clauses(pdf_text)
        
        st.markdown(f"총 {len(chunks)}개의 조항(Chunk)이 감지되었습니다.")
        
        if st.button("전체 조항 분석 시작하기", type="primary", key="batch_start_btn"):
            # 3. vectorstore를 run_batch_analysis로 전달
            run_batch_analysis(app, chunks, current_threshold_value, vectorstore)

if __name__ == "__main__":
    main_chatbot_ui()