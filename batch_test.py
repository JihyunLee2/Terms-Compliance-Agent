import pandas as pd
pd.set_option('display.max_columns', None)  # 모든 컬럼 표시
import json
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langchain_core.tracers.context import tracing_v2_enabled
from config2 import *
from prompts import (
    ACTIVE_FAIRNESS_CLASSIFY_PROMPT,
    ACTIVE_UNFAIR_TYPE_PROMPT,
    ACTIVE_GENERATE_PROPOSAL_PROMPT
)
import re
from tqdm import tqdm
from collections import Counter 
import argparse 
import contextlib 

FAIRNESS_MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.8

load_dotenv()

embeddings = UpstageEmbeddings(model=EMBEDDING_MODEL)
llm = ChatUpstage(model=LLM_MODEL)

def load_vectordb():
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name="contract_laws",
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def clean_text(text: str) -> str:
    cleaned = re.sub(r'^[\s•\-\*]+', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', cleaned)
    cleaned = re.sub(r'\(\d+\)\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def prepare_full_clause(clause: str, context: str = None) -> str:
    if context and pd.notna(context) and context.strip():
        return f"[참조 조항]\n{context}\n\n[본 조항]\n{clause}"
    return clause

def safe_get_value(row, key, default="N/A"):
    val = row.get(key, None)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, str) and val.strip() == "":
        return default
    return val

# def classify_fairness(clause: str, langsmith_project: str) -> str:
#     prompt = ACTIVE_FAIRNESS_PROMPT.format(clause=clause)
#     with tracing_v2_enabled(
#         project_name=langsmith_project, # "contract-review-batch",
#         tags=["fairness_classification", "batch_test"]
#     ):
#         result = llm.invoke(prompt).content.strip()
        
#         # 첫 줄만 추출
#         result = result.split('\n')[0].strip()
#         # '공정' 또는 '불공정'만 추출
#         if '공정' in result and '불공정' not in result:
#             return '공정'
#         elif '불공정' in result:
#             return '불공정'
#     return result

def classify_fairness(clause: str, langsmith_project: str, enable_tracing: bool = True) -> str:
    """
    [수정된 함수]
    공정/불공정 분류 (최대 3회 반복 및 총합)
    1. 3회 반복. CONFIDENCE_THRESHOLD(0.8) 이상 점수 나오면 "즉시 종료" (Fast Path)
    2. 3회 모두 CONFIDENCE_THRESHOLD(0.8) 미만이면, "신뢰도 점수 총합"으로 결정 (Fallback)
    """
    
    # 'results_history'와 'score_sums'는 이 노드에서 새로 시작
    results_history = [] 
    score_sums = {'공정': 0.0, '불공정': 0.0} 
    
    # [수정] 트래킹 활성화 여부에 따라 컨텍스트 분기
    tracer = tracing_v2_enabled(
        project_name=langsmith_project,
        tags=["fairness_classification", "batch_test"]
    ) if enable_tracing else contextlib.nullcontext()
    
    print(f"\n[노드2] 공정/불공정 분류 시작 (최대 {FAIRNESS_MAX_ITERATIONS}회)\n")

    for iteration in range(1, FAIRNESS_MAX_ITERATIONS + 1):
        # 인자로 받은 'clause'를 프롬프트에 사용
        prompt = ACTIVE_FAIRNESS_CLASSIFY_PROMPT.format(cleaned_text=clause)

        try:
            with tracer:
                # langsmith 1. 실행(run)에 지정할 동적 이름 생성
                run_name_str = f"fairness_check_iter_{iteration}" 
                # langsmith 2. .with_config()를 사용하여 run_name 전달
                result = llm.with_config({"run_name": run_name_str}).invoke(prompt).content.strip()
            
            # --- '파싱 로직' ---
            lines = result.split("\n")
            if len(lines) < 2:
                print(f"  [경고] {iteration}차 시도: 신뢰도 점수 미반환. ({lines[0]})")
                classification_text = lines[0].strip()
                confidence = 0.0 # 신뢰도 없으면 0.0으로 처리
            else:
                classification_text = lines[0].strip()
                confidence = float(lines[1].strip())
            
            # '공정' 또는 '불공정' 라벨 추출
            label = 'N/A' # 기본값
            if '공정' in classification_text and '불공정' not in classification_text:
                label = '공정'
            elif '불공정' in classification_text:
                label = '불공정'
            
            if label == 'N/A':
                print(f"  [경고] {iteration}차 시도: '공정'/'불공정' 식별 불가. ({classification_text})")
                continue # 유효한 분류가 아니면 다음 시도

            # --- (결과 기록) ---
            results_history.append((label, confidence))
            print(f"  [{iteration}차 시도] 결과: {label}, 신뢰도: {confidence:.2f}") ######
            
            # --- 1. "Fast Path" (신뢰도가 임계값을 넘으면 즉시 종료)  (배치 효율성) ---
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"   [즉시 확정] {iteration}차: {label} ({confidence:.2f})")
                # return label # 점수가 높으므로 즉시 종료 및 '공정' 또는 '불공정' 문자열 반환
                # 상세 로깅을 위해 딕셔너리 반환
                return {
                    "final_label": label,
                    "final_confidence": confidence,
                    "method": "Fast Path",
                    "iterations_run": iteration,
                    "threshold_used": CONFIDENCE_THRESHOLD,
                    "all_iteration_results": results_history,
                    "score_sums": score_sums # (참고용)
                }

            # --- 2. "Fallback Path" (점수 누적) ---
            if label in score_sums:
                score_sums[label] += confidence
            
        except Exception as e:
            print(f"  [오류] {iteration}차 시도 중 예외 발생: {e}")
            results_history.append(('N/A', 0.0)) 
        
    # --- (Loop가 모두 끝남: "즉시 종료" 실패) ---
    # --- 3. "Fallback Path" (패자부활전) 로직 실행 ---
    print(f"  [패자부활전] 3회 모두 임계값 미달. 총합으로 결정.")
    print(f"[노드2] 공정/불공정 분류 결과: {classification_text}, 확신도: {confidence}, 반복: {iteration}")
    
    avg_confidence = 0.0
    
    # 3번의 시도에서 유효한 분류가 하나도 없었는지 확인
    if score_sums['공정'] == 0.0 and score_sums['불공정'] == 0.0:
        final_classification = "분류 실패"
    else:
        # 1. 최종 라벨 결정 (신뢰도 총합이 높은 쪽)
        final_classification = max(score_sums, key=score_sums.get)
        
        # 2. 최종 선택된 라벨 총 점수
        total_confidence = score_sums[final_classification]
        
        # 3. 최종 선택된 라벨이 득표한 횟수 (0으로 나누기 방지)
        count = sum(1 for label, _ in results_history if label == final_classification)
        
        if count > 0:
            # 4. 최종 선택된 라벨의 '평균 신뢰도' 계산
            avg_confidence = total_confidence / count

    print(f"[최종 판단] 공정성 분류: {final_classification}, 평균 확신도: {avg_confidence:.2f}, score_sum: {score_sums}")
    
    # 상세 로깅을 위해 딕셔너리 반환
    return {
        "final_label": final_classification,
        "final_confidence": avg_confidence, # 0.0일 수 있음
        "method": "Fallback Sum" if final_classification != "분류 실패" else "Fallback Failed",
        "iterations_run": FAIRNESS_MAX_ITERATIONS,
        "threshold_used": CONFIDENCE_THRESHOLD,
        "all_iteration_results": results_history,
        "score_sums": score_sums
    }
    
def classify_unfair_type(clause: str, fairness: str, langsmith_project: str, enable_tracing: bool = True) -> str:
    # fairness가 부연 설명 포함한 경우 첫 줄만 추출
    fairness_line = fairness.split('\n')[0].strip()
    if '공정' in fairness_line and '불공정' not in fairness_line:
        return "N/A"
    
    prompt = ACTIVE_UNFAIR_TYPE_PROMPT.format(cleaned_text=clause)
    
    # [수정] 트래킹 활성화 여부에 따라 컨텍스트 분기
    tracer = tracing_v2_enabled(
        project_name=langsmith_project,
        tags=["unfair_type_classification", "batch_test"]
    ) if enable_tracing else contextlib.nullcontext()

    with tracer:
        # [수정] run_name 추가
        result = llm.with_config({"run_name": "unfair_type_check"}).invoke(prompt).content.strip()
                
        # 여러 유형이 출력된 경우 첫 번째만 추출
        # 패턴: "불공정(숫자. 유형명)"을 찾아서 첫 번째만 반환
        pattern = r'불공정\(\d+\.\s*[^)]+\)'
        matches = re.findall(pattern, result)
        
        if matches:
            return matches[0]  # 첫 번째 유형만 반환
        
        # 패턴이 없으면 첫 줄만 반환
        return result.split('\n')[0].strip()

def retrieve_cases_and_laws(vectorstore, clause: str, unfair_type: str, threshold: float = SIMILARITY_THRESHOLD):
    if unfair_type == "N/A":
        return [], []
    search_query = f"{unfair_type} {clause}"
    results_cases = vectorstore.similarity_search_with_relevance_scores(
        search_query, k=SEARCH_TOP_K_CASES, filter={"source_type": "case"}
    )
    filtered_cases = [
        {"similarity": score, "content": doc.page_content, "metadata": doc.metadata}
        for doc, score in results_cases if score >= threshold
    ][:MAX_DISPLAY_CASES]
    law_query = " ".join([c['metadata'].get('related_law', '') for c in filtered_cases if c['metadata'].get('related_law')])
    if not law_query:
        law_query = search_query
    results_laws = vectorstore.similarity_search_with_relevance_scores(
        law_query, k=SEARCH_TOP_K_LAWS, filter={"source_type": "law"}
    )
    filtered_laws = [
        {"similarity": score, "content": doc.page_content}
        for doc, score in results_laws if score >= threshold
    ][:MAX_DISPLAY_LAWS]
    return filtered_cases, filtered_laws

def generate_improvement_proposal(clause: str, unfair_type: str, cases: List[Dict], laws: List[Dict], langsmith_project: str, enable_tracing: bool = True) -> str:
    if unfair_type == "N/A":
        return "This clause is considered fair and does not require improvement."
    cases_text = "\n".join([f"- {c['content']}" for c in cases])
    laws_text = "\n".join([f"- {l['content']}" for l in laws])
    prompt = ACTIVE_GENERATE_PROPOSAL_PROMPT.format(
        original_clause=clause,
        unfair_type=unfair_type,
        related_cases=cases_text,
        related_laws=laws_text,
        feedback_context=""
    )
    
    # [수정] 트래킹 활성화 여부에 따라 컨텍스트 분기
    tracer = tracing_v2_enabled(
        project_name=langsmith_project,
        tags=["improvement_generation", "batch_test"]
    ) if enable_tracing else contextlib.nullcontext()

    with tracer:
        # [수정] run_name 추가
        result = llm.with_config({"run_name": "proposal_generation"}).invoke(prompt).content
    return result   

def batch_process_test_data(
    csv_file: str, 
    output_file: str = 'batch_test_results_251116.jsonl', 
    # error_file: str = 'batch_test_errors_251116.jsonl', # [수정] 오류 파일 경로 추가
    langsmith_project: str = "contract-review-batch",
    enable_tracing: bool = True, # [수정] 트래킹 플래그 추가
    scope: str = 'full' # <-- [추가] 1. scope 파라미터 추가 (기본값 'full')
):

    print(f"Loading test data: {csv_file}")
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"Total records: {len(df)}")
    print(f"Columns detected: {df.columns.tolist()}\n")
    if 'clause' not in df.columns:
        print("'clause' column missing. Please check CSV.")
        return
    has_context = 'context' in df.columns
    print(f"Context column present: {has_context}\n")
    vectorstore = load_vectordb()
    results = []
    
    print(f"LangSmith project: {langsmith_project}")
    print(f"LangSmith tracing enabled: {enable_tracing}\n")
    
    # [수정] 실행 시마다 결과 파일/오류 파일 초기화
    print(f"Resetting output files: {output_file}")
    # print(f"Resetting output files: {output_file}, {error_file}")
    open(output_file, 'w').close()
    # open(error_file, 'w').close()
    
    print("Starting batch processing...\n")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            clause_raw = row['clause']
            context = row.get('context', None) if has_context else None
            full_clause = prepare_full_clause(clause_raw, context)
            cleaned = clean_text(full_clause)
            
            # Ground Truth 값 추출
            gt_fairness = safe_get_value(row, 'label', default="N/A")
            gt_type = safe_get_value(row, 'ground_truth_type', default="N/A")
            gt_improvement = safe_get_value(row, 'ground_truth_improvement', default="N/A")
            
            # --- LLM 파이프라인 실행 ---
            # ---  1. 공정성 분류 (상세 결과 받기) ---
            fairness_details = classify_fairness(cleaned, langsmith_project, enable_tracing)
            fairness_label = fairness_details["final_label"] # '공정' 또는 '불공정' 문자열
            # fairness = classify_fairness(cleaned, langsmith_project, enable_tracing)
            
            if fairness_label == '공정':
                unfair_type = "N/A"
                improvement = None
                cases = []
                laws = []
            else:
                # --- 2. 불공정 유형 분류 ---
                unfair_type = classify_unfair_type(cleaned, fairness_label, langsmith_project, enable_tracing)

            # --- [수정] 3. scope에 따라 분기 ---
                if scope == 'classify_only':
                    # 'classify_only' 모드이면 RAG와 개선안 생성을 건너뜁니다.
                    improvement = None
                    cases = []
                    laws = []
                else:
                    # 'full' 모드 (기본값)일 때만 RAG와 개선안을 실행합니다.
                    cases, laws = retrieve_cases_and_laws(vectorstore, cleaned, unfair_type)
                    improvement = generate_improvement_proposal(cleaned, unfair_type, cases, laws, langsmith_project, enable_tracing)

            # --- 정답 비교 로직 (fairness_label 사용) ---
            is_correct_fairness = (gt_fairness == fairness_label)
            is_correct_type = (gt_type == "N/A" or gt_type == unfair_type)
            is_correct_overall = is_correct_fairness and is_correct_type
            # ---------------------------
            
            # ---  오류 원인 기록 ---
            error_reason_list = []
            if not is_correct_fairness:
                error_reason_list.append("공정성(1번) 분류 틀림")
            if not is_correct_type:
                error_reason_list.append("불공정 유형(2번) 분류 틀림")
            error_reason_str = ", ".join(error_reason_list) if error_reason_list else None
            # ------------------------------------
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "index": int(idx),
                
                # 비교 결과
                "is_correct_fairness": is_correct_fairness,
                "is_correct_type": is_correct_type,
                "is_correct_overall": is_correct_overall,
                "error_reason": error_reason_str, # [수정] 오류 원인 필드 추가
                
                # 예측 값
                "predicted_fairness": fairness_label,
                # --- 상세 로깅 결과 추가 ---
                "fairness_details": {
                    "final_confidence": fairness_details["final_confidence"],
                    "method": fairness_details["method"],
                    "iterations_run": fairness_details["iterations_run"],
                    "threshold": fairness_details["threshold_used"],
                    "all_iteration_results": fairness_details["all_iteration_results"],
                    "score_sums": fairness_details["score_sums"]
                },
                # -------------------------------------
                
                "predicted_type": unfair_type,
                "predicted_improvement": improvement,
                
                # 정답 (Ground Truth)
                "label": gt_fairness,
                "ground_truth_type": gt_type,
                "ground_truth_improvement": gt_improvement,
                
                # 입력 값
                "original_clause": clause_raw,
                "context": context if (context and pd.notna(context)) else None,
                "full_clause": full_clause,
                "cleaned_text": cleaned,
                
                # RAG 결과
                "retrieved_cases_count": len(cases),
                "retrieved_laws_count": len(laws),
                "retrieved_cases": [c['content'] for c in cases],
                "retrieved_laws": [l['content'] for l in laws]
            }
            results.append(result)
            
            # 1. 전체 결과 파일에 쓰기
            # with open(output_file, 'a', encoding='utf-8') as f:
            #     f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            # 2. 틀린 경우, 오류 파일에 따로 쓰기
            # if not is_correct_overall:
            #     with open(error_file, 'a', encoding='utf-8') as f:
            #         f.write(json.dumps(result, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"\nError processing index {idx}: {e}")
            error_result = {
                "timestamp": datetime.now().isoformat(),
                "index": int(idx),
                "original_clause": clause_raw,
                "error": str(e),
                "predicted_fairness": None,
                "fairness_details": None, # 상세 로깅 필드 (에러 시 null)
                "predicted_type": None,
                "predicted_improvement": None,
                "label": gt_fairness,
                "ground_truth_type": gt_type,
                "ground_truth_improvement": gt_improvement,
                "is_correct_overall": False, # 에러는 '틀린 것'으로 간주
                "error_reason": "런타임 에러" # [수정] 런타임 에러 명시
            }
            results.append(error_result)
            
            # 에러도 기록
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
            # with open(error_file, 'a', encoding='utf-8') as f:
            #     f.write(json.dumps(error_result, ensure_ascii=False) + '\n')

    print(f"\nBatch processing finished!")
    print(f"All results saved to: {output_file}")
    # print(f"Errors/Incorrect saved to: {error_file}")
    
    return results

if __name__ == "__main__":
    # [수정] sys.argv 대신 argparse 사용
    parser = argparse.ArgumentParser(description="Batch process contract clauses for fairness.")
    parser.add_argument("csv_file", type=str, help="Path to the test CSV file.")
    parser.add_argument(
        "project_name", 
        type=str, 
        nargs='?', # ?는 0개 또는 1개를 의미
        default="contract-review-batch", 
        help="LangSmith project name (default: contract-review-batch)"
    )
    parser.add_argument(
        "--no-trace", 
        action="store_true", 
        help="Disable LangSmith tracing."
    )
    parser.add_argument(
        "--scope", 
        type=str, 
        choices=['full', 'classify_only'], # 2가지 옵션
        default='full', 
        help="Test scope: 'full' (all steps) or 'classify_only' (classification only)."
    )
    
    args = parser.parse_args()
    
    # [수정] --no-trace 플래그에 따라 enable_tracing 설정
    enable_tracing = not args.no_trace
    
    batch_process_test_data(
        args.csv_file, 
        langsmith_project=args.project_name, 
        enable_tracing=enable_tracing,
        scope=args.scope 
    )
    
    # 1번(공정), 2번(유형) 분류만 테스트 (RAG, 개선안 생성 X)
    # python batch_test.py test.csv batch-test --scope classify_only

    # (기존 방식) 모든 파이프라인 실행 (기본값)
    # python batch_test.py test.csv batch-test --scope full

    # (--scope full은 기본값이므로 생략 가능)
    # python batch_test.py test.csv batch-test

    # [수정] 실행 명령어 예시
    # (트래킹 O) python batch_test.py test.csv batch-test
    # (트래킹 X) python batch_test.py test.csv batch-test --no-trace