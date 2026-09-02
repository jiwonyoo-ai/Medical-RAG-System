```python
import os
import json
import re
from dotenv import load_dotenv

# PDF 로딩, 텍스트 분할, 벡터 검색에 필요한 LangChain 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 임베딩 모델과 LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 프롬프트와 LLM 출력 처리
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================
# 기본 설정
# =========================

load_dotenv()

DATA_DIR = "./data"
FAISS_DIR = "./FAISS_dB"

# 원래 RAG 시스템에서 사용한 모델과 동일하게 설정
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# PDF 하나당 생성할 평가 질문 수
QUESTIONS_PER_PDF = 3

# Retrieval에서 검색할 문서 개수
K = 3


# =========================
# 1. PDF 읽기
# =========================

def load_pdfs():

    # 모든 PDF의 페이지를 하나의 리스트에 저장
    all_pages = []

    # data 폴더 안의 PDF 파일 탐색
    pdf_files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(
            f"{DATA_DIR} 폴더에 PDF 파일이 없습니다."
        )

    print(f"\n발견된 PDF: {len(pdf_files)}개")

    # 각 PDF를 읽어서 페이지 단위 Document로 변환
    for pdf_file in pdf_files:

        print(f"PDF 읽는 중: {pdf_file}")

        path = os.path.join(DATA_DIR, pdf_file)

        loader = PyPDFLoader(path)
        pages = loader.load()

        # 읽은 페이지들을 전체 리스트에 추가
        all_pages.extend(pages)

    return all_pages


# =========================
# 2. 문서 분할
# =========================

def split_documents(pages):

    # 원래 RAG 코드와 동일한 방식으로 문서를 chunk로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0
    )

    split_docs = splitter.split_documents(pages)

    print(f"\n전체 문서 수: {len(split_docs)}개 chunk")

    return split_docs


# =========================
# 3. 기존 FAISS 불러오기
# =========================

def load_vectorstore():

    index_path = os.path.join(
        FAISS_DIR,
        "index.faiss"
    )

    # 평가 대상이 되는 기존 FAISS가 존재하는지 확인
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "FAISS_dB/index.faiss가 없습니다.\n"
            "먼저 의료AI.py를 실행해서 FAISS를 생성해주세요."
        )

    print("\n기존 FAISS 불러오는 중...")

    # 원래 RAG에서 사용한 것과 동일한 임베딩 모델 사용
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # 기존에 저장해둔 FAISS 벡터 DB 로드
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS 로드 완료")

    return vectorstore


# =========================
# 4. 평가용 Benchmark 생성
# =========================

def generate_benchmark(split_docs):

    print("\n==============================")
    print("평가용 Benchmark 생성")
    print("==============================")

    # 문서 내용을 바탕으로 평가 질문과 기준 정답을 생성할 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    benchmark = []

    # PDF별로 chunk들을 다시 묶음
    # → 각 질문이 어느 PDF에서 나온 것인지 알 수 있도록 하기 위함
    documents_by_source = {}

    for doc in split_docs:

        source = os.path.basename(
            doc.metadata.get("source", "")
        )

        if source not in documents_by_source:
            documents_by_source[source] = []

        documents_by_source[source].append(
            doc.page_content
        )

    # PDF별로 평가 질문 생성
    for source, contents in documents_by_source.items():

        # PDF의 chunk들을 하나의 텍스트로 결합
        text = "\n\n".join(contents)

        # LLM 입력이 지나치게 길어지는 것을 방지
        text = text[:12000]

        print(f"\n질문 생성 중: {source}")

        # 문서에 근거한 질문과 기준 정답을 생성하도록 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
당신은 의료 문서 기반 QA benchmark를 만드는 사람입니다.

주어진 의료 문서의 내용만 사용하여
평가용 질문과 정답을 만들어주세요.

규칙:
1. 질문은 실제 의료 지식을 확인할 수 있어야 합니다.
2. 문서에서 답을 찾을 수 있어야 합니다.
3. 정답은 문서 내용에 근거해야 합니다.
4. 너무 단순한 질문보다 핵심 내용을 확인하는 질문을 만드세요.
5. 질문과 정답을 반드시 JSON 배열 형태로 출력하세요.

형식:

[
  {
    "question": "질문",
    "reference_answer": "정답"
  }
]
"""
            ),
            (
                "human",
                f"""
다음은 {source}의 내용입니다.

{text}

이 문서를 기반으로 평가 질문을
{QUESTIONS_PER_PDF}개 만들어주세요.
"""
            )
        ])

        # Prompt → LLM → 문자열 출력
        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({})

        # LLM 응답에서 JSON 배열 부분만 추출
        match = re.search(
            r"\[.*\]",
            result,
            re.DOTALL
        )

        if not match:
            print("질문 생성 실패")
            continue

        try:
            # 문자열 형태의 JSON을 Python 객체로 변환
            questions = json.loads(
                match.group()
            )

        except json.JSONDecodeError:
            print("JSON 변환 실패")
            continue

        # Benchmark에 질문, 기준 정답, 관련 문서 저장
        for q in questions:

            benchmark.append({
                "question": q["question"],
                "reference_answer":
                    q["reference_answer"],
                "relevant_source":
                    source
            })

    print(
        f"\n총 Benchmark 질문: "
        f"{len(benchmark)}개"
    )

    return benchmark


# =========================
# 5. Retrieval 평가
# =========================

def evaluate_retrieval(
    vectorstore,
    benchmark
):

    print("\n==============================")
    print("Retrieval 평가")
    print("==============================")

    success = 0
    results = []

    # Benchmark의 모든 질문에 대해 검색 성능 평가
    for i, item in enumerate(
        benchmark,
        start=1
    ):

        question = item["question"]

        # 해당 질문의 정답이 포함된 기준 문서
        expected_source = os.path.basename(
            item["relevant_source"]
        )

        # 질문과 가장 가까운 상위 K개 문서 검색
        docs_with_scores = (
            vectorstore
            .similarity_search_with_score(
                question,
                k=K
            )
        )

        retrieved_sources = []

        # 검색된 문서의 파일명 확인
        for doc, score in docs_with_scores:

            source = os.path.basename(
                doc.metadata.get(
                    "source",
                    ""
                )
            )

            retrieved_sources.append(
                source
            )

        # 정답 문서가 Top-K 안에 포함되었는지 확인
        hit = (
            expected_source
            in retrieved_sources
        )

        if hit:
            success += 1

        print(
            f"\nQ{i}: {question}"
        )

        print(
            f"정답 문서: {expected_source}"
        )

        print(
            f"검색 문서: {retrieved_sources}"
        )

        print(
            f"결과: {'성공' if hit else '실패'}"
        )

        # 질문별 Retrieval 평가 결과 저장
        results.append({

            "question": question,

            "expected_source":
                expected_source,

            "retrieved_sources":
                retrieved_sources,

            "retrieval_hit":
                hit
        })

    # 정답 문서가 Top-K에 포함된 질문의 비율
    recall = (
        success / len(benchmark)
        if benchmark
        else 0
    )

    print("\n------------------------------")

    print(
        f"Retrieval Recall@{K}: "
        f"{recall:.2%}"
    )

    return recall, results


# =========================
# 6. Generation 평가
# =========================

def evaluate_generation(
    vectorstore,
    benchmark
):

    print("\n==============================")
    print("Generation 평가")
    print("==============================")

    # 실제 답변을 생성하는 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # -------------------------
    # 답변 생성용 Prompt
    # -------------------------

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
당신은 의료 문서 기반 질문에 답변하는 AI입니다.

반드시 제공된 문서 내용을 근거로 답변하세요.
문서에 없는 내용은 추측하지 마세요.

한국어로 답변하세요.

{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ])

    # 검색된 문서 + 질문 → LLM 답변 생성
    answer_chain = (
        answer_prompt
        | llm
        | StrOutputParser()
    )


    # -------------------------
    # 평가용 Prompt
    # -------------------------

    # 기준 정답과 AI 답변을 비교하는 별도의 평가 프롬프트
    judge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
당신은 의료 QA 평가자입니다.

기준 정답과 AI 답변을 비교하여
AI 답변이 핵심 내용을 정확하게 포함하는지 평가하세요.

다음 기준으로 판단하세요.

1점:
핵심 내용이 정확하다.

0점:
핵심 내용이 틀리거나 중요한 내용이 빠졌다.

반드시 0 또는 1 하나만 출력하세요.
"""
        ),
        (
            "human",
            """
질문:
{question}

기준 정답:
{reference_answer}

AI 답변:
{model_answer}
"""
        )
    ])

    # 평가 LLM에게 기준 정답과 AI 답변을 전달
    judge_chain = (
        judge_prompt
        | llm
        | StrOutputParser()
    )

    success = 0
    results = []

    # 모든 Benchmark 질문에 대해 Generation 평가
    for i, item in enumerate(
        benchmark,
        start=1
    ):

        question = item["question"]

        reference_answer = (
            item["reference_answer"]
        )

        # -------------------------
        # 1) 관련 문서 검색
        # -------------------------

        docs = vectorstore.similarity_search(
            question,
            k=K
        )

        # 검색된 문서를 하나의 context로 결합
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # -------------------------
        # 2) AI 답변 생성
        # -------------------------

        model_answer = answer_chain.invoke({
            "context": context,
            "question": question
        })

        # -------------------------
        # 3) 기준 정답과 비교
        # -------------------------

        judge_result = judge_chain.invoke({

            "question": question,

            "reference_answer":
                reference_answer,

            "model_answer":
                model_answer

        }).strip()

        # 평가 LLM이 1을 반환하면 정답으로 처리
        score = (
            1
            if judge_result.startswith("1")
            else 0
        )

        if score == 1:
            success += 1

        print(
            f"\nQ{i}: {question}"
        )

        print(
            f"AI 답변: {model_answer}"
        )

        print(
            f"평가: {'정답' if score else '오답'}"
        )

        # 질문별 Generation 평가 결과 저장
        results.append({

            "question": question,

            "reference_answer":
                reference_answer,

            "model_answer":
                model_answer,

            "generation_score":
                score
        })

    # 전체 질문 중 정답으로 평가된 비율
    accuracy = (
        success / len(benchmark)
        if benchmark
        else 0
    )

    print("\n------------------------------")

    print(
        f"Generation Accuracy: "
        f"{accuracy:.2%}"
    )

    return accuracy, results


# =========================
# 7. 전체 실행
# =========================

def main():

    print("=" * 60)

    print(
        "Medical RAG Benchmark Evaluation"
    )

    print("=" * 60)


    # ① 평가에 사용할 PDF 읽기
    pages = load_pdfs()


    # ② 원래 RAG 코드와 동일한 방식으로 chunk 생성
    split_docs = split_documents(pages)


    # ③ 실제 RAG 시스템에서 사용한 FAISS 로드
    vectorstore = load_vectorstore()


    # ④ 문서 기반 평가용 Benchmark 자동 생성
    benchmark = generate_benchmark(
        split_docs
    )

    if not benchmark:

        print(
            "\nBenchmark 생성에 실패했습니다."
        )

        return


    # ⑤ 생성한 Benchmark 저장
    with open(
        "benchmark_auto.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            benchmark,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nbenchmark_auto.json 저장 완료"
    )


    # ⑥ Retrieval 성능 평가
    retrieval_score, retrieval_results = (
        evaluate_retrieval(
            vectorstore,
            benchmark
        )
    )


    # ⑦ Generation 성능 평가
    generation_score, generation_results = (
        evaluate_generation(
            vectorstore,
            benchmark
        )
    )


    # ⑧ Retrieval + Generation 결과를 하나로 정리
    final_result = {

        "number_of_questions":
            len(benchmark),

        f"retrieval_recall_at_{K}":
            retrieval_score,

        "generation_accuracy":
            generation_score,

        "retrieval_results":
            retrieval_results,

        "generation_results":
            generation_results
    }


    # ⑨ 최종 평가 결과 JSON 저장
    with open(
        "evaluation_result.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_result,
            f,
            ensure_ascii=False,
            indent=2
        )


    # -------------------------
    # 최종 결과 출력
    # -------------------------

    print("\n")

    print("=" * 60)
    print("최종 평가 결과")
    print("=" * 60)

    print(
        f"Retrieval Recall@{K}: "
        f"{retrieval_score:.2%}"
    )

    print(
        f"Generation Accuracy: "
        f"{generation_score:.2%}"
    )

    print(
        "\n상세 결과:"
    )

    print(
        "benchmark_auto.json"
    )

    print(
        "evaluation_result.json"
    )


# 프로그램 실행
if __name__ == "__main__":
    main()
```
