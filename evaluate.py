import os
import json
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# 기본 설정
# =========================

load_dotenv()

DATA_DIR = "./data"
FAISS_DIR = "./FAISS_dB"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# PDF 하나당 생성할 평가 문제 수
QUESTIONS_PER_PDF = 3

# 검색할 문서 개수
K = 3


# =========================
# 1. PDF 읽기
# =========================

def load_pdfs():

    all_pages = []

    pdf_files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(
            f"{DATA_DIR} 폴더에 PDF 파일이 없습니다."
        )

    print(f"\n발견된 PDF: {len(pdf_files)}개")

    for pdf_file in pdf_files:

        print(f"PDF 읽는 중: {pdf_file}")

        path = os.path.join(DATA_DIR, pdf_file)

        loader = PyPDFLoader(path)

        pages = loader.load()

        all_pages.extend(pages)

    return all_pages


# =========================
# 2. 문서 분할
# =========================

def split_documents(pages):

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

    if not os.path.exists(index_path):

        raise FileNotFoundError(
            "FAISS_dB/index.faiss가 없습니다.\n"
            "먼저 의료AI.py를 실행해서 FAISS를 생성해주세요."
        )

    print("\n기존 FAISS 불러오는 중...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS 로드 완료")

    return vectorstore


# =========================
# 4. 평가용 질문 자동 생성
# =========================

def generate_benchmark(split_docs):

    print("\n==============================")
    print("평가용 Benchmark 생성")
    print("==============================")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    benchmark = []

    # PDF별로 묶기
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

    for source, contents in documents_by_source.items():

        # 너무 길어지는 것을 방지
        text = "\n\n".join(contents)

        text = text[:12000]

        print(f"\n질문 생성 중: {source}")

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

        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({})

        # JSON 부분만 추출
        match = re.search(
            r"\[.*\]",
            result,
            re.DOTALL
        )

        if not match:
            print("질문 생성 실패")
            continue

        try:

            questions = json.loads(
                match.group()
            )

        except json.JSONDecodeError:

            print("JSON 변환 실패")
            continue

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

    for i, item in enumerate(
        benchmark,
        start=1
    ):

        question = item["question"]

        expected_source = os.path.basename(
            item["relevant_source"]
        )

        docs_with_scores = (
            vectorstore
            .similarity_search_with_score(
                question,
                k=K
            )
        )

        retrieved_sources = []

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

        results.append({

            "question": question,

            "expected_source":
                expected_source,

            "retrieved_sources":
                retrieved_sources,

            "retrieval_hit":
                hit
        })

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

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # 답변 생성용 prompt
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

    answer_chain = (
        answer_prompt
        | llm
        | StrOutputParser()
    )

    # 평가용 LLM
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

    judge_chain = (
        judge_prompt
        | llm
        | StrOutputParser()
    )

    success = 0

    results = []

    for i, item in enumerate(
        benchmark,
        start=1
    ):

        question = item["question"]

        reference_answer = (
            item["reference_answer"]
        )

        # 문서 검색
        docs = vectorstore.similarity_search(
            question,
            k=K
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # AI 답변 생성
        model_answer = answer_chain.invoke({
            "context": context,
            "question": question
        })

        # 정답과 비교
        judge_result = judge_chain.invoke({

            "question": question,

            "reference_answer":
                reference_answer,

            "model_answer":
                model_answer
        }).strip()

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

        results.append({

            "question": question,

            "reference_answer":
                reference_answer,

            "model_answer":
                model_answer,

            "generation_score":
                score
        })

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

    # PDF 자동 읽기
    pages = load_pdfs()

    # 기존 코드와 동일하게 chunk 생성
    split_docs = split_documents(pages)

    # 기존 FAISS 사용
    vectorstore = load_vectorstore()

    # Benchmark 자동 생성
    benchmark = generate_benchmark(
        split_docs
    )

    if not benchmark:

        print(
            "\nBenchmark 생성에 실패했습니다."
        )

        return

    # Benchmark 저장
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

    # Retrieval 평가
    retrieval_score, retrieval_results = (
        evaluate_retrieval(
            vectorstore,
            benchmark
        )
    )

    # Generation 평가
    generation_score, generation_results = (
        evaluate_generation(
            vectorstore,
            benchmark
        )
    )

    # 최종 결과
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


if __name__ == "__main__":
    main()