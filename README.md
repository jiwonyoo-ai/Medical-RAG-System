# Medical-LLM-Assistant 

> A retrieval-augmented question answering system that retrieves relevant information from medical documents and generates context-aware answers using GPT-4o.

본 프로젝트는 **의료 문서를 외부 지식으로 구축하고, 사용자 질의와 관련된 정보를 검색하여 GPT-4o의 context로 활용하는 RAG (Retrieval-Augmented Generation) 기반 질의응답 시스템**입니다.

의료 PDF 문서를 전처리하고 text embedding과 FAISS 기반 벡터 검색을 적용하여 사용자 질의와 관련된 문서를 검색한 뒤, 검색 결과를 GPT-4o에 전달하여 답변을 생성하는 **End-to-End retrieval and generation pipeline**을 구현했습니다.

또한 검색 결과와 생성 결과를 평가하여 **문서 검색 성능과 최종 답변의 품질을 분리하여 분석**했습니다.

---

## Project Overview

### Motivation

LLM 기반 질의응답 시스템에서는 모델의 사전 학습 지식에만 의존하기보다, 외부 문서에서 관련 정보를 검색하고 이를 답변 생성에 활용하는 과정이 중요합니다.

본 프로젝트에서는 MSD Manual 기반 의료 문서를 검색 가능한 지식 데이터로 구축하고, 사용자의 질의를 embedding space로 변환하여 관련 문서를 검색하는 **RAG pipeline**을 구현했습니다.

검색된 문서는 GPT-4o의 context로 전달되며, 이를 기반으로 사용자의 질문에 대한 답변을 생성하도록 구성했습니다.

---

## Objectives

* 의료 PDF 문서 전처리 및 검색 데이터 구축
* 문서 chunking 및 embedding 생성
* FAISS 기반 vector similarity search 구현
* 사용자 질의 기반 관련 문서 검색
* 검색 결과를 활용한 GPT-4o 답변 생성
* Retrieval과 Generation 과정의 분리된 평가
* 검색 결과 및 응답 품질 분석

---

## System Architecture

```text
Medical PDF Documents
        ↓
Document Processing
        ↓
Text Splitting
        ↓
Text Embedding
        ↓
FAISS Vector Store
        ↓
User Query
        ↓
Query Embedding
        ↓
Similarity Search
        ↓
Retrieved Documents
        ↓
Context Construction
        ↓
GPT-4o
        ↓
Generated Answer
        ↓
Evaluation
```

---

## Workflow

### 1. Medical Document Processing

MSD Manual 기반 의료 PDF 문서를 질의응답에 활용할 수 있도록 전처리했습니다.

`PyPDFLoader`와 `PyMuPDF`를 이용하여 PDF에서 텍스트를 추출하고, 검색 가능한 문서 데이터로 변환했습니다.

---

### 2. Text Splitting

전처리된 문서를 검색 단위인 text chunk로 분할했습니다.

각 chunk가 독립적인 검색 단위로 활용될 수 있도록 문서 구조를 변환하고, 이후 embedding 및 vector search에 사용할 수 있도록 구성했습니다.

---

### 3. Embedding & Vector Store

OpenAI의 `text-embedding-3-small` 모델을 사용하여 각 text chunk를 벡터로 변환했습니다.

생성된 embedding은 **FAISS Vector Store**에 저장하여 사용자 질의와 문서 간의 의미적 유사도를 기반으로 관련 문서를 검색할 수 있도록 구현했습니다.

```text
Document Chunk
      ↓
text-embedding-3-small
      ↓
Embedding Vector
      ↓
FAISS
```

---

### 4. Query-based Retrieval

사용자의 질문을 embedding으로 변환한 후 FAISS를 이용하여 관련성이 높은 문서를 검색합니다.

```text
User Query
    ↓
Query Embedding
    ↓
FAISS Similarity Search
    ↓
Top-K Relevant Documents
```

검색 과정에서는 `similarity_search_with_score`를 활용하여 검색된 문서와 질의 간의 유사도 정보를 함께 확인할 수 있도록 구성했습니다.

---

### 5. Context Construction

검색된 문서를 하나의 context로 구성하여 GPT-4o에 전달합니다.

이를 통해 LLM이 검색된 외부 문서를 참고하여 사용자 질문에 대한 답변을 생성하도록 구성했습니다.

```text
Retrieved Documents
        +
User Query
        ↓
Context Construction
        ↓
GPT-4o
```

---

### 6. LLM-based Answer Generation

GPT-4o에 사용자 질의와 검색된 문서 context를 함께 전달하여 최종 답변을 생성합니다.

이를 통해 **Retrieval → Context Construction → Generation**으로 이어지는 RAG pipeline을 구현했습니다.

---

## Evaluation

본 프로젝트에서는 **Retrieval 성능과 Generation 성능을 분리하여 평가**했습니다.

테스트 질의 87개를 구성하여 검색된 문서가 정답 정보를 포함하는지 확인하고, 검색 결과를 기반으로 생성된 답변의 정확성을 평가했습니다.

### Retrieval Performance

**Recall@3: 87.36% (76 / 87)**

Top-3 검색 결과 안에 정답 정보를 포함하는지 평가했습니다.

### Generation Performance

**Generation Accuracy: 71.26% (62 / 87)**

검색된 context를 기반으로 생성된 최종 답변의 정확성을 평가했습니다.

### Performance Gap

**Retrieval Recall@3: 87.36%**
**Generation Accuracy: 71.26%**

두 단계 사이에서 **16.09%p의 성능 차이**가 나타났으며, 이를 통해 관련 문서를 충분히 검색하더라도 최종 답변 생성 단계에서 추가적인 성능 저하가 발생할 수 있음을 확인했습니다.

---

## Analysis

검색 성능과 최종 답변 성능을 분리하여 평가함으로써 RAG 시스템의 어느 단계에서 성능 저하가 발생하는지 분석했습니다.

특히 높은 Retrieval Recall에 비해 Generation Accuracy가 낮게 나타난 결과를 통해, **문서 검색뿐만 아니라 검색된 context를 효과적으로 활용하는 Generation 과정 역시 RAG 시스템 성능에 중요한 요소**임을 확인했습니다.

---

## My Contributions

* RAG 기반 질의응답 시스템 기획 및 전체 pipeline 설계
* 의료 PDF 문서 전처리 및 검색 데이터 구축
* Text chunking 및 embedding pipeline 구현
* FAISS 기반 vector similarity search 구현
* 검색 결과와 GPT-4o를 연결하는 context 구성 및 prompt 설계
* Retrieval / Generation 성능 평가 방법 설계
* 테스트 데이터셋 기반 검색 및 답변 성능 분석

---

## Limitations

본 시스템은 의료 문서를 기반으로 한 **연구용 질의응답 프로토타입**으로, 실제 의료 전문가의 진단이나 판단을 대체하기 위한 시스템은 아닙니다.

주요 한계는 다음과 같습니다.

* 사용한 의료 문서의 범위와 최신성에 따른 한계
* 다양한 자연어 질의 표현에 따른 검색 성능 차이
* Retrieval 이후 Generation 단계에서 발생하는 정보 활용 한계
* 제한된 테스트 데이터셋에 따른 평가 범위
* 실제 의료 환경에서의 추가적인 검증 필요

향후에는 더 다양한 질의와 문서 데이터를 활용하고, retrieval 방식 및 reranking 등의 검색 방법을 개선하여 **검색 정확도와 최종 답변 품질을 함께 향상**시킬 수 있습니다.

---

## Tech Stack

**Language**

* Python

**LLM & RAG**

* GPT-4o
* LangChain
* RAG
* Prompt Engineering

**Information Retrieval**

* FAISS
* OpenAI `text-embedding-3-small`
* Vector Similarity Search

**Document Processing**

* PyMuPDF
* PyPDFLoader

**Evaluation**

* Recall@3
* Generation Accuracy
* Retrieval Result Analysis
