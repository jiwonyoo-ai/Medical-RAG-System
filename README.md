# Medical-LLM-Assistant

> A Retrieval-Augmented Generation (RAG) system for medical question answering using vectorized medical documents and large language models.

PDF 기반 의료 문서를 외부 지식으로 구축하고, 사용자의 질문과 관련된 의료 정보를 검색한 뒤 검색 결과를 LLM의 context로 활용하여 답변을 생성하는 **RAG (Retrieval-Augmented Generation) 기반 의료 질의응답 시스템**입니다.

본 프로젝트에서는 의료 문서의 전처리부터 임베딩, 벡터 검색, 검색 결과 기반 LLM 응답 생성까지의 **End-to-End RAG pipeline을 설계 및 구현**했습니다.

---

## Project Overview

### Motivation

의료 질의응답에서는 단순히 LLM의 사전 학습 지식에 의존하기보다, **질문과 관련된 외부 의료 정보를 검색하고 이를 바탕으로 답변을 생성하는 과정**이 중요합니다.

본 프로젝트에서는 의료 문서를 외부 지식으로 구축하고, 사용자의 질문과 의미적으로 관련된 문서를 검색하여 LLM에 context로 제공하는 RAG 구조를 구현했습니다.

이를 통해 **Retrieval과 Generation을 결합한 의료 질의응답 pipeline**을 구축했습니다.

---

## Objectives

* 의료 PDF 문서의 전처리 및 검색 가능한 데이터 구축
* 의료 문서의 텍스트 임베딩 및 벡터 데이터베이스 구축
* 사용자 질문과 관련된 의료 정보 검색
* 검색 결과를 활용한 LLM 기반 답변 생성
* End-to-End 의료 질의응답 pipeline 구현
* 질의응답 인터페이스 구현

---

## System Architecture

```text
Medical PDF Documents
        ↓
Document Loading
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
LLM Prompt
        ↓
GPT-4o
        ↓
Generated Answer
```

---

## Workflow

### 1. Medical Document Processing

MSD 매뉴얼에서 수집한 의료 PDF 문서를 질의응답에 활용할 수 있도록 전처리했습니다.

`PyPDFLoader`를 이용하여 PDF 문서를 불러오고, 문서의 텍스트를 추출한 뒤 검색 가능한 형태로 구성했습니다.

---

### 2. Text Splitting & Embedding

전처리된 문서를 검색 단위로 분할한 후 OpenAI의 `text-embedding-3-small` 모델을 이용하여 각 텍스트 chunk의 벡터 임베딩을 생성했습니다.

이를 통해 사용자의 질문과 의료 문서 간의 의미적 유사도를 기반으로 검색할 수 있도록 구성했습니다.

---

### 3. Vector Database

생성된 문서 임베딩을 **FAISS Vector Store**에 저장했습니다.

FAISS를 이용하여 사용자 질문과 의미적으로 유사한 의료 문서를 효율적으로 검색할 수 있도록 벡터 검색 구조를 구현했습니다.

---

### 4. Similarity Search

사용자가 입력한 질문을 동일한 embedding space로 변환한 후 FAISS를 이용하여 관련성이 높은 의료 문서를 검색합니다.

검색된 문서는 이후 LLM이 답변을 생성할 때 활용되는 **context**로 전달됩니다.

---

### 5. LLM-based Answer Generation

검색된 의료 문서와 사용자의 질문을 함께 LLM에 전달하여, 검색된 정보를 context로 활용한 자연어 답변을 생성합니다.

이를 통해 LLM의 자체 지식뿐만 아니라 **검색된 외부 의료 문서를 활용하는 답변 생성 구조**를 구현했습니다.

---

### 6. Web Interface

사용자가 증상이나 질병과 관련된 질문을 입력하고, 검색 및 LLM 응답 결과를 확인할 수 있는 웹 인터페이스를 구현했습니다.

---

## Core Technologies

### Retrieval-Augmented Generation

본 시스템은 **Retrieval**과 **Generation**을 결합한 RAG 구조로 구성되어 있습니다.

```text
User Query
    ↓
Document Retrieval
    ↓
Relevant Medical Context
    ↓
LLM
    ↓
Answer Generation
```

사용자 질문과 관련된 의료 문서를 검색하고, 검색된 정보를 LLM의 context로 제공하여 답변을 생성합니다.

---

### FAISS

FAISS를 이용하여 벡터화된 의료 문서와 사용자 질문 간의 유사도를 기반으로 관련 문서를 검색했습니다.

---

### LangChain

LangChain을 활용하여 문서 로딩, 텍스트 분할, 임베딩, 벡터 검색 및 LLM 호출 과정을 연결하고 전체 RAG pipeline을 구성했습니다.

---

## Experimental Setup

| Component       | Technology               |
| --------------- | ------------------------ |
| LLM             | GPT-4o                   |
| Embedding       | `text-embedding-3-small` |
| Vector Store    | FAISS                    |
| Framework       | LangChain                |
| Document Loader | PyPDFLoader              |
| PDF Processing  | PyMuPDF                  |
| Interface       | Streamlit                |
| Language        | Python                   |

---

## Evaluation

증상 및 질병 관련 테스트 질의를 구성하여 **검색된 의료 문서의 관련성**과 **생성된 답변의 적절성**을 확인했습니다.

또한 similarity score를 확인하여 사용자 질문과 검색 문서 간의 의미적 관련성을 분석했습니다.

> *Note: similarity score의 절대값은 사용한 distance/similarity metric과 설정에 따라 해석이 달라질 수 있으므로, score 자체를 성능 지표로 해석하기보다 검색 결과의 관련성을 확인하는 데 활용했습니다.*

---

## My Contributions

* 의료 질의응답 시스템 기획 및 전체 RAG pipeline 설계
* 의료 PDF 문서 전처리 및 검색 데이터 구축
* 문서 chunking 및 embedding pipeline 구현
* FAISS 기반 벡터 검색 구현
* LangChain 기반 RAG pipeline 구축
* 검색 결과와 LLM을 연결하는 context 구성 및 prompt 설계
* GPT-4o 기반 답변 생성 구현
* Streamlit 기반 사용자 인터페이스 구현
* 테스트 질의를 통한 검색 및 답변 결과 검증

---

## Tech Stack

**Language**

* Python

**LLM & RAG**

* GPT-4o
* LangChain
* RAG
* Prompt Engineering

**Retrieval**

* FAISS
* OpenAI Embeddings

**Document Processing**

* PyMuPDF
* PyPDFLoader

