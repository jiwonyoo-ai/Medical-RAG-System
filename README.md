# Medical-LLM-Assistant 

> A GPT-4o-based medical consultation RAG system integrating medical document retrieval, symptom-based disease inference, and conversational question answering.

PDF 기반 의료 문서를 외부 지식으로 구축하고, 사용자의 증상 및 질의와 관련된 의료 정보를 검색한 뒤 이를 GPT-4o의 context로 활용하여 답변을 생성하는 **RAG (Retrieval-Augmented Generation) 기반 의료 상담 시스템**입니다.

사용자의 증상 정보가 부족한 경우 추가 질문을 수행하고, 수집된 증상을 바탕으로 질병 후보를 추론할 수 있도록 **의료 문서 검색과 AI 기반 질병 추론 기능을 결합**했습니다.

---

## Project Overview

### Motivation

의료 질의응답에서는 LLM의 사전 학습 지식에만 의존하기보다, 신뢰할 수 있는 의료 문서를 검색하고 이를 근거로 답변을 생성하는 과정이 중요합니다.

본 프로젝트에서는 MSD Manual 기반 의료 문서를 외부 지식으로 구축하고, 사용자의 증상 및 질문과 관련된 문서를 검색하여 GPT-4o에 context로 제공하는 RAG 구조를 구현했습니다.

또한 사용자의 입력 정보가 부족할 경우 추가 질문을 수행하여 증상 정보를 보완하고, 이를 바탕으로 질병 후보를 추론하도록 설계했습니다.

---

## Objectives

* 의료 PDF 문서 전처리 및 검색 데이터 구축
* 의료 문서 임베딩 및 FAISS 기반 벡터 검색 구현
* 사용자 증상 및 질의 기반 의료 정보 검색
* 증상 정보 보완을 위한 추가 질문 구조 구현
* 사용자 증상 기반 질병 후보 추론
* 검색 결과 기반 GPT-4o 의료 정보 설명
* 대화 문맥을 유지하는 Conversational Memory 구현
* 검색 결과 및 응답에 대한 평가 수행

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
User Symptoms / Query
        ↓
Additional Questions
        ↓
Similarity Search
        ↓
Retrieved Medical Documents
        ↓
Context Construction
        ↓
Disease Inference
        ↓
GPT-4o
        ↓
Medical Explanation
        ↓
Conversational Response
```

---

## Workflow

### 1. Medical Document Processing

MSD Manual 기반 의료 PDF 문서를 질의응답에 활용할 수 있도록 전처리했습니다.

`PyPDFLoader`를 이용하여 PDF 문서의 텍스트를 추출하고 검색 가능한 형태로 정제했습니다.

---

### 2. Text Splitting & Embedding

전처리된 의료 문서를 검색 단위로 분할하고 OpenAI의 `text-embedding-3-small` 모델을 이용하여 각 text chunk의 벡터 임베딩을 생성했습니다.

생성된 임베딩은 FAISS Vector Store에 저장하여 사용자 질의와 의료 문서 간의 의미적 유사도를 기반으로 검색할 수 있도록 구성했습니다.

---

### 3. Symptom-based Questioning

사용자가 제공한 증상 정보가 부족한 경우 추가 질문을 수행하도록 설계했습니다.

이를 통해 첫 번째 입력만으로 답변을 생성하는 것이 아니라, **대화형 질문을 통해 필요한 증상 정보를 단계적으로 확보**할 수 있도록 구성했습니다.

---

### 4. Medical Document Retrieval

사용자의 증상 및 질문을 embedding space로 변환한 후 FAISS를 이용하여 관련성이 높은 의료 문서를 검색합니다.

검색된 문서는 GPT-4o가 답변을 생성할 때 활용되는 **context**로 전달됩니다.

---

### 5. Disease Inference

수집된 증상 정보를 바탕으로 관련 질병 후보를 추론하도록 구성했습니다.

질병 추론 결과와 검색된 의료 문서를 함께 활용하여 관련 의료 정보를 자연어로 설명할 수 있도록 설계했습니다.

---

### 6. LLM-based Medical Response

검색된 의료 문서와 사용자 증상 및 질문을 GPT-4o에 전달하여 의료 정보를 설명하는 자연어 답변을 생성합니다.

검색된 외부 의료 문서를 context로 활용하여 **LLM의 자체 지식에만 의존하지 않고 검색 결과를 기반으로 답변을 생성하는 구조**를 구현했습니다.

---

### 7. Conversational Interface

사용자와의 지속적인 상담을 위해 이전 대화의 문맥을 유지하는 Memory 기능을 포함했습니다.

사용자는 웹 인터페이스를 통해 증상 및 질병 관련 질문을 입력하고 AI의 질문과 의료 정보 기반 답변을 확인할 수 있습니다.

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
| Language        | Python                   |

---

## Evaluation

증상 및 질병 관련 테스트 질의를 구성하여 **검색된 의료 문서의 관련성, 답변의 일관성 및 질의응답 흐름**을 확인했습니다.

또한 `similarity_search_with_score`를 활용하여 사용자 질의와 검색 문서 간의 유사도 거리를 분석했습니다.

![Medical RAG System](./images/image.png)

검색 결과를 통해 질환 유형에 따라 사용자 증상과 의료 문서 간의 검색 관련성이 달라질 수 있음을 확인했습니다.

> **Note:** Similarity score의 절대값은 사용한 distance metric 및 검색 설정에 따라 달라질 수 있으므로, score 자체를 의료 답변의 정확도를 나타내는 절대적인 성능 지표로 해석하기보다 검색 결과의 관련성을 분석하는 기준으로 활용했습니다.

---

## My Contributions

* 의료 상담 시스템 기획 및 전체 RAG pipeline 설계
* 의료 PDF 문서 전처리 및 검색 데이터 구축
* 문서 chunking, embedding 및 FAISS 기반 검색 구현
* 사용자 증상 기반 질의응답 및 추가 질문 흐름 설계
* 검색 결과와 GPT-4o를 연결하는 context 및 prompt 설계
* GPT-4o 기반 의료 정보 설명 및 Conversational Memory 구현
* 테스트 질의를 통한 검색 결과 및 응답 검증

---

## Limitations

본 시스템은 의료 정보 탐색 및 상담을 보조하기 위한 연구용 프로토타입으로, 실제 의료 전문가의 진단이나 판단을 대체하지 않습니다.

주요 한계는 다음과 같습니다.

* 의료 문서의 최신성 및 데이터 범위에 따른 한계
* 다양한 자연어 증상 표현에 대한 대응 한계
* AI 추론 결과와 실제 임상 판단 간의 차이
* 국내 의료 환경에 특화된 데이터 및 모델 부재
* 의료 전문가의 검증을 통한 추가적인 평가 필요

향후에는 국내 의료 환경에 적합한 데이터셋 구축, 도메인 특화 모델 적용 및 의료 전문가의 피드백을 반영한 평가 체계를 통해 시스템의 신뢰성과 정확성을 개선할 수 있습니다.

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
