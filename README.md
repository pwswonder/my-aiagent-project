# 🤖 AI Paper Code Generator

**기술논문에서 AI 모델 코드를 자동 생성하는 지능형 Agent 시스템**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org)

---

## V2: evidence-grounded PyTorch generation

V1의 TensorFlow/Jinja 경로는 호환 목적으로 유지됩니다. V2는 `/api/v2`와
`src/paper_agent_v2`에 병행 구축되며 다음 흐름을 사용합니다.

`PDF/official code → cited Architecture IR → user approval → registry/custom block generation → isolated validation`

모델 전체를 Jinja로 렌더링하지 않습니다. Jinja는 `config.py`, 테스트, README 같은 패키지
골격에만 쓰며, 알 수 없는 구조를 임의의 CNN/MLP로 대체하지 않습니다. 근거 또는 명시적
assumption이 없는 node와 blocking unresolved item이 있는 spec은 승인할 수 없습니다.

### V2 실행

Python 3.11과 Docker Compose가 필요합니다.

```bash
cp .env.example .env
# .env의 POSTGRES_PASSWORD와 선택한 LLM provider 설정을 입력합니다.
docker build -f docker/sandbox.Dockerfile -t ai-paper-agent-sandbox:latest .
docker compose up --build
```

- V2 UI: http://localhost:8502
- API docs: http://localhost:8000/docs
- V1 UI/API는 기존 `app.py`, `backend.main:app` 경로로 유지됩니다.

### V2 구조

```text
src/paper_agent_v2/
├── api.py, models.py, jobs.py, worker.py  # API, durable job table, workers
├── parser.py, retrieval.py, analysis.py   # layout-aware PDF, pgvector/FTS RRF, IR extraction
├── ir.py                                 # typed graph/evidence/assumption contract
├── providers/                            # OpenAI/Azure OpenAI adapters
├── generation/                           # deterministic registry + constrained custom blocks
└── sandbox.py, repair.py                 # isolated checks and bounded repair policy
```

DB 스키마는 `alembic upgrade head`로만 관리합니다. 운영 파일은 `var/storage`에 보관되며 Git에
추가되지 않습니다. 과거 `backend/.env`는 삭제만으로 안전해지지 않으므로 [SECURITY.md](SECURITY.md)의
자격증명 회전 및 history purge 절차를 반드시 수행해야 합니다.

12편 acceptance benchmark는 `benchmarks/manifest.example.json`을 복사해 합법적으로 보유한 PDF의
절대 경로를 입력한 뒤 실행합니다. 기본 모드는 IR/evidence까지만 측정하며, blocking unresolved가 없는
spec의 승인·생성까지 자동 측정하려면 명시적으로 `--approve`를 추가합니다.

```bash
python scripts/run_v2_benchmark.py benchmarks/manifest.local.json
python scripts/run_v2_benchmark.py benchmarks/manifest.local.json --approve
```

---

## 📋 프로젝트 개요

**AI Paper Code Generator**는 기술논문(PDF)을 분석하여 논문에서 제안하는 AI 모델의 TensorFlow/Keras 구현 코드를 자동으로 생성하는 시스템입니다.

### 🎯 주요 기능

- **📤 PDF 논문 업로드 및 자동 분석**
- **🤖 AI 모델 아키텍처 추출 및 스펙 생성**
- **🏗️ 다양한 모델 템플릿 지원** (Transformer, CNN, ResNet, RNN, GAN, VAE, UNet 등)
- **⚡ 지능형 템플릿 라우팅** (논문 내용 기반 최적 템플릿 선택)
- **🔧 슬롯 기반 코드 완성** (LLM + 자동블록 주입)
- **📊 코드 품질 분석 및 리플렉션**
- **💬 논문 내용 기반 QA 시스템**
- **💾 생성 코드 영속화 및 다운로드**

### 🏗️ 시스템 아키텍처
![모델 다이어그램](assets/Mermaid_Chart-2025-09-07-144649.png "System Architecture")


---

## 🚀 설치 방법

### 📋 필수 요구사항

- **Python 3.8+**
- **pip 또는 conda**
- **Git**

### 🛠️ 설치 단계

1. **저장소 클론**
```bash
git clone https://github.com/your-username/ai-paper-code-generator.git
cd ai-paper-code-generator
```

2. **가상환경 생성 및 활성화**
```bash
# conda 사용 시
conda create -n paper-ai python=3.8
conda activate paper-ai

# venv 사용 시  
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **의존성 패키지 설치**
```bash
pip install -r requirements.txt
```

4. **환경변수 설정**
```bash
# .env 파일 생성
# .env 파일에 API 키 설정
OPENAI_API_KEY=
# 기타 필요한 환경변수들...
```

5. **데이터베이스 초기화**
```bash
# SQLite DB는 자동으로 생성됩니다
# 필요시 migration 실행
```

---

## 💡 사용법 및 예제

### 🖥️ 시스템 실행

1. **Backend API 서버 시작**
```bash
# FastAPI 서버 실행 (포트 8000)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Frontend UI 실행**
```bash
# 새 터미널에서 Streamlit 앱 실행
streamlit run app.py
```

3. **브라우저에서 접속**
```
http://localhost:8501
```

### 📖 사용 예제

#### 1. PDF 논문 업로드 및 분석

```python
# API를 통한 직접 호출 예제
import requests

# 논문 업로드 및 즉시 질문
files = {"file": ("paper.pdf", open("paper.pdf", "rb"), "application/pdf")}
data = {"question": "이 논문에서 제안하는 모델의 구조를 설명해주세요"}

response = requests.post(
    "http://localhost:8000/documents/upload",
    files=files,
    data=data
)

result = response.json()
print(f"생성된 코드 경로: {result['basecode_py_path']}")
print(f"모델 요약: {result['basecode_summary']}")
```

#### 2. 생성된 코드 실행 예제

```python
# 생성된 코드 파일을 import하여 모델 빌드
import sys
sys.path.append('/path/to/generated/code')

from transformer_basecode import build_model

# 모델 생성
model = build_model()
print(model.summary())

# 모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
```

#### 3. 기존 문서에 추가 질문

```python
# 기존 문서에 대한 질문
payload = {
    "document_id": 1,
    "question": "이 모델의 하이퍼파라미터 설정 방법은?"
}

response = requests.post(
    "http://localhost:8000/qa/ask_existing",
    json=payload
)

answer = response.json()["answer"]
print(answer)
```

### 🎨 지원하는 모델 템플릿

| 모델 계열 | 템플릿 | 지원 기능 |
|-----------|--------|-----------|
| **Transformer** | `transformer.j2` | Multi-head Attention, Encoder-Decoder |
| **CNN Family** | `cnn_family.j2` | Conv2D, Pooling, Inception, SE Block |
| **ResNet** | `resnet.j2` | Residual Connection, Bottleneck |
| **RNN/LSTM** | `rnn_seq.j2` | LSTM, GRU, Attention Mechanism |
| **GAN** | `gan.j2` | Generator, Discriminator, GAN Loss |
| **VAE** | `vae.j2` | Encoder, Decoder, KL Divergence |
| **U-Net** | `unet.j2` | Encoder-Decoder with Skip Connection |
| **Autoencoder** | `autoencoder.j2` | Encoder, Decoder, Regularization |

---

## 📚 API 문서

### 🌐 FastAPI 자동 문서

시스템 실행 후 다음 URL에서 Interactive API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 🔗 주요 엔드포인트

#### 📤 문서 업로드 및 분석

```http
POST /documents/upload
Content-Type: multipart/form-data

Parameters:
- file: PDF 파일 (required)
- question: 즉시 질문 (required)

Response:
{
    "filename": "paper.pdf",
    "document_id": 1,
    "summary": "논문 요약...",
    "domain": "computer_vision",
    "answer": "질문에 대한 답변...",
    "used_model": "transformer",
    "basecode_py_path": "/path/to/generated.py",
    "basecode_source": "# Generated TensorFlow code...",
    "basecode_summary": "모델 아키텍처 요약..."
}
```

#### 📋 문서 목록 조회

```http
GET /documents

Response:
[
    {
        "id": 1,
        "filename": "paper.pdf", 
        "domain": "computer_vision",
        "summary": "논문 요약...",
        "uploaded_at": "2024-01-01T00:00:00"
    }
]
```

#### 💬 기존 문서 질문

```http
POST /qa/ask_existing
Content-Type: application/json

Body:
{
    "document_id": 1,
    "question": "모델의 성능은 어떤가요?"
}

Response:
{
    "answer": "질문에 대한 상세한 답변..."
}
```

#### 📊 QA 히스토리 조회

```http
GET /qa/{document_id}

Response:
[
    {
        "question": "질문 내용",
        "answer": "답변 내용", 
        "created_at": "2024-01-01T00:00:00"
    }
]
```

#### 💾 생성된 코드 조회

```http
GET /documents/{doc_id}/basecode

Response:
{
    "exists": true,
    "model_key": "transformer",
    "py_path": "/path/to/generated.py",
    "source": "# Generated Python code...",
    "summary": "모델 구조 요약..."
}
```

---

## 🛠️ 개발 가이드

### 📁 프로젝트 구조

```
ai-paper-code-generator/
├── 📁 backend/              # FastAPI 백엔드
│   ├── 📄 main.py          # API 서버 진입점
│   ├── 📁 routes/          # API 엔드포인트들
│   ├── 📄 models.py        # SQLAlchemy 데이터베이스 모델
│   ├── 📄 schemas.py       # Pydantic 스키마
│   └── 📄 database.py      # DB 연결 설정
├── 📁 services/            # 핵심 서비스 로직
│   ├── 📄 pipeline_basecode.py    # 메인 코드 생성 파이프라인
│   ├── 📄 graph_builder.py        # LangGraph 오케스트레이션
│   ├── 📄 routing.py              # 템플릿 라우팅 로직
│   ├── 📄 codegen.py              # Jinja2 코드 렌더링
│   ├── 📁 templates/              # Jinja2 템플릿 파일들
│   └── 📄 spec_schema.py          # 모델 스펙 스키마
├── 📄 app.py               # Streamlit 프론트엔드
├── 📄 templates_manifest.json     # 템플릿 메타데이터
├── 📄 requirements.txt     # Python 의존성
├── 📄 .env.example         # 환경변수 예제
└── 📄 README.md           # 이 파일
```

---

## 📄 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

```
MIT License

Copyright (c) 2025 AI Paper Code Generator Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 이 프로젝트는 다음 오픈소스 라이브러리들을 사용합니다:

- [FastAPI](https://fastapi.tiangolo.com/) - 현대적인 고성능 웹 API 프레임워크
- [Streamlit](https://streamlit.io/) - 데이터 앱 구축 도구  
- [LangChain](https://langchain.com/) - LLM 애플리케이션 개발 프레임워크
- [TensorFlow](https://tensorflow.org/) - 머신러닝 플랫폼
- [Jinja2](https://jinja.palletsprojects.com/) - 템플릿 엔진
- [SQLAlchemy](https://sqlalchemy.org/) - SQL 툴킷 및 ORM

---
