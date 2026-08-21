# AI Paper Agent V2

기술 논문과 논문에 연결된 공식 코드를 근거로 모델 구조를 분석하고, 사용자가 승인한
Architecture IR에서 검증 가능한 PyTorch 패키지를 생성하는 AI agent입니다.

V2의 목표는 논문을 비슷한 모델 템플릿에 끼워 맞추는 것이 아닙니다. 논문에 구현 정보가
없거나 해석이 불확실하면 임의의 CNN·MLP로 대체하지 않고 `needs_review`와 누락 근거를
반환합니다.

> 현재 버전은 `2.0.0a1`입니다. 로컬 단일 사용자와 PyTorch 모델 생성을 1차 범위로 하며,
> 인증·멀티테넌시·전체 학습 파이프라인은 아직 포함하지 않습니다.

## 핵심 파이프라인

```text
PDF 및 공식 코드
       ↓
페이지·section·caption·table·equation 단위 근거 추출
       ↓
PDF 시각 자료 분석 + hybrid retrieval
       ↓
Evidence가 연결된 Architecture IR 생성
       ↓
사용자 수정 및 승인
       ↓
PyTorch registry 조립 + 제한된 custom block 생성
       ↓
격리 sandbox 검증 + 최대 3회 제한적 수정
       ↓
검증 결과와 provenance를 포함한 ZIP artifact
```

코드 생성 전에 사용자가 Architecture IR의 graph, tensor shape, evidence, assumption,
unresolved item을 검토하고 승인해야 합니다.

## V2에서 달라진 점

| 영역 | V1 | V2 |
| --- | --- | --- |
| 모델 생성 | 모델 family별 Jinja 템플릿과 routing | Architecture IR DAG를 PyTorch component registry로 결정적 조립 |
| 새로운 구조 | 가까운 템플릿으로 대체 가능 | registry에 없는 block만 계약이 제한된 `nn.Module`로 생성 |
| 근거 | 생성 코드와 논문 근거 연결이 약함 | 주요 node에 PDF/공식 코드 evidence 또는 명시적 assumption 필수 |
| 사용자 통제 | 분석과 생성이 밀접하게 연결 | IR 수정·버전 관리·승인 후에만 생성 가능 |
| 검증 | 정적 검사 중심 | compile부터 optimizer 1 step까지 격리 컨테이너에서 실행 |
| 실패 처리 | fallback 모델로 성공 처리 가능 | `needs_review`와 구체적인 실패 원인 반환 |
| 작업 상태 | 요청 처리 중 상태 유실 가능 | PostgreSQL 작업 테이블과 재시작 복구 지원 |
| QA | 일반 RAG 응답 | answerability, 페이지 citation, retrieval debug 포함 |

Jinja는 V2 모델 아키텍처를 만들지 않습니다. 생성 패키지의 `config.py`, 예제 입력,
테스트, README 같은 프로젝트 골격에만 사용됩니다.

## 빠른 시작

### 요구사항

- Docker와 Docker Compose
- OpenAI API 또는 Azure OpenAI 자격증명
- 로컬 개발 시 Python 3.11 또는 3.12

### 1. 환경변수 설정

```bash
cp .env.example .env
```

`.env`에서 최소한 다음 값을 변경합니다.

```dotenv
POSTGRES_PASSWORD=replace-with-a-strong-password
DATABASE_URL=postgresql+psycopg://paper_agent:replace-with-a-strong-password@postgres:5432/paper_agent

LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-5.4
```

Azure OpenAI를 사용한다면 `LLM_PROVIDER=azure`로 변경하고
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`를 설정합니다.
논문에 포함된 GitHub 저장소를 더 안정적으로 조회하려면 선택적으로 `GITHUB_TOKEN`을
설정할 수 있습니다.

`.env`에는 실제 비밀값이 들어가므로 절대 Git에 커밋하지 마세요.

### 2. sandbox 이미지 빌드 및 서비스 실행

```bash
docker build -f docker/sandbox.Dockerfile -t ai-paper-agent-sandbox:latest .
docker compose up --build
```

Compose는 다음 서비스를 실행합니다.

| 서비스 | 역할 |
| --- | --- |
| `postgres` | PostgreSQL 16 + pgvector |
| `migrate` | API 시작 전 `alembic upgrade head` 실행 |
| `api` | FastAPI V2 API |
| `worker` | 분석·생성 작업 처리 및 실패 재시도 |
| `sandbox-runner` | 생성 패키지를 제한된 Docker 컨테이너에서 검증 |
| `ui` | V2 Streamlit 검토 화면 |

실행 후 접속 주소:

- V2 UI: <http://localhost:8502>
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

종료할 때는 다음 명령을 사용합니다.

```bash
docker compose down
```

DB와 artifact volume까지 제거하려면 `docker compose down -v`를 사용할 수 있지만, 저장된
분석 결과와 생성물이 함께 삭제됩니다.

## 사용 흐름

V2 UI는 다음 다섯 단계로 분리되어 있습니다.

1. PDF 업로드 및 분석 시작
2. run 진행률과 event log 확인
3. Architecture IR의 graph, shape, evidence, assumption 검토
4. IR 승인 후 PyTorch 패키지 생성 및 sandbox 결과 확인
5. 논문 근거 citation이 포함된 QA

업로드 요청은 분석이 끝날 때까지 기다리지 않고 `202 Accepted`와 UUID 기반
`document_id`, `analysis_run_id`를 즉시 반환합니다. worker가 비동기로 분석한 뒤 IR을
`needs_review` 또는 `draft` 상태로 제공합니다.

### API 예제

논문 업로드:

```bash
curl -X POST http://localhost:8000/api/v2/documents \
  -H 'accept: application/json' \
  -F 'file=@paper.pdf;type=application/pdf'
```

진행 상태와 SSE event 확인:

```bash
curl http://localhost:8000/api/v2/runs/<RUN_ID>
curl -N http://localhost:8000/api/v2/runs/<RUN_ID>/events
```

최신 IR 조회 및 승인:

```bash
curl http://localhost:8000/api/v2/documents/<DOCUMENT_ID>/spec
curl -X POST \
  http://localhost:8000/api/v2/documents/<DOCUMENT_ID>/spec/<VERSION>/approve
```

blocking unresolved item이 남아 있으면 승인 요청은 `409 Conflict`로 거절됩니다. IR을
수정하려면 전체 `ModelGraphSpec` JSON을 `PATCH /api/v2/documents/<DOCUMENT_ID>/spec`에
보내며, 기존 버전을 덮어쓰지 않고 새 버전을 생성합니다.

승인된 IR로 코드 생성:

```bash
curl -X POST http://localhost:8000/api/v2/documents/<DOCUMENT_ID>/generations
curl http://localhost:8000/api/v2/generations/<GENERATION_ID>
curl -OJ http://localhost:8000/api/v2/artifacts/<ARTIFACT_ID>/download
```

논문 QA:

```bash
curl -X POST http://localhost:8000/api/v2/documents/<DOCUMENT_ID>/questions \
  -H 'content-type: application/json' \
  -d '{"question":"이 모델의 핵심 skip connection은 어떻게 구성되는가?","limit":6}'
```

QA 응답은 다음 구조를 사용합니다.

```json
{
  "answer": "...",
  "citations": [
    {
      "page": 4,
      "section": "method",
      "chunk_id": "p0004-0007-a1b2c3d4e5f6",
      "evidence": "..."
    }
  ],
  "answerability": "answerable",
  "retrieval_debug": {
    "retrieved": 6,
    "supported": 3,
    "chunk_ids": ["..."],
    "dense_used": true
  }
}
```

근거가 부족하면 답을 추측하지 않고 `answer`를 `null`, `answerability`를
`insufficient_evidence`로 반환합니다.

## 공개 API

| Method | Endpoint | 설명 |
| --- | --- | --- |
| `POST` | `/api/v2/documents` | PDF 저장, document/run 생성, 분석 enqueue |
| `GET` | `/api/v2/runs/{run_id}` | run 상태, 단계, 진행률, 오류 조회 |
| `GET` | `/api/v2/runs/{run_id}/events` | 진행 이벤트 SSE stream |
| `GET` | `/api/v2/documents/{id}/spec` | 최신 Architecture IR 조회 |
| `PATCH` | `/api/v2/documents/{id}/spec` | 수정된 IR을 새 version으로 저장 |
| `POST` | `/api/v2/documents/{id}/spec/{version}/approve` | unresolved 검증 후 IR 승인 |
| `POST` | `/api/v2/documents/{id}/generations` | 승인된 최신 IR의 생성 작업 enqueue |
| `GET` | `/api/v2/generations/{id}` | repair, validation, artifact 상태 조회 |
| `GET` | `/api/v2/artifacts/{id}/download` | 검증된 ZIP artifact 다운로드 |
| `POST` | `/api/v2/documents/{id}/questions` | citation과 answerability가 포함된 QA |

## Architecture IR

`ModelGraphSpec`은 논문 모델을 실행 가능한 typed DAG로 표현합니다.

- `TensorSpec`: tensor 이름, dtype, symbolic shape, 의미
- `NodeSpec`: operation/module, 입력·출력, 반복, 조건, weight sharing, parameter
- `EvidenceRef`: PDF 페이지·section·chunk 또는 공식 코드 URL·commit SHA
- `TrainingSpec`: loss, optimizer, scheduler, initialization, metric
- `Assumption`: 논문에 없어 추정한 값과 이유, confidence
- `UnresolvedItem`: 구현 전에 결정해야 할 질문과 blocking 여부

IR 검증 규칙:

- graph cycle과 아직 생성되지 않은 tensor 참조를 거절합니다.
- 모든 주요 node는 하나 이상의 evidence 또는 명시적 assumption을 가져야 합니다.
- 존재하지 않는 evidence와 weight-sharing node 참조를 거절합니다.
- symbolic shape 문법을 검사합니다.
- blocking unresolved item이 하나라도 있으면 `needs_review`가 됩니다.
- 승인된 IR만 코드 생성으로 넘어갈 수 있습니다.

## 하이브리드 PyTorch 생성

IR의 DAG를 위상 정렬해 `nn.Module` 구성과 `forward()` 데이터 흐름을 결정적으로
생성합니다. 현재 registry는 다음 primitive와 조합 block을 제공합니다.

- Linear, Conv1d/2d/3d, ConvTranspose2d
- BatchNorm, LayerNorm, activation, dropout, pooling, embedding
- Multi-head attention, Transformer encoder/decoder layer
- LSTM, GRU
- add/residual/skip, concat, reshape, permute
- residual block, graph convolution, VAE reparameterization
- time-series decomposition, multimodal fusion

registry에 없는 operation만 LLM이 독립된 custom `nn.Module`로 생성할 수 있습니다. custom
module은 입력·출력 계약과 class name을 제공받으며 AST 검사를 통과해야 합니다. 허용 import는
`torch`, `typing`, `math`로 제한되고, LLM은 프로젝트 전체를 자유롭게 다시 작성할 수 없습니다.

생성 ZIP에는 다음 파일이 포함됩니다.

```text
model.py
config.py
example_inputs.py
architecture.json
provenance.json
validation.json
README.md
tests/
└── test_model.py
```

`provenance.json`에는 PDF hash, spec/schema version, provider/model, generator version, artifact
fingerprint, 제한적 repair 기록을 저장합니다.

## Sandbox 검증과 자동 수정

생성 코드는 API key나 DB 자격증명 없이 별도 컨테이너에서 실행합니다. sandbox는 network를
비활성화하고 read-only root filesystem, capability drop, `no-new-privileges`, CPU/RAM/PID/time
제한을 적용합니다.

검증 순서:

```text
compileall → import → instantiate → dummy forward → output shape
           → semantic contract → backward → optimizer 1 step
```

검증 오류는 syntax, import, shape, dtype/device, OOM/timeout, semantic contract 등으로
분류됩니다. 자동 수정은 최대 3회이며 IR patch, custom block, glue code의 제한된 범위만
허용합니다. 끝까지 통과하지 못하면 fallback 모델을 만들지 않고 generation을
`needs_review`로 종료합니다.

sandbox만 별도로 확인하려면 다음 smoke test를 실행합니다.

```bash
docker build -f docker/sandbox.Dockerfile -t ai-paper-agent-sandbox:latest .
uv run python scripts/sandbox_smoke.py
```

## 데이터와 작업 복구

- 업로드한 PDF와 생성 artifact는 Git 밖의 `var/storage` 또는 Compose의
  `ai-paper-agent-storage` volume에 저장합니다.
- 메타데이터, section, chunk, evidence, IR version, generation, QA, 평가 결과는
  PostgreSQL에 저장합니다.
- worker는 row locking으로 작업을 claim하며, 오래된 `running` 작업을 다시 `queued`로
  돌려 서버 재시작 후 처리할 수 있습니다.
- PostgreSQL 환경에서는 LangGraph Postgres checkpointer를 사용할 수 있는 기반을
  제공합니다.

## 프로젝트 구조

```text
.
├── src/paper_agent_v2/
│   ├── api.py, schemas.py, models.py  # V2 API와 영속 데이터 모델
│   ├── parser.py, analysis.py         # layout-aware PDF와 Architecture IR 추출
│   ├── retrieval.py                   # semantic + full-text RRF 검색
│   ├── official_code.py               # 공식 GitHub code/commit/license 근거
│   ├── ir.py                           # typed Architecture IR와 graph validation
│   ├── providers/                      # OpenAI/Azure OpenAI provider adapter
│   ├── generation/                     # registry, renderer, custom block, package writer
│   ├── jobs.py, worker.py              # durable queue, 재시도, 재시작 복구
│   └── sandbox.py, sandbox_runner.py   # 격리 검증 실행
├── migrations/                         # Alembic schema migration
├── tests/                              # V2 unit/integration-oriented tests
├── benchmarks/                         # 12-family benchmark manifest
├── docker/                             # API, worker, runner, sandbox image
├── scripts/                            # benchmark, sandbox, history 정리 도구
├── v2_app.py                           # V2 Streamlit UI
├── docker-compose.yml
└── pyproject.toml
```

## 로컬 개발과 테스트

의존성은 `pyproject.toml`과 `uv.lock`을 기준으로 관리합니다.

```bash
uv sync --frozen --extra worker --extra dev
uv run alembic upgrade head
uv run ruff check src tests scripts/run_v2_benchmark.py scripts/sandbox_smoke.py
uv run mypy src/paper_agent_v2 --ignore-missing-imports
uv run pytest
uv run pip-audit
```

Compose 설정만 검증하려면 `.env`를 만든 뒤 다음을 실행합니다.

```bash
docker compose config --quiet
```

CI는 pull request와 `main` push에서 gitleaks, locked dependency 설치, Ruff, pytest,
dependency audit, Compose 검증, sandbox image build를 실행합니다.

## 12편 acceptance benchmark

예제 manifest를 복사하고 합법적으로 보유한 PDF의 절대 경로를 입력합니다.

```bash
cp benchmarks/manifest.example.json benchmarks/manifest.local.json
uv run python scripts/run_v2_benchmark.py benchmarks/manifest.local.json
```

기본 실행은 IR과 evidence까지만 측정합니다. blocking unresolved가 없는 spec에 한해
승인·생성까지 자동 측정하려면 명시적으로 `--approve`를 추가합니다.

```bash
uv run python scripts/run_v2_benchmark.py benchmarks/manifest.local.json --approve
```

benchmark family는 CNN, ViT, encoder-decoder, UNet, RNN, VAE, GAN, GNN, time-series,
diffusion, multimodal, novel custom block으로 구성됩니다. 논문 PDF 자체는 저작권과 용량 문제로
저장소에 포함하지 않습니다.

## 보안

- `.env`, 업로드 PDF, 생성물, benchmark 로컬 manifest를 커밋하지 않습니다.
- 생성 코드는 신뢰하지 않는 입력으로 취급하고 host Python에서 직접 실행하지 않습니다.
- 운영에서는 Docker socket에 접근하는 `sandbox-runner`의 권한을 별도 host/VM 또는 전용
  container runtime으로 더 강하게 격리하는 것을 권장합니다.
- 과거 credential 노출 대응과 history purge 절차는 [SECURITY.md](SECURITY.md)를 확인하세요.
- 보안 문제에 실제 비밀값을 포함해 공개 issue를 만들지 마세요.

## V1 호환 모드

기존 TensorFlow/Jinja 구현은 회귀 비교와 호환 목적으로만 남아 있습니다.

- V1 UI: `app.py`
- V1 API: `backend.main:app`
- V1 services/templates: `services/`

V1의 모델 family Jinja 템플릿, regex routing, `/documents`, `/qa` 경로는 V2 생성
파이프라인에서 사용하지 않습니다. 새로운 개발은 `src/paper_agent_v2`와 `/api/v2`를
기준으로 진행합니다.

## 현재 범위와 제한

- 생성 대상은 PyTorch입니다. TensorFlow는 V1 호환 경로에만 남아 있습니다.
- 실제 dataset downloader와 전체 training pipeline은 기본 artifact에 포함하지 않습니다.
- 공식 코드 자동 탐색은 논문 본문에 명시된 GitHub URL을 우선하며, 제목과 저장소 정보의
  일치 여부를 보수적으로 확인합니다.
- 스캔 PDF처럼 추출 가능한 텍스트가 없는 문서는 현재 거절됩니다.
- benchmark acceptance 목표는 12/12 IR schema 생성, 10/12 dummy forward/backward 통과,
  실패 사유 100% 명시, QA citation page 정확도 90% 이상입니다. 이는 목표 기준이며 각 실행의
  실제 결과는 benchmark report로 확인해야 합니다.
