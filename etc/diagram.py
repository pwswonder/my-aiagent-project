<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1400" viewBox="0 0 2400 1400">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#444444"/>
    </marker>
    <style>
      text { font-family: Arial, sans-serif; fill:#222; }
      .cluster { stroke-width:3; fill-opacity:1; }
      .node { stroke:#666; stroke-width:2; fill:#FFF; }
      .label { font-size:18px; font-weight:bold; }
      .sublabel { font-size:14px; }
      .edge { stroke:#444; stroke-width:2; fill:none; }
      .edge.dashed { stroke-dasharray:6 6; }
    </style>
  </defs>

  <!-- Title -->
  <text x="1200" y="30" text-anchor="middle" font-size="26" font-weight="bold">
    System Architecture Diagram – AI Technical Paper Agent
  </text>

  <!-- Clusters -->
  <!-- Frontend -->
  <rect class="cluster" x="40" y="40" width="480" height="320" rx="14" ry="14" fill="#F6FAFF" stroke="#4F81BD"/>
  <text x="280" y="68" text-anchor="middle" font-size="22" font-weight="bold">Frontend (Streamlit UI)</text>

  <!-- Backend API -->
  <rect class="cluster" x="540" y="40" width="560" height="380" rx="14" ry="14" fill="#F8FFF0" stroke="#9BBB59"/>
  <text x="820" y="68" text-anchor="middle" font-size="22" font-weight="bold">Backend API (FastAPI)</text>

  <!-- Services -->
  <rect class="cluster" x="1120" y="40" width="920" height="780" rx="14" ry="14" fill="#FFF2F0" stroke="#C0504D"/>
  <text x="1580" y="68" text-anchor="middle" font-size="22" font-weight="bold">Services / Agents (LangGraph Orchestrated)</text>

  <!-- Templates -->
  <rect class="cluster" x="2060" y="40" width="300" height="400" rx="14" ry="14" fill="#F7F4FF" stroke="#8064A2"/>
  <text x="2210" y="68" text-anchor="middle" font-size="22" font-weight="bold">Templates &amp; Registry</text>

  <!-- Data Layer -->
  <rect class="cluster" x="540" y="460" width="560" height="540" rx="14" ry="14" fill="#F0FBFF" stroke="#4BACC6"/>
  <text x="820" y="488" text-anchor="middle" font-size="22" font-weight="bold">Data Layer</text>

  <!-- External -->
  <rect class="cluster" x="2060" y="480" width="300" height="340" rx="14" ry="14" fill="#FFF7EF" stroke="#F79646"/>
  <text x="2210" y="508" text-anchor="middle" font-size="22" font-weight="bold">External Services</text>

  <!-- Nodes: Frontend -->
  <!-- User -->
  <rect class="node" x="70" y="120" width="140" height="60" rx="10" ry="10" fill="#E6F2FF" stroke="#4F81BD"/>
  <text x="140" y="156" class="label" text-anchor="middle">User</text>

  <!-- Streamlit -->
  <rect class="node" x="240" y="120" width="300" height="160" rx="10" ry="10" fill="#EEF5FF" stroke="#4F81BD"/>
  <text x="390" y="152" class="label" text-anchor="middle">Streamlit App</text>
  <text x="390" y="178" class="sublabel" text-anchor="middle">Main View, Sidebar, Chat Input</text>

  <!-- Nodes: API -->
  <rect class="node" x="860" y="110" width="200" height="60" rx="10" ry="10" fill="#F2FAE8" stroke="#7CB342"/>
  <text x="960" y="146" class="label" text-anchor="middle">FastAPI App</text>

  <rect class="node" x="600" y="110" width="220" height="60" rx="10" ry="10" fill="#F8FFF0" stroke="#7CB342"/>
  <text x="710" y="146" class="label" text-anchor="middle">/documents Router</text>

  <rect class="node" x="600" y="200" width="220" height="60" rx="10" ry="10" fill="#F8FFF0" stroke="#7CB342"/>
  <text x="710" y="236" class="label" text-anchor="middle">/qa Router</text>

  <rect class="node" x="600" y="290" width="220" height="60" rx="10" ry="10" fill="#F8FFF0" stroke="#7CB342"/>
  <text x="710" y="326" class="label" text-anchor="middle">/users Router</text>

  <!-- Nodes: Services -->
  <rect class="node" x="1150" y="110" width="260" height="70" rx="10" ry="10" fill="#FDE9D9" stroke="#C0504D"/>
  <text x="1280" y="140" class="label" text-anchor="middle">LangGraph Orchestrator</text>
  <text x="1280" y="166" class="sublabel" text-anchor="middle">graph_builder.py</text>

  <rect class="node" x="1160" y="220" width="230" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1275" y="256" class="label" text-anchor="middle">Embedder</text>

  <rect class="node" x="1160" y="300" width="230" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1275" y="336" class="label" text-anchor="middle">Summarizer Agent</text>

  <rect class="node" x="1160" y="380" width="230" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1275" y="416" class="label" text-anchor="middle">Classifier Agent</text>

  <rect class="node" x="1160" y="460" width="230" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1275" y="496" class="label" text-anchor="middle">Model Extractor Agent</text>

  <rect class="node" x="1160" y="540" width="260" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1290" y="576" class="label" text-anchor="middle">Basecode Service</text>

  <rect class="node" x="1470" y="540" width="260" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1600" y="576" class="label" text-anchor="middle">LLM Codegen Assist</text>

  <rect class="node" x="1160" y="660" width="260" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1290" y="690" class="label" text-anchor="middle">Quality Reflection</text>
  <text x="1290" y="712" class="sublabel" text-anchor="middle">AST checks, fixes</text>

  <rect class="node" x="1470" y="660" width="260" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1600" y="690" class="label" text-anchor="middle">LangGraph Reflection</text>
  <text x="1600" y="712" class="sublabel" text-anchor="middle">feedback loop</text>

  <rect class="node" x="1470" y="220" width="260" height="60" rx="10" ry="10" fill="#FFF2E6" stroke="#C0504D"/>
  <text x="1600" y="256" class="label" text-anchor="middle">QA Agent</text>

  <!-- Nodes: Templates -->
  <rect class="node" x="2080" y="130" width="260" height="60" rx="10" ry="10" fill="#F3F0FF" stroke="#8064A2"/>
  <text x="2210" y="166" class="label" text-anchor="middle">Template Registry</text>
  <text x="2210" y="188" class="sublabel" text-anchor="middle">template_registry.py</text>

  <rect class="node" x="2080" y="230" width="260" height="120" rx="10" ry="10" fill="#F3F0FF" stroke="#8064A2"/>
  <text x="2210" y="262" class="label" text-anchor="middle">Jinja Templates (*.j2)</text>
  <text x="2210" y="286" class="sublabel" text-anchor="middle">transformer, cnn_family, rnn_seq, ...</text>

  <!-- Nodes: Data -->
  <rect class="node" x="590" y="550" width="450" height="120" rx="10" ry="10" fill="#E6FAFF" stroke="#4BACC6"/>
  <text x="815" y="586" class="label" text-anchor="middle">PostgreSQL</text>
  <text x="815" y="610" class="sublabel" text-anchor="middle">users, documents, qa_history</text>

  <rect class="node" x="590" y="710" width="450" height="120" rx="10" ry="10" fill="#E6FAFF" stroke="#4BACC6"/>
  <text x="815" y="746" class="label" text-anchor="middle">Vector Store</text>
  <text x="815" y="770" class="sublabel" text-anchor="middle">Chroma / FAISS</text>

  <rect class="node" x="590" y="870" width="450" height="120" rx="10" ry="10" fill="#E6FAFF" stroke="#4BACC6"/>
  <text x="815" y="906" class="label" text-anchor="middle">Artifacts</text>
  <text x="815" y="930" class="sublabel" text-anchor="middle">base_code.py, logs</text>

  <!-- Nodes: External -->
  <rect class="node" x="2080" y="560" width="260" height="120" rx="10" ry="10" fill="#FFF4E6" stroke="#F79646"/>
  <text x="2210" y="596" class="label" text-anchor="middle">Azure OpenAI</text>
  <text x="2210" y="620" class="sublabel" text-anchor="middle">GPT-4.1, GPT-4o-mini, Embeddings</text>

  <!-- Edges: Frontend -> API -->
  <path class="edge" d="M 210 150 L 240 150" marker-end="url(#arrow)"/>
  <text x="225" y="140" font-size="16">Interact</text>

  <path class="edge" d="M 540 190 L 600 190" marker-end="url(#arrow)"/>
  <text x="570" y="180" font-size="16">Upload PDF / Manage Documents</text>

  <path class="edge" d="M 540 230 L 600 230" marker-end="url(#arrow)"/>
  <text x="570" y="220" font-size="16">Ask Questions</text>

  <path class="edge" d="M 540 270 L 600 270" marker-end="url(#arrow)"/>
  <text x="570" y="260" font-size="16">User Session / Profile</text>

  <!-- Routers dashed to FastAPI -->
  <path class="edge dashed" d="M 820 140 L 860 140" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 820 230 L 860 140" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 820 320 L 860 140" marker-end="url(#arrow)"/>

  <!-- API -> Services -->
  <path class="edge" d="M 820 140 L 1150 145" marker-end="url(#arrow)"/>
  <text x="1000" y="130" font-size="16">Start Analysis Pipeline</text>

  <path class="edge" d="M 820 230 L 1470 250" marker-end="url(#arrow)"/>
  <text x="1120" y="220" font-size="16">Handle Q&amp;A</text>

  <!-- Orchestration flow -->
  <path class="edge" d="M 1410 145 L 1160 250" marker-end="url(#arrow)"/>
  <text x="1290" y="200" font-size="16">Step 1: Create embeddings</text>

  <path class="edge" d="M 1275 280 L 815 710" marker-end="url(#arrow)"/>
  <text x="1030" y="520" font-size="16">Index / Store</text>

  <path class="edge" d="M 1280 180 L 1275 300" marker-end="url(#arrow)"/>
  <text x="1260" y="240" font-size="16">Step 2: Summarize</text>

  <path class="edge" d="M 1275 360 L 815 550" marker-end="url(#arrow)"/>
  <text x="1040" y="480" font-size="16">Persist summary</text>

  <path class="edge" d="M 1280 180 L 1275 380" marker-end="url(#arrow)"/>
  <text x="1260" y="320" font-size="16">Step 3: Classify domain/model</text>

  <path class="edge" d="M 1275 440 L 815 550" marker-end="url(#arrow)"/>
  <text x="1040" y="540" font-size="16">Persist domain</text>

  <path class="edge" d="M 1280 180 L 1275 460" marker-end="url(#arrow)"/>
  <text x="1260" y="400" font-size="16">Step 4: Extract structured spec</text>

  <path class="edge" d="M 1390 490 L 1470 570" marker-end="url(#arrow)"/>
  <text x="1430" y="520" font-size="16">Spec → Codegen payloads</text>

  <path class="edge" d="M 1410 145 L 1160 570" marker-end="url(#arrow)"/>
  <text x="1260" y="520" font-size="16">Step 5: Build base code</text>

  <!-- Codegen interactions -->
  <path class="edge" d="M 1420 570 L 2080 160" marker-end="url(#arrow)"/>
  <text x="1710" y="540" font-size="16">Select template key</text>

  <path class="edge dashed" d="M 1730 570 L 2080 160" marker-end="url(#arrow)"/>
  <text x="1900" y="540" font-size="16">Lookup mapping</text>

  <path class="edge" d="M 1730 570 L 2080 290" marker-end="url(#arrow)"/>
  <text x="1900" y="590" font-size="16">Resolve CUSTOM_BLOCK / slots</text>

  <path class="edge" d="M 1420 570 L 1470 570" marker-end="url(#arrow)"/>
  <text x="1445" y="560" font-size="16">Combine rendered blocks</text>

  <path class="edge" d="M 1290 600 L 815 870" marker-end="url(#arrow)"/>
  <text x="1050" y="760" font-size="16">Emit base_code.py</text>

  <!-- Quality & reflection loop -->
  <path class="edge" d="M 1290 600 L 1290 660" marker-end="url(#arrow)"/>
  <text x="1300" y="635" font-size="16">Static checks (AST)</text>

  <path class="edge dashed" d="M 1420 690 L 1470 690" marker-end="url(#arrow)"/>
  <text x="1445" y="680" font-size="16">If fail: propose fixes</text>

  <path class="edge dashed" d="M 1730 690 L 1730 570" marker-end="url(#arrow)"/>
  <text x="1745" y="635" font-size="16">Apply patches</text>

  <path class="edge" d="M 1250 690 L 820 170" marker-end="url(#arrow)"/>
  <text x="1040" y="520" font-size="16">If pass: return code path</text>

  <!-- QA flow -->
  <path class="edge" d="M 1600 250 L 815 710" marker-end="url(#arrow)"/>
  <text x="1200" y="480" font-size="16">Retrieve context</text>

  <path class="edge" d="M 1600 250 L 815 550" marker-end="url(#arrow)"/>
  <text x="1170" y="430" font-size="16">Use summary/domain/spec</text>

  <path class="edge dashed" d="M 1600 250 L 815 870" marker-end="url(#arrow)"/>
  <text x="1180" y="620" font-size="16">(Optional) Read base code</text>

  <path class="edge" d="M 1730 250 L 820 230" marker-end="url(#arrow)"/>
  <text x="1200" y="240" font-size="16">Answer</text>

  <!-- API <-> DB -->
  <path class="edge" d="M 960 170 L 815 550" marker-end="url(#arrow)"/>
  <text x="900" y="360" font-size="16">CRUD users/documents/qa_history</text>

  <!-- LLM usage (dashed) -->
  <path class="edge dashed" d="M 1390 250 L 2080 620" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 1275 330 L 2080 620" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 1275 410 L 2080 620" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 1275 490 L 2080 620" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 1730 570 L 2080 620" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 1600 250 L 2080 620" marker-end="url(#arrow)"/>
  <path class="edge dashed" d="M 1600 690 L 2080 620" marker-end="url(#arrow)"/>
  <text x="2050" y="610" font-size="16">LLM calls</text>

  <!-- Frontend display -->
  <path class="edge" d="M 600 170 L 240 200" marker-end="url(#arrow)"/>
  <text x="420" y="190" font-size="16">Show summary/domain/base code</text>

  <path class="edge" d="M 600 230 L 240 230" marker-end="url(#arrow)"/>
  <text x="420" y="220" font-size="16">Show answers</text>
</svg>
