# glassbox-ai-workbench
Concepts of RAG, LLM, Agentic AI, GEN AI may look confusing and you may think what exactly is going on behind the scene. This application is a step-by-step wizard exposing all intermediate pipeline stages across RAG, LLM Mechanics, Generative AI, and Agentic AI workflows. It is  built using Python and Streamlit. It helps you to understand output of each step and thus grasp each concept in detail. 

Instead of hiding AI behavior behind a black-box chat window, each track walks users through every intermediate pipeline stage with 
`< Back`, `Next >`, and `Cancel` wizard navigation:

1. **RAG Pipeline Wizard:** Document selection → Overlapping text chunking → Dense vector embeddings & ChromaDB indexing → Top-K cosine similarity retrieval → Context-grounded prompt & cited answer.
   
## Data & Licensing
- Built-in behavioral psychology and kinesics reference modules are original educational summaries created for this workbench.
- User-uploaded PDFs are processed transiently in browser session memory and are not stored on disk. If testing with open-access textbooks such as OpenStax *Psychology 2e* (CC BY-NC-SA 4.0, free at https://openstax.org), please adhere to the source document's attribution and non-commercial terms.
- 
2. **LLM Mechanics Wizard:** System/User prompt configuration → Tokenization & cost breakdown → Low vs. high temperature decoding comparison → Guardrail validation → Final telemetry.
  
3. **Generative AI Studio Wizard:** Artifact selection → Enforced JSON schema → Structured synthesis → Quality & word-count audit → Downloadable Markdown export.
   
4. **Agentic AI Workflow Wizard:** Multi-step goal definition → Planner task decomposition → Deterministic Python tool execution trace → Critic reflection check → Executive decision brief.
