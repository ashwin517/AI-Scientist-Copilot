# AI Scientist Copilot v2

## Mission

Build an industrial-grade AI-first scientific assistant for scientific data analysis, modeling, simulation, optimization, and knowledge retrieval.

This is NOT a dashboard demo.

This is a portfolio-quality AI software engineering product.

---

## Product Goals

Users should be able to:

- create projects
- upload datasets
- analyze data
- train ML models
- chat with AI copilot
- upload scientific documents
- perform RAG-based Q&A
- run simulations
- optimize experiments
- perform what-if analysis

---

## Stack

Frontend:
- Next.js
- TypeScript
- TailwindCSS

Backend:
- FastAPI
- Python

Database:
- PostgreSQL

Infra:
- Docker Compose

Local AI:
- Ollama

Preferred local model:
- llama3.2:3b initially

---

## Constraints

- No paid OpenAI API for now
- Local LLM required
- RTX 3070 available
- Maintain clean architecture

---

## Architecture Principles

1. AI-first product
2. Modular services
3. Persistent workspace memory
4. Scientific realism
5. Production-minded code structure

---

## Planned Architecture

Frontend:
- app shell
- project workspace
- chat interface
- dataset explorer
- document manager
- simulator UI

Backend services:
- project service
- dataset service
- model service
- chat service
- rag service
- simulator service
- optimization service

Persistence:
- PostgreSQL
- pgvector later

AI:
- Ollama initially
- tool calling later

---

## Phased Roadmap

Phase 1:
Foundation
- repo scaffold
- Docker
- PostgreSQL
- FastAPI
- Next.js
- clean architecture

Phase 2:
Workspace
- projects
- datasets
- persistence
- dataset loading

Phase 3:
ML
- profiling
- baseline models
- training workflows

Phase 4:
AI Copilot
- Ollama integration
- dataset-aware prompts
- model-aware chat

Phase 5:
RAG
- document upload
- embeddings
- vector search
- grounded Q&A

Phase 6:
Tools
- model interrogation
- experiment assistant
- optimization tools

Phase 7:
Simulation
- equation parsing
- ODE solving
- synthetic data

Phase 8:
Optimization
- what-if analysis
- Bayesian optimization

Phase 9:
Productionization
- auth
- tests
- CI/CD
- deployment

---

## Codex Rules

When modifying code:
- read this file first
- preserve working functionality
- make focused changes
- avoid full rewrites unless requested
- maintain clean architecture
- do not introduce unnecessary frameworks