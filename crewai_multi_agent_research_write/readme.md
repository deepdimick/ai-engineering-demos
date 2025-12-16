# CrewAI Multi-Agent Research & Writing System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-0.80.0-purple)
![LangChain](https://img.shields.io/badge/LangChain-0.3.20-green)
![License](https://img.shields.io/badge/License-MIT-green)

An autonomous multi-agent system that orchestrates AI agents to perform research and content creation tasks. Uses CrewAI framework with LLaMA 3.3 70B to coordinate a research analyst and content writer in a sequential workflow.

## Problem Statement

Manual research and content creation requires significant time and coordination between specialists. This project automates the workflow using specialized AI agents that collaborate autonomously to gather information and produce polished content.

## Features

- Multi-agent orchestration with role-based specialization
- Web search integration via SerperDev API
- Sequential task processing with inter-agent delegation
- Token usage tracking and performance metrics
- Configurable LLM backend (WatsonX/LLaMA 3.3)

## Quick Start

### Prerequisites

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### API Keys

Sign up for a [SerperDev API key](https://serper.dev/) for web search functionality.

### Run

**Option 1: Python Script**
```bash
python crewai_multi_agent_research_write.py
```

**Option 2: Jupyter Notebook**
```bash
jupyter notebook CrewAI_Multi-Agent_Research-Write.ipynb
```

You'll be prompted to enter your SERPER_API_KEY when running either version.

## System Architecture

| Component | Details |
|-----------|---------|
| **Research Agent** | Senior Research Analyst with web search tool |
| **Writer Agent** | Tech Content Strategist with delegation capability |
| **LLM** | LLaMA 3.3 70B Instruct (via WatsonX) |
| **Workflow** | Sequential process (research → write) |
| **Tools** | SerperDevTool for web search |

## Project Structure

```
├── crewai_multi_agent_research_write.py    # Production Python script
├── CrewAI_Multi-Agent_Research-Write.ipynb # Interactive Jupyter notebook
├── requirements.txt                         # Project dependencies
└── README.md                                # Documentation
```

### File Descriptions

- **crewai_multi_agent_research_write.py**: Clean, production-ready Python script with proper structure and error handling. Suitable for integration into larger systems or command-line execution.
- **CrewAI_Multi-Agent_Research-Write.ipynb**: Interactive notebook version with detailed outputs and experimentation capabilities. Ideal for exploration and learning.
- **requirements.txt**: All project dependencies with pinned versions for reproducibility.

## Agent Roles

### Research Agent
- **Role:** Senior Research Analyst
- **Goal:** Uncover cutting-edge information with comprehensive analysis
- **Tools:** Web search via SerperDev
- **Capabilities:** Trend identification, source verification, insight synthesis

### Writer Agent
- **Role:** Tech Content Strategist  
- **Goal:** Craft well-structured and engaging content
- **Capabilities:** Narrative translation, audience targeting, delegation
- **Output:** 4-paragraph blog posts optimized for tech enthusiasts

## Workflow

1. **Research Task:** Agent analyzes topic, identifies trends and technologies
2. **Writing Task:** Agent transforms research into engaging blog post
3. **Output Analysis:** View individual task outputs and token usage metrics

## Results

The system produces structured research reports and polished blog content. See the notebook for sample outputs on "Latest Generative AI breakthroughs" and token usage statistics.

## Skills Demonstrated

- Agentic AI system design
- Multi-agent orchestration with CrewAI
- LLM integration and prompt engineering
- Tool usage and API integration
- Sequential workflow automation

## License

MIT

## Acknowledgments

- Project completed as part of IBM Advanced RAG & Agentic AI Certification
- Framework: [CrewAI](https://www.crewai.com/)
- LLM: Meta LLaMA 3.3 70B via IBM WatsonX
- Search API: [SerperDev](https://serper.dev/)
