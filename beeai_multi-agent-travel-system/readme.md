# BeeAI Framework Multi-Agent System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![BeeAI](https://img.shields.io/badge/BeeAI-Framework-purple)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=fff)
![WatsonX](https://img.shields.io/badge/IBM-WatsonX-054ADA)
![License](https://img.shields.io/badge/License-MIT-green)

Progressive demonstrations of AI agent capabilities using the [BeeAI Framework](https://github.com/i-am-bee/beeai-framework) with IBM WatsonX and OpenAI backends.

## Problem Statement

Building effective AI agents requires understanding how to combine LLMs with tools, execution control, and multi-agent coordination. This project demonstrates these capabilities progressively, from basic chat through sophisticated multi-agent orchestration.

## Features

- Basic LLM interaction with async patterns
- Prompt templates and structured output generation
- Tool integration (Wikipedia, Weather, ThinkTool)
- Execution control via ConditionalRequirement
- Human-in-the-loop approval workflows
- Custom tool creation
- Multi-agent coordination with HandoffTool

## Quick Start

### Prerequisites

```bash
pip install beeai-framework pydantic
```

### Configuration

Edit `01_environment_setup.py` with your WatsonX or OpenAI credentials.

### Run

```bash
python 01_environment_setup.py
python 02_basic_chat.py
```

## Architecture

| Component | Details |
|-----------|---------|
| **LLM Backend** | WatsonX (Granite, Llama 4) / OpenAI |
| **Agent Type** | RequirementAgent |
| **Memory** | UnconstrainedMemory |
| **Middleware** | GlobalTrajectoryMiddleware |
| **Tools** | WikipediaTool, ThinkTool, OpenMeteoTool, Custom |

## Project Structure

```
├── 01_environment_setup.py
├── 02_basic_chat.py
├── 03_prompt_templates.py
├── 04_structured_output.py
├── 05_minimal_agent.py
├── 06_wikipedia_agent.py
├── 07_reasoning_agent.py
├── 08_controlled_execution.py
├── 09_force_after_pattern.py
├── 10_permission_workflow.py
├── 11_custom_tool.py
├── 12_multi_agent_system.py
└── README.md
```

## Demonstrations

### Foundation

| File | Description |
|------|-------------|
| `01_environment_setup.py` | WatsonX/OpenAI credential configuration |
| `02_basic_chat.py` | Basic ChatModel interaction with async patterns |
| `03_prompt_templates.py` | Dynamic prompt rendering with variable injection |
| `04_structured_output.py` | Pydantic-based structured LLM responses |

### Agent Capabilities

| File | Description |
|------|-------------|
| `05_minimal_agent.py` | RequirementAgent without tools (pure LLM reasoning) |
| `06_wikipedia_agent.py` | Agent with WikipediaTool and middleware tracking |
| `07_reasoning_agent.py` | Combined ThinkTool + WikipediaTool |
| `08_controlled_execution.py` | ConditionalRequirement for tool execution control |

### Advanced Patterns

| File | Description |
|------|-------------|
| `09_force_after_pattern.py` | Mandatory reasoning after tool calls via force_after |
| `10_permission_workflow.py` | Human-in-the-loop with AskPermissionRequirement |
| `11_custom_tool.py` | Custom SimpleCalculatorTool implementation |
| `12_multi_agent_system.py` | Multi-agent travel planner with HandoffTool coordination |

## Models Used

| Model | Provider | Use Case |
|-------|----------|----------|
| `ibm/granite-3-3-8b-instruct` | WatsonX | Chat and prompt demos |
| `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | WatsonX | Agent demonstrations |
| `gpt-5-nano` | OpenAI | Structured output |

## Skills Demonstrated

- Async programming with asyncio
- Tool integration and custom tool creation
- Execution flow control with requirements
- Human-in-the-loop security patterns
- Multi-agent orchestration and coordination

## License

MIT

## Acknowledgments

- [BeeAI Framework](https://github.com/i-am-bee/beeai-framework) by IBM
- Project completed as part of IBM AI Engineering Professional Certificate
