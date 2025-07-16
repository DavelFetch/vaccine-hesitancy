# Vaccine Hesitancy Insights Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![domain:data](https://img.shields.io/badge/data-3D8BD3)

## Description

This AI agent provides advanced analysis and insights on vaccine hesitancy data using Supabase. It can answer custom analytical queries, summarize trends, and extract actionable insights from vaccine hesitancy datasets. Input your query or request and receive a detailed, AI-powered response.

## Features

- Advanced analysis of vaccine hesitancy data
- Support for custom user queries (natural language)
- Natural language summaries and explanations
- Insight extraction from structured datasets
- Easy integration into analytics and reporting workflows

## Technical Implementation

- **Framework**: Fetch.ai uAgents, Protocols, and ASI-1 Mini
- **Database**: Supabase (PostgreSQL) for data storage and retrieval
- **AI Model**: ASI-1 Mini (for reasoning, summarization, and query planning)
- **Architecture**: User query → ASI-1 prompt → SQL tool call → Data retrieval → AI summary
- **Key Components**:
  - Natural language to SQL query translation (via ASI-1 tools)
  - Data aggregation and filtering
  - Statistical and trend analysis
  - Insight and summary generation

## Input Data Model

```python
class VHAgentRequest(Model):
    query: str  # User's analytical or data query (natural language)
```

## Output Data Model

```python
class VHAgentResponse(Model):
    response: str  # Analytical summary, answer, or extracted insights
    status: str    # 'success', 'error', or other status
``` 