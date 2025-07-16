# COVID-19 Vaccine Resource Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![domain:health](https://img.shields.io/badge/health-3D8BD3)

## Description

This AI agent is designed to support the **Moderna team** and public health professionals with **comprehensive, evidence-based information about COVID-19 vaccines**—including the latest guidance, safety, efficacy, and policy updates.

While COVID-19 is the primary focus, the agent is also equipped with authoritative resources on general immunization best practices, influenza, and childhood vaccines, drawing from CDC, UK Health Security Agency, WHO, NHS, and UK Parliament sources.

## Features

- **COVID-19 vaccine expertise:** Up-to-date guidance, variant information, and policy recommendations.
- **Broader immunization support:** General best practices, flu, and childhood vaccine data.
- **Real-time search** through curated medical documentation.
- **Evidence-based responses** citing official guidelines and sources.
- **Natural language query processing** with medical term enhancement.

## Technical Implementation

- **Framework**: Fetch.ai uAgents, Chat Protocol, and ASI-1 Mini
- **Database**: Qdrant Vector Database for semantic document search
- **Embeddings**: OpenAI text-embedding-3-small for consistent vector search
- **AI Model**: ASI-1 Mini for intelligent query processing and response generation
- **Architecture**: User query → Query enhancement → Vector search → Document retrieval → AI-powered response
- **Key Components**:
  - Medical terminology expansion and synonym matching
  - Semantic vector search with relevance scoring
  - Document chunking and metadata preservation
  - Source citation and evidence-based responses

## Database Scope

The agent’s knowledge base is **centered on COVID-19**, but also includes:

1. **Green Book Chapter 14a: COVID-19 (SARS-CoV-2)**  
   *UK Health Security Agency* — COVID-19 disease profile, variants, vaccination program updates, and guidance.

2. **CDC MMWR: COVID-19 Vaccine Recommendations 2024–2025**  
   *CDC* — Latest ACIP recommendations for COVID vaccines in the U.S., including Omicron variants.

3. **General Best Practice Guidelines for Immunization**  
   *CDC* — Foundational immunization practices for all vaccines.

4. **WHO Position Paper: Vaccines Against Influenza (May 2022)**  
   *WHO* — Global influenza vaccine policies and effectiveness.

5. **Childhood Immunisation Statistics – UK (May 2025)**  
   *UK Parliament Research Briefing* — UK-wide vaccine uptake statistics.

6. **WHO Position Paper Development Process**  
   *WHO* — How global vaccine policy is developed.

7. **Why Vaccination Is Important – NHS**  
   *NHS (UK)* — Public-facing vaccine education and myth-busting.

## Input Data Model

```python
class ChatMessage(Model):
    content: List[TextContent]  # User's vaccine-related questions
    timestamp: datetime
    msg_id: UUID
```

## Output Data Model

```python
class ChatMessage(Model):
    content: List[TextContent]  # Evidence-based response with source citations
    timestamp: datetime
    msg_id: UUID
```

## Example Queries

- "What are the latest COVID-19 vaccine recommendations for 2024?"
- "What is the risk of myocarditis after COVID-19 vaccination?"
- "How do COVID-19 vaccine policies differ between the UK and US?"
- "What are the best practices for vaccine storage and handling?"
- "How effective are flu vaccines in tropical regions?"
- "What are the current childhood immunisation rates in the UK?"

## Usage

The agent uses the Fetch.ai Chat Protocol and can be accessed through:
- Direct agent-to-agent communication via uAgents
- Agentverse mailbox integration
- Chat interface for interactive conversations

## Important Notes

- **Primary focus:** COVID-19 vaccines, policy, and safety.
- **Broader support:** General immunization, flu, and childhood vaccines.
- **Sources:** CDC, UKHSA, WHO, NHS, UK Parliament.
- **Limitations:** No coverage of travel vaccines, rare vaccines, or non-official sources.
- **Medical Advice:** Always recommends consulting healthcare professionals for personalized advice.

## Environment Variables Required

```bash
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
ASI1_API_KEY=your_asi1_api_key
```

## Dependencies

```bash
pip install uagents
pip install qdrant-client
pip install openai
pip install python-dotenv
pip install requests
``` 