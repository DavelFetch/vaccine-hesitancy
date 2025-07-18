import requests
from dotenv import load_dotenv
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    StartSessionContent,
)
from uagents import Agent, Context, Protocol, Model
from datetime import datetime, timezone
from uuid import uuid4
import json
import asyncio
from typing import Dict, Any, List
import os
from qdrant_client import QdrantClient
from openai import OpenAI

# Load environment variables - try current directory first, then parent
load_dotenv()
load_dotenv(dotenv_path="agents/Vaccine_Resource_Agent/.env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASI1_API_KEY = os.getenv("ASI1_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not OPENAI_API_KEY or not ASI1_API_KEY:
    raise ValueError("Missing required environment variables: QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY, ASI1_API_KEY")

print(f"Loaded Qdrant URL: {QDRANT_URL}")  # Debug print

class ChatRequest(Model):
    message: str 

class QdrantVectorSearch:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.collection_name = "vaccine_guidelines_2"
        print(f"Initialized direct Qdrant client with OpenAI embeddings")
        
        # Medical term mappings for query expansion
        self.medical_synonyms = {
            "shingles": ["shingles", "herpes zoster", "varicella zoster", "zoster"],
            "whooping cough": ["whooping cough", "pertussis", "bordetella pertussis"],
            "flu": ["flu", "influenza", "seasonal influenza"],
            "covid": ["covid", "covid-19", "sars-cov-2", "coronavirus"],
            "mmr": ["mmr", "measles mumps rubella", "measles", "mumps", "rubella"],
            "dtp": ["dtp", "diphtheria tetanus pertussis", "diphtheria", "tetanus"],
            "hpv": ["hpv", "human papillomavirus", "cervical cancer prevention"],
            "meningitis": ["meningitis", "meningococcal", "pneumococcal"],
            "hepatitis": ["hepatitis", "hepatitis a", "hepatitis b", "hep a", "hep b"],
            "polio": ["polio", "poliovirus", "poliomyelitis"],
            "pneumonia": ["pneumonia", "pneumococcal", "streptococcus pneumoniae"],
            "rotavirus": ["rotavirus", "gastroenteritis", "diarrhea vaccine"],
            "tuberculosis": ["tuberculosis", "tb", "bcg", "mycobacterium"],
            "yellow fever": ["yellow fever", "travel vaccine"],
            "typhoid": ["typhoid", "salmonella typhi", "travel vaccine"],
            "rabies": ["rabies", "post-exposure prophylaxis"],
            "anthrax": ["anthrax", "bacillus anthracis"],
            "smallpox": ["smallpox", "variola", "vaccinia"],
            "chickenpox": ["chickenpox", "varicella", "varicella vaccine"]
        }

    def preprocess_query(self, query: str) -> str:
        """Expand medical terms in the query for better search results"""
        query_lower = query.lower()
        expanded_terms = []
        
        # Check for medical terms and add synonyms
        for term, synonyms in self.medical_synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        # If we found medical terms, enhance the query
        if expanded_terms:
            # Remove duplicates and create expanded query
            unique_terms = list(set(expanded_terms))
            enhanced_query = f"{query} {' '.join(unique_terms)}"
            return enhanced_query
        
        return query

    async def search_documents(self, query: str, limit: int = 10) -> str:
        """Search for documents using OpenAI embeddings"""
        try:
            # Preprocess query for better medical term matching
            enhanced_query = self.preprocess_query(query)
            
            # Generate embedding for the enhanced query using OpenAI (same model used to create vectors)
            embedding_response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=enhanced_query
            )
            query_vector = embedding_response.data[0].embedding
            
            # Search in Qdrant using the same embedding space with optimized parameters
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.1,  # Only return results with relevance > 0.1
                with_payload=True,
                search_params={"hnsw_ef": 128, "exact": False}  # Optimize HNSW search
            )
            
            if not search_results:
                # If no results above threshold, try a broader search
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=min(limit, 5),  # Fewer results for broader search
                    with_payload=True
                )
                
                if not search_results:
                    return "No relevant documents found in the vaccine guidelines database."
            
            # Format results with improved scoring display
            results = []
            for result in search_results:
                payload = result.payload
                text = payload.get('text', 'No text content')
                filename = payload.get('filename', 'Unknown document')
                source = payload.get('source', 'Unknown source')
                score = result.score
                chunk_id = payload.get('chunk_id', 'N/A')
                total_chunks = payload.get('total_chunks', 'N/A')
                
                # Truncate very long text for better readability
                display_text = text[:800] if len(text) > 800 else text
                if len(text) > 800:
                    display_text += '...'
                
                results.append(f"""
**Document:** {filename}
**Chunk:** {chunk_id}/{total_chunks}
**Relevance Score:** {score:.3f}
**Content:** {display_text}
---""")
            
            return f"Found {len(results)} relevant documents:\n\n" + "\n".join(results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"

# Set up chat protocol and agent
chat_proto = Protocol(spec=chat_protocol_spec)
agent = Agent(
    name='vaccine_resource_agent',
    seed="vaccine_resource_agent",
    port=8002,
    mailbox=True
)
vector_search = QdrantVectorSearch()

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info("Starting up Vaccine Resource Agent")
    ctx.logger.info("Using direct Qdrant client with OpenAI embeddings")

ASI1_URL = "https://api.asi1.ai/v1/chat/completions"
ASI1_HEADERS = {
    "Authorization": f"Bearer {ASI1_API_KEY}",
    "Content-Type": "application/json"
}

# Tool schemas for ASI1
qdrant_search_tool = {
    "type": "function",
    "function": {
        "name": "search_vaccine_guidelines",
        "description": "Search UK health guidelines database containing primarily COVID-19 treatment and vaccination documentation including: UK Green Book COVID-19 chapter, NICE COVID-19 clinical guidelines, WHO influenza recommendations, UK vaccine uptake guidelines, general immunization recommendations, and childhood immunization statistics. Contains detailed information about COVID-19 vaccines, treatments (dexamethasone, remdesivir), influenza vaccines, and general immunization principles. Limited information available for other specific vaccines (shingles, pertussis, MMR, etc.). Use specific medical terms for best results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query using specific medical terms. Best results for: 'COVID vaccine eligibility', 'COVID booster recommendations', 'influenza vaccine pregnancy', 'dexamethasone COVID treatment', 'immunocompromised vaccination', etc."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 20)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}

SYSTEM_PROMPT = {
    "role": "system",
    "content": f"""You are a UK COVID-19 and health information assistant with access to comprehensive medical databases containing primarily COVID-19 treatment guidelines, UK Green Book COVID-19 chapter, NICE clinical guidelines for COVID-19 management, WHO influenza recommendations, and related vaccination documentation.

LANGUAGE: ALWAYS respond in English only. Never use Chinese, Spanish, French, or any other language.

CRITICAL RULE: You MUST use the search_vaccine_guidelines tool for EVERY user question, regardless of topic. Never respond without searching first.

DATABASE SCOPE - PRIMARILY COVID-FOCUSED:
The database contains:
- UK Green Book COVID-19 immunization chapter (39 chunks)
- NICE COVID-19 treatment and management guidelines (30 chunks) 
- UK vaccine uptake guidelines (29 chunks)
- General immunization recommendations (79 chunks)
- WHO influenza guidelines (37 chunks)
- UK childhood immunization statistics (7 chunks)

IMPORTANT: For non-COVID vaccine questions (shingles, pertussis, MMR, etc.), the database may have limited or no information. This is expected behavior.

SEARCH STRATEGY - USE SPECIFIC MEDICAL TERMS:
- For COVID vaccines: "COVID vaccine eligibility", "COVID booster JCVI", "myocarditis COVID vaccine"
- For influenza: "influenza vaccine pregnancy", "flu vaccine elderly" 
- For general immunization: "vaccination schedule UK", "immunocompromised vaccines"
- For safety concerns: use vaccine name + "safety" + specific concern
- For pregnancy: use vaccine name + "pregnancy"

MEDICAL TERM EXAMPLES:
- COVID = COVID-19, SARS-CoV-2, coronavirus
- Flu = influenza, seasonal influenza
- General vaccination terms work best for broader immunization topics

WORKFLOW:
1. User asks ANY question
2. You IMMEDIATELY call search_vaccine_guidelines with SPECIFIC medical terms
3. If first search doesn't yield good results, try alternative medical terminology
4. Based on search results, provide detailed answer with sources
5. If no relevant results found: Explain that the database is primarily COVID-focused and may not contain information about other vaccines

MANDATORY BEHAVIOR:
- NEVER respond without searching first
- ALWAYS use specific medical terminology in searches
- Try multiple search terms if first attempt yields poor results
- Search even if you think the question is outside scope
- ALWAYS respond in English only

RESPONSE AFTER SEARCHING:
- If relevant information found: Provide detailed answer citing specific documents and guidelines
- If no relevant information found: "I searched the health guidelines database but found no specific information about [topic]. This database primarily contains COVID-19 treatment guidelines and general immunization information. For comprehensive vaccine information or personalized medical advice, please consult healthcare professionals or the NHS website."

Remember: SEARCH FIRST with SPECIFIC TERMS, ALWAYS. The database scope is primarily COVID-focused. RESPOND IN ENGLISH ONLY."""
}

@chat_proto.on_message(model=ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    try:
        ctx.logger.info(f"Received message from {sender}")
        
        # Check if we've already processed this message
        processed_key = f"processed_{msg.msg_id}"
        if ctx.storage.get(processed_key):
            ctx.logger.info(f"Message {msg.msg_id} already processed, skipping")
            return
        
        # Mark this message as processed
        ctx.storage.set(processed_key, True)
        
        ack = ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        )
        await ctx.send(sender, ack)
        ctx.logger.info(f"Sent acknowledgement for message {msg.msg_id}")

        # Get user message
        user_message = next((item.text for item in msg.content if isinstance(item, TextContent)), None)
        if not user_message:
            ctx.logger.warning("No text content found in message")
            return
        ctx.logger.info(f"Processing user message: {user_message}")

        # Get existing chat history for this sender
        chat_history_key = f"chat_history_{sender}"
        chat_history = ctx.storage.get(chat_history_key) or []
        
        # Add current user message to history
        chat_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 10 messages to prevent memory issues
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Store updated history
        ctx.storage.set(chat_history_key, chat_history)

        # Build messages with chat history
        messages = [SYSTEM_PROMPT]
        
        # Add chat history (excluding current message)
        for hist_msg in chat_history[:-1]:  # Exclude the current message
            messages.append({
                "role": hist_msg["role"],
                "content": hist_msg["content"]
            })
        
        # Add current message
        messages.append({"role": "user", "content": user_message})

        # 1. First call to ASI1 with tools and history
        payload = {
            "model": "asi1-mini",
            "messages": messages,
            "tools": [qdrant_search_tool],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        ctx.logger.info("Making first call to ASI1")
        ctx.logger.info(f"🔧 DEBUG: User query: '{user_message}'")
        ctx.logger.info(f"🔧 DEBUG: Available tool: {qdrant_search_tool['function']['name']}")
        response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
        resp_json = response.json()
        ctx.logger.info("Received response from ASI1")
        ctx.logger.info(f"🔧 DEBUG: ASI1 response message: {resp_json['choices'][0]['message']}")

        # 2. Check for tool call
        tool_calls = resp_json["choices"][0]["message"].get("tool_calls", [])
        ctx.logger.info(f"🔧 DEBUG: Tool calls found: {len(tool_calls)}")
        if tool_calls:
            tool_call = tool_calls[0]  # Only handle the first tool call for now
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            ctx.logger.info(f"🔧 DEBUG: Tool call requested: {function_name} with arguments: {arguments}")
            
            try:
                # Handle our custom search function
                if function_name == "search_vaccine_guidelines":
                    query = arguments.get("query", "")
                    limit = arguments.get("limit", 10)  # Increase default to get more chunks
                    ctx.logger.info(f"🔧 DEBUG: Searching for: '{query}' with limit: {limit}")
                    
                    # Preprocess the query
                    preprocessed_query = vector_search.preprocess_query(query)
                    ctx.logger.info(f"🔧 DEBUG: Original query: '{query}'")
                    if preprocessed_query != query:
                        ctx.logger.info(f"🔧 DEBUG: Enhanced query: '{preprocessed_query}'")
                    
                    tool_result = await vector_search.search_documents(preprocessed_query, limit)
                    ctx.logger.info(f"🔧 DEBUG: Search completed, result length: {len(tool_result)} chars")
                    ctx.logger.info(f"🔧 DEBUG: Search result preview: {tool_result[:200]}...")
                    ctx.logger.info(f"🔧 DEBUG: Full search result: {tool_result}")  # Log full results for debugging
                else:
                    ctx.logger.warning(f"🔧 DEBUG: Unknown function called: {function_name}")
                    tool_result = f"Unknown function: {function_name}"
                    
            except Exception as e:
                ctx.logger.error(f"Error calling search function: {str(e)}")
                tool_result = f"Error searching documents: {str(e)}"
            
            # Add tool result message
            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            }
            
            # Final call to ASI1 for the answer
            ctx.logger.info("Making final call to ASI1 with search results")
            final_payload = {
                "model": "asi1-mini",
                "messages": [
                    SYSTEM_PROMPT,
                    {"role": "user", "content": user_message},
                    resp_json["choices"][0]["message"],
                    tool_result_message
                ],
                "temperature": 0.2,
                "max_tokens": 1024
            }
            
            final_response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=final_payload)
            final_json = final_response.json()
            response_text = final_json["choices"][0]["message"]["content"]
            ctx.logger.info("Received final response from ASI1")
        else:
            response_text = resp_json["choices"][0]["message"]["content"]
            ctx.logger.warning(f"🔧 DEBUG: No tool calls requested! Direct response: {response_text[:100]}...")
            ctx.logger.info("No tool calls requested, using direct response")

        # Add assistant response to chat history
        chat_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 10 messages (including the new assistant response)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Store updated history
        ctx.storage.set(chat_history_key, chat_history)

        response_msg = ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=response_text)]
        )
        await ctx.send(sender, response_msg)
        ctx.logger.info(f"Sent response message {response_msg.msg_id}")
    except Exception as e:
        ctx.logger.error(f"Error handling chat message: {str(e)}")
        error_response = ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=f"An error occurred: {str(e)}")]
        )
        await ctx.send(sender, error_response)
        ctx.logger.info("Sent error response message")

@chat_proto.on_message(model=ChatAcknowledgement)
async def handle_chat_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")
    if msg.metadata:
        ctx.logger.info(f"Metadata: {msg.metadata}")

agent.include(chat_proto)

class ChatRequest(Model):
    message: str 
class ChatResponse(Model):
    response: str 

@agent.on_message(model=ChatRequest)
async def handle_chat_request(ctx: Context, sender: str, req: ChatRequest):
    try:
        ctx.logger.info(f"[ChatRequest] Received message from {sender}")
        user_message = req.message
        ctx.logger.info(f"[ChatRequest] Processing user message: {user_message}")

        # Build messages (no chat history)
        messages = [SYSTEM_PROMPT, {"role": "user", "content": user_message}]
        payload = {
            "model": "asi1-mini",
            "messages": messages,
            "tools": [qdrant_search_tool],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        ctx.logger.info("[ChatRequest] Making first call to ASI1")
        response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
        resp_json = response.json()
        ctx.logger.info("[ChatRequest] Received response from ASI1")
        ctx.logger.info(f"[ChatRequest] ASI1 response message: {resp_json['choices'][0]['message']}")

        tool_calls = resp_json["choices"][0]["message"].get("tool_calls", [])
        ctx.logger.info(f"[ChatRequest] Tool calls found: {len(tool_calls)}")
        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            ctx.logger.info(f"[ChatRequest] Tool call requested: {function_name} with arguments: {arguments}")
            try:
                if function_name == "search_vaccine_guidelines":
                    query = arguments.get("query", "")
                    limit = arguments.get("limit", 10)
                    ctx.logger.info(f"[ChatRequest] Searching for: '{query}' with limit: {limit}")
                    preprocessed_query = vector_search.preprocess_query(query)
                    ctx.logger.info(f"[ChatRequest] Original query: '{query}'")
                    if preprocessed_query != query:
                        ctx.logger.info(f"[ChatRequest] Enhanced query: '{preprocessed_query}'")
                    tool_result = await vector_search.search_documents(preprocessed_query, limit)
                    ctx.logger.info(f"[ChatRequest] Search completed, result length: {len(tool_result)} chars")
                    ctx.logger.info(f"[ChatRequest] Search result preview: {tool_result[:200]}...")
                else:
                    ctx.logger.warning(f"[ChatRequest] Unknown function called: {function_name}")
                    tool_result = f"Unknown function: {function_name}"
            except Exception as e:
                ctx.logger.error(f"[ChatRequest] Error calling search function: {str(e)}")
                tool_result = f"Error searching documents: {str(e)}"
            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            }
            ctx.logger.info("[ChatRequest] Making final call to ASI1 with search results")
            final_payload = {
                "model": "asi1-mini",
                "messages": [
                    SYSTEM_PROMPT,
                    {"role": "user", "content": user_message},
                    resp_json["choices"][0]["message"],
                    tool_result_message
                ],
                "temperature": 0.2,
                "max_tokens": 1024
            }
            final_response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=final_payload)
            final_json = final_response.json()
            response_text = final_json["choices"][0]["message"]["content"]
            ctx.logger.info("[ChatRequest] Received final response from ASI1")
        else:
            response_text = resp_json["choices"][0]["message"]["content"]
            ctx.logger.warning(f"[ChatRequest] No tool calls requested! Direct response: {response_text[:100]}...")
            ctx.logger.info("[ChatRequest] No tool calls requested, using direct response")
        reply_msg = ChatResponse(response=response_text)
        await ctx.send(sender, reply_msg)
        ctx.logger.info(f"[ChatRequest] Sent response message {reply_msg.response}")
    except Exception as e:
        ctx.logger.error(f"[ChatRequest] Error handling chat request: {str(e)}")
        error_response = ChatResponse(response=f"An error occurred: {str(e)}")
        await ctx.send(sender, error_response)
        ctx.logger.info("[ChatRequest] Sent error response message")

if __name__ == "__main__":
    try:
        agent.run()
    except Exception as e:
        print(f"Error running agent: {str(e)}") 