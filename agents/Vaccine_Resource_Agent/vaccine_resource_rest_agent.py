from uagents import Agent, Context, Model
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent
from pydantic import Field
from datetime import datetime, timezone
from uuid import uuid4
from typing import Optional
import logging

# REST agent config
REST_AGENT_PORT = 8006
VACCINE_RESOURCE_AGENT_ADDRESS = "agent1q0tds2u7q4ak8vj2pd9kn25pauuczm9pvqg50jmstuj36tvrf9c57fmj7hy"

class ChatRequest(Model):
    message: str 

class ChatResponse(Model):
    response: str 

agent = Agent(
    name="vaccine_resource_rest_agent",
    port=REST_AGENT_PORT,
    seed="vaccine_resource_rest_agent_seed_2024",
    mailbox=True
)

@agent.on_rest_post("/chat", ChatRequest, ChatResponse)
async def handle_chat(ctx: Context, req: ChatRequest) -> ChatResponse:
    ctx.logger.info(f"[REST] Received /chat request: {req.message}")
    ctx.logger.info(f"[REST] Using agent address: {VACCINE_RESOURCE_AGENT_ADDRESS}")
    # Construct ChatMessage
    chat_msg = ChatRequest(message=req.message)
    ctx.logger.info(f"[REST] Outgoing ChatMessage: {chat_msg}")
    # Relay to vaccine resource agent and wait for reply
    try:
        reply_obj = await ctx.send_and_receive(
            VACCINE_RESOURCE_AGENT_ADDRESS,
            chat_msg,
            response_type=ChatResponse,  # Specify expected response type
            timeout=60  # seconds
        )
        ctx.logger.info(f"[REST] Received reply: {reply_obj}")
        # Extract text from reply
        # reply_text = None
        # if hasattr(reply_obj, "response"):
        #     reply_text = reply_obj.response
        # Unpack tuple if needed
        if isinstance(reply_obj, tuple):
            reply_msg, _ = reply_obj
        else:
            reply_msg = reply_obj

        reply_text = getattr(reply_msg, "response", None)
        if not reply_text:
            reply_text = "No response from agent."
        return ChatResponse(response=reply_text)
    except Exception as e:
        ctx.logger.error(f"[REST] Error in send_and_receive: {str(e)}")
        return ChatResponse(response=f"Error: {str(e)}")

if __name__ == "__main__":
    agent.run() 