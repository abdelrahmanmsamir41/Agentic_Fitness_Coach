"""Tool-routing agent over the client's plan.
Run: python -m Retrieval_Pipeline.agent [client_id]
"""
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (ClearToolUsesEdit, ContextEditingMiddleware, SummarizationMiddleware,
                                         wrap_model_call, wrap_tool_call)
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage

from . import plan_store
from .context import TrainerContext
from .prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from .tools import TOOLS

load_dotenv()
logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # SDK's AFC warning; LangChain runs the tools, AFC is unused

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_CLIENT_ID = "client_self"
EMPTY_REPLY_RETRIES = 2
HISTORY_TOKEN_LIMIT = 3000  # past this, older turns are summarized (one extra LLM call, occasionally)
KEEP_RECENT_MESSAGES = 8  # verbatim tail after summarizing: the last 2 tool-using questions (4 messages each)
KEEP_TOOL_RESULTS = 2  # older tool outputs become "[cleared]"; the model can call the tool again
THINKING_BUDGET = 0  # thinking off (2.5-flash allows 0); dynamic thinking gave blank replies on "all dumbbell exercises", 1024 fixed it


@wrap_tool_call
async def tool_errors_to_messages(request, handler):
    """A failing tool (Mongo/Qdrant down, bad args) becomes a message the model can react to, not a crash."""
    try:
        return await handler(request)
    except Exception as error:
        return ToolMessage(content=f"Tool error: {error}", tool_call_id=request.tool_call["id"])


@wrap_model_call
async def retry_empty_replies(request, handler):
    """Gemini with tools sometimes returns a blank reply (0 tokens, finish_reason STOP) at random; ask again."""
    for _ in range(EMPTY_REPLY_RETRIES):
        response = await handler(request)
        reply = response.result[-1]
        if reply.tool_calls or reply.text.strip():
            return response
    return await handler(request)


def build_agent(checkpointer):
    model = init_chat_model(
        MODEL_NAME,
        model_provider="google_genai",
        temperature=0.2,
        timeout=600,
        max_tokens=4096,  # output incl. Gemini thinking tokens; 1000 truncated full-week answers
        streaming=True,
        thinking_budget=THINKING_BUDGET,
    )
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=TrainerContext,
        middleware=[
            SummarizationMiddleware(model, trigger=("tokens", HISTORY_TOKEN_LIMIT),
                                    keep=("messages", KEEP_RECENT_MESSAGES), summary_prompt=SUMMARY_PROMPT),
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=0, keep=KEEP_TOOL_RESULTS)]),
            retry_empty_replies,
            tool_errors_to_messages,
        ],
        checkpointer=checkpointer,
    )


async def ask(agent, question, client_id, thread_id):
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        context=TrainerContext(client_id=client_id),
    )
    return result["messages"][-1].text


async def stream_answer(agent, question, client_id, thread_id):
    """Yields the answer's text as the model produces it; tool calls and tool results are skipped."""
    text_by_message_id, last_chunk_by_message_id = {}, {}  # message id -> text yielded so far / last chunk yielded
    async for chunk, metadata in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        context=TrainerContext(client_id=client_id),
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "model" or chunk.type != "AIMessageChunk" or not chunk.text:
            continue
        # ponytail: Gemini occasionally re-sends the whole message or its last chunk; exact repeats are dropped,
        # which would also drop a genuinely repeated sentence-sized chunk (never seen in practice)
        if chunk.text in (text_by_message_id.get(chunk.id), last_chunk_by_message_id.get(chunk.id)):
            continue
        text_by_message_id[chunk.id] = text_by_message_id.get(chunk.id, "") + chunk.text
        last_chunk_by_message_id[chunk.id] = chunk.text
        yield chunk.text


async def format_turn_stats(agent, thread_id):
    """Tool calls and summed token usage of the last question, read back from the saved chat state."""
    messages = (await agent.aget_state({"configurable": {"thread_id": thread_id}})).values["messages"]
    last_question_index = max(index for index, message in enumerate(messages) if message.type == "human")
    ai_messages = [message for message in messages[last_question_index + 1:] if message.type == "ai"]
    tool_calls = [f"{call['name']}({call['args']})" for message in ai_messages for call in message.tool_calls]
    usages = [message.usage_metadata or {} for message in ai_messages]

    def sum_tokens(key, details_key=None):
        return sum((usage.get(details_key, {}) if details_key else usage).get(key, 0) or 0 for usage in usages)

    return (f"🔧 {', '.join(tool_calls) or 'no tools'}\n"
            f"🪙 in {sum_tokens('input_tokens')} (cached {sum_tokens('cache_read', 'input_token_details')}) · "
            f"out {sum_tokens('output_tokens')} (thinking {sum_tokens('reasoning', 'output_token_details')}) · "
            f"total {sum_tokens('total_tokens')} · {len(ai_messages)} model call(s)")


async def chat(client_id):
    print("⏳ loading models...")
    await asyncio.to_thread(plan_store.warm_up)
    with plan_store.chat_checkpointer() as checkpointer:
        agent = build_agent(checkpointer)
        thread_id = client_id  # ponytail: one ongoing thread per client; pass a conversation id once the app has multiple chats
        checkpointer.delete_thread(thread_id)  # start clean: a killed session never reached the finally below
        print(f"💬 chatting as {client_id} (empty line to quit)")
        try:
            while question := input("> ").strip():
                async for text in stream_answer(agent, question, client_id, thread_id):
                    print(text, end="", flush=True)
                print("\n" + await format_turn_stats(agent, thread_id) + "\n")
        finally:
            checkpointer.delete_thread(thread_id)  # conversation over: drop its history; the 30-day TTL covers hard kills
            print("🧹 chat history cleared")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(chat(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLIENT_ID))
