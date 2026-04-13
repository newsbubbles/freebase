"""Freebase Agent - CLI for testing the Freebase MCP Server.

Usage:
    python agent.py [--model MODEL]

Examples:
    python agent.py
    python agent.py --model openai/gpt-4o
    python agent.py --model anthropic/claude-3.5-sonnet
"""

import argparse
import asyncio
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

# Optional Logfire integration
try:
    import logfire
    logfire_key = os.getenv("LOGFIRE_API_KEY")
    if logfire_key:
        logfire.configure(token=logfire_key)
        logfire.instrument_openai()
except ImportError:
    pass


def parse_args():
    """Parse command line arguments."""
    default_model = "x-ai/grok-4-fast"
    parser = argparse.ArgumentParser(description="Freebase Agent - Git History Surgeon")
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model identifier to use with OpenRouter (default: {default_model})",
    )
    return parser.parse_args()


def filtered_message_history(
    result: Optional[AgentRunResult],
    limit: Optional[int] = None,
    include_tool_messages: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """Filter and limit the message history from an AgentRunResult.

    Args:
        result: The AgentRunResult object with message history
        limit: Optional int, if provided returns only system message + last N messages
        include_tool_messages: Whether to include tool messages in the history

    Returns:
        Filtered list of messages in the format expected by the agent
    """
    if result is None:
        return None

    messages: list[ModelMessage] = result.all_messages()

    # Extract system message
    system_message = next(
        (msg for msg in messages if type(msg.parts[0]) == SystemPromptPart), None
    )

    # Filter non-system messages
    non_system_messages = [
        msg for msg in messages if type(msg.parts[0]) != SystemPromptPart
    ]

    # Apply tool message filtering if requested
    if not include_tool_messages:
        non_system_messages = [
            msg
            for msg in non_system_messages
            if not any(
                isinstance(part, ToolCallPart) or isinstance(part, ToolReturnPart)
                for part in msg.parts
            )
        ]

    # Find the most recent UserPromptPart before applying limit
    latest_user_prompt_part = None
    latest_user_prompt_index = -1
    for i, msg in enumerate(non_system_messages):
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                latest_user_prompt_part = part
                latest_user_prompt_index = i

    # Apply limit if specified
    if limit is not None and limit > 0:
        tool_call_ids = {}
        tool_return_ids = set()

        for i, msg in enumerate(non_system_messages):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_ids[part.tool_call_id] = i
                elif isinstance(part, ToolReturnPart):
                    tool_return_ids.add(part.tool_call_id)

        if len(non_system_messages) > limit:
            included_indices = set(
                range(len(non_system_messages) - limit, len(non_system_messages))
            )

            # Include paired tool messages
            for i, msg in enumerate(non_system_messages):
                if i in included_indices:
                    for part in msg.parts:
                        if (
                            isinstance(part, ToolReturnPart)
                            and part.tool_call_id in tool_call_ids
                        ):
                            included_indices.add(tool_call_ids[part.tool_call_id])

            # Handle excluded user prompt
            if (
                latest_user_prompt_index >= 0
                and latest_user_prompt_index not in included_indices
                and latest_user_prompt_part is not None
                and system_message is not None
            ):
                user_prompt_index = next(
                    (
                        i
                        for i, part in enumerate(system_message.parts)
                        if isinstance(part, UserPromptPart)
                    ),
                    None,
                )

                if user_prompt_index is not None:
                    system_message.parts[user_prompt_index] = latest_user_prompt_part
                else:
                    system_message.parts.append(latest_user_prompt_part)

            non_system_messages = [
                msg
                for i, msg in enumerate(non_system_messages)
                if i in included_indices
            ]

    result_messages = []
    if system_message:
        result_messages.append(system_message)
    result_messages.extend(non_system_messages)

    return result_messages


def load_agent_prompt(agent_name: str) -> str:
    """Load agent prompt from file, replacing time_now variable."""
    time_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    prompt_path = os.path.join(
        os.path.dirname(__file__), "agents", f"{agent_name}.md"
    )
    with open(prompt_path, "r") as f:
        agent_prompt = f.read()
    return agent_prompt.replace("{time_now}", time_now)


async def main():
    """CLI testing in a conversation with the agent."""
    args = parse_args()

    # Set up OpenRouter model
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return

    model = OpenAIModel(
        args.model,
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        ),
    )

    # MCP environment variables
    env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "FREEBASE_PROFILE_MODEL": os.getenv("FREEBASE_PROFILE_MODEL", "openai:gpt-4o-mini"),
        "LOGGER_PATH": os.getenv("LOGGER_PATH", ""),
        "LOGGER_NAME": "freebase_mcp",
    }

    # Set up MCP Server
    mcp_servers = [
        MCPServerStdio("python", ["./mcp_server.py"], env=env),
    ]

    # Load agent prompt
    agent_name = "freebase"
    agent_prompt = load_agent_prompt(agent_name)

    print(f"\n🔪 Freebase Agent - Git History Surgeon")
    print(f"Using model: {args.model}")
    print("Type 'quit' or 'exit' to end the session.\n")

    agent = Agent(model, mcp_servers=mcp_servers, system_prompt=agent_prompt)

    async with agent.run_mcp_servers():
        result: AgentRunResult = None

        while True:
            if result:
                print(f"\n{result.output}")

            try:
                user_input = input("\n> ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if not user_input.strip():
                continue

            err = None
            for attempt in range(2):
                try:
                    result = await agent.run(
                        user_input,
                        message_history=filtered_message_history(
                            result,
                            limit=24,
                            include_tool_messages=True,
                        ),
                    )
                    break
                except Exception as e:
                    err = e
                    traceback.print_exc()
                    await asyncio.sleep(2)

            if result is None:
                print(f"\nError: {err}. Try again...\n")
                continue
            elif len(result.output) == 0:
                continue


if __name__ == "__main__":
    asyncio.run(main())
