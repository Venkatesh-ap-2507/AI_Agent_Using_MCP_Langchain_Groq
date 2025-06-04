from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClient(MultiServerMCPClient):
    @classmethod
    def from_config_file(cls, config_file: str):
        """Initialize MCPClient from a JSON config file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded MCP config from {config_file}")
            return cls(config.get("mcpServers", {}))
        except FileNotFoundError:
            logger.error(f"Config file {config_file} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise

    async def close_all_sessions(self):
        """Close all active MCP server sessions."""
        try:
            for session in self.sessions.values():
                await session.aclose()
            logger.info("All MCP sessions closed successfully")
        except Exception as e:
            logger.error(f"Error closing MCP sessions: {e}")


class MCPAgent:
    def __init__(self, llm, client: MCPClient, max_steps: int = 10, memory_enabled: bool = False):
        """Initialize an MCPAgent with an LLM, MCP client, and optional memory."""
        raise RuntimeError("Use async MCPAgent.create() instead.")

    @classmethod
    async def create(cls, llm, client: MCPClient, max_steps: int = 10, memory_enabled: bool = False):
        """Async constructor for MCPAgent."""
        self = cls.__new__(cls)
        self.llm = llm
        self.client = client
        self.max_steps = max_steps
        self.memory_enabled = memory_enabled
        self.checkpointer = MemorySaver() if memory_enabled else None

        try:
            # Await get_tools since it's async
            tools = await client.get_tools()
            logger.info(f"Loaded {len(tools)} tools from MCP servers")

            # Create ReAct agent with checkpointer for memory
            self.agent = create_react_agent(
                model=llm,
                tools=tools,
                checkpointer=self.checkpointer
            )

            logger.info("MCPAgent created successfully")
            return self

        except Exception as e:
            logger.error(f"Error creating MCPAgent: {e}")
            raise

    async def run(self, user_input: str, thread_id: str = "default") -> str:
        """Run the agent with user input and return the response."""
        try:
            logger.info(f"Processing user input for thread {thread_id}")
            messages = [HumanMessage(content=user_input)]
            config = {"configurable": {"thread_id": thread_id}
                      } if self.memory_enabled else {}

            response = await self.agent.ainvoke({"messages": messages}, config)
            output = response["messages"][-1].content

            logger.info("Successfully processed user input")
            return output

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def clear_conversation_history(self, thread_id: str = "default"):
        """Clear the conversation memory for a given thread_id."""
        if self.memory_enabled and self.checkpointer:
            try:
                # Clear checkpoint by resetting the thread
                config = {"configurable": {"thread_id": thread_id}}
                self.checkpointer.put(config, {}, {})
                logger.info(
                    f"Cleared conversation history for thread {thread_id}")
            except Exception as e:
                logger.error(f"Error clearing conversation history: {e}")
        else:
            logger.warning("Memory not enabled or checkpointer not available")

    async def get_available_tools(self):
        """Get list of available tools."""
        try:
            tools = await self.client.get_tools()
            return [{"name": tool.name, "description": tool.description} for tool in tools]
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return []
