# web_search_agent.py

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

class WebSearchAgent:
    def __init__(self, model_id: str = "gpt-3.5-turbo"):
        # Initialize the Agno Agent with DuckDuckGo search capability
        self.agent = Agent(
            model=OpenAIChat(id=model_id),
            description="You are a web-search assistant. Use your search tool to fetch relevant web results.",
            tools=[DuckDuckGoTools()],
            show_tool_calls=True,
            markdown=True
        )

    def search(self, query: str) -> str:
        """Run the agent with a search prompt and return its synthesized answer."""
        prompt = f'Search the web for: "{query}"'
        return self.agent.run(prompt)


if __name__ == "__main__":
    # Quick smoke-test when running directly:
    ws = WebSearchAgent()
    result = ws.search("latest breakthroughs in renewable energy")
    print(result)

