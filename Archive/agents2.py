import openai
import logging
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat

# Configure logging
logging.basicConfig(level=logging.INFO)

class WebSearchAgent(Agent):
    def __init__(self, shared_memory):
        super().__init__(
            model=OpenAIChat(id="gpt-3.5-turbo"),
            tools=[DuckDuckGoTools()],
        )
        self.shared_memory = shared_memory

    def execute(self):
        logging.info("WebSearchAgent execution started.")
        user_input = self.shared_memory.get("user_input")
        if not user_input:
            raise ValueError("Missing 'user_input' in shared_memory for WebSearchAgent.")

        # Determine search query
        search_query = self.determine_search_query(user_input)
        self.shared_memory.set("search_query", search_query)

        # Perform web search
        prompt = f"Use web search to find relevant travel information for: '{search_query}'"
        search_results = self.run(prompt)

        # Store search results in shared memory
        self.shared_memory.set("search_results", search_results)
        logging.info("WebSearchAgent execution completed successfully.")

    def determine_search_query(self, user_input: str) -> str:
        """Determine a suitable search query from user input."""
        prompt = f"Given the following user input: '{user_input}', what is a suitable search query to find relevant travel information?"
        search_query = self.run(prompt)
        return search_query


class RecommenderAgent(Agent):
    def __init__(self, shared_memory):
        super().__init__(
            model=OpenAIChat(id="gpt-3.5-turbo"),
            tools=[],
            description="You are an evaluator of search results, pick the best search results based on user input."
        )
        self.shared_memory = shared_memory

    def execute(self):
        logging.info("RecommenderAgent execution started.")
        user_input = self.shared_memory.get("user_input")
        search_query = self.shared_memory.get("search_query")
        search_results = self.shared_memory.get("search_results")

        if not user_input:
            raise ValueError("Missing 'user_input' in shared_memory for RecommenderAgent.")
        if not search_query:
            raise ValueError("Missing 'search_query' in shared_memory for RecommenderAgent.")
        if not search_results:
            raise ValueError("Missing 'search_results' in shared_memory for RecommenderAgent.")

        # Evaluate search results
        evaluation = self.evaluate_search_path(user_input, search_query, search_results)
        self.shared_memory.set("evaluation", evaluation)
        logging.info("RecommenderAgent execution completed successfully.")

    def evaluate_search_path(self, user_input: str, search_query: str, search_results: str) -> str:
        """Evaluate search results and provide recommendations."""
        short_results = str(search_results)[:500]
        prompt = f"""
        Based on the user input: '{user_input}' and the search query: '{search_query}', evaluate the following search results: '{short_results}'.
        Pick the best results, score from 1 to 10, and provide a brief reason for your evaluation.
        """
        evaluation = self.run(prompt)
        return evaluation


def _build_prompt(self, recommendations_text: str, style: str) -> str:
        """Build a prompt for summarizing recommendations based on the desired style."""
        if style == "bullet":
            return f"""
            Summarize the following recommendations using bullet points:
            {recommendations_text}
            """
        elif style == "paragraph":
            return f"""
            Summarize the following recommendations in a concise paragraph:
            {recommendations_text}
            """
        else:
            raise ValueError(f"Unsupported style: {style}. Supported styles are 'bullet' and 'paragraph'.")

