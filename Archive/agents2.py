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


class SummarizerAgent(Agent):
    def __init__(self, shared_memory, model_id: str = "gpt-3.5-turbo"):
        logging.info("Initializing SummarizerAgent.")
        super().__init__(
            model=OpenAIChat(id=model_id),
            description="You are a travel recommendation summarizer. You take outputs from the RecommenderAgent and generate concise summaries for users.",
            markdown=True
        )
        self.shared_memory = shared_memory
        logging.info("SummarizerAgent initialized successfully.")

    def execute(self, style: str = "bullet"):
        logging.info("SummarizerAgent execution started.")
        try:
            recommendations_text = self.shared_memory.get("evaluation")
            if not recommendations_text:
                raise ValueError("No recommendations found in shared_memory to summarize.")
            logging.info("Recommendations retrieved from shared_memory.")

            prompt = self._build_prompt(recommendations_text, style)
            logging.debug(f"Generated prompt: {prompt}")

            summary = self.run(prompt)
            self.shared_memory.set("summary", summary)
            logging.info("Summary generated and saved to shared_memory successfully.")
        except Exception as e:
            logging.error(f"Error during SummarizerAgent execution: {e}")
            raise
        logging.info("SummarizerAgent execution completed successfully.")

    def _build_prompt(self, text, style):
        """Build prompt based on the requested summary style."""
        logging.info(f"Building prompt for style: {style}")
        if style == "bullet":
            return f"Summarize the following recommended NYC travel places into concise bullet points:\n\n{text}"
        elif style == "headline":
            return f"Create a short and catchy headline summarizing these recommended NYC travel places:\n\n{text}"
        else:
            return f"Summarize the following NYC travel recommendations into a short paragraph:\n\n{text}"

class TranslationAgent(Agent):
    def __init__(self, shared_memory, model_id: str = "gpt-3.5-turbo"):
        logging.info("Initializing TranslationAgent.")
        super().__init__(
            model=OpenAIChat(id=model_id),
            description="You are a translation agent. Translate the provided text into the target language accurately while keeping the original meaning.",
            markdown=True
        )
        self.shared_memory = shared_memory
        logging.info("TranslationAgent initialized successfully.")

    def execute(self):
        logging.info("TranslationAgent execution started.")
        try:
            summary = self.shared_memory.get("summary")
            target_language = self.shared_memory.get("target_language")  # e.g., "French", "Spanish"

            if not summary:
                raise ValueError("No summary found in shared_memory to translate.")
            if not target_language:
                raise ValueError("No target language specified in shared_memory.")

            logging.info(f"Retrieved summary and target language: {target_language}.")
            prompt = f"Translate the following text into {target_language}:\n\n{summary}"
            logging.debug(f"Generated prompt for translation: {prompt}")

            translated_text = self.run(prompt)
            self.shared_memory.set("translated_summary", translated_text)
            logging.info("Translation completed and saved to shared_memory.")
        except Exception as e:
            logging.error(f"Error during TranslationAgent execution: {e}")
