import openai

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools

class WebSearchAgent(Agent):
    def __init__(self, shared_memory):
        super().__init__(
            model=OpenAIChat(id="gpt-3.5-turbo"),  # Use OpenAI GPT-3.5 Turbo model
            tools=[DuckDuckGoTools()],
        )
        self.shared_memory = shared_memory


    def execute(self):
        user_input = self.shared_memory.get("user_input")

        # Use Mistral to determine search query
        search_query = self.determine_search_query(user_input)
        self.shared_memory.set("search_query", search_query)
        
        # Perform web search using DuckDuckGoTools
        prompt = f"Use web search to find relevant travel information for: '{search_query}'"
        search_results = self.run(prompt)

        # Store search results in shared memory
        self.shared_memory.set("search_results", search_results)


    def determine_search_query(self, user_input):
        # Construct prompt for Mistral
        prompt = f"Given the following user input: '{user_input}', what is a suitable search query to find relevant travel information?"

        # Use Mistral to generate search query
        search_query = self.run(prompt)

        return search_query


class RecommenderAgent(Agent):
    def __init__(self, shared_memory):
        super().__init__(
            model=OpenAIChat(id="gpt-3.5-turbo"),  # Use OpenAI GPT-3.5 Turbo model
            tools=[],  # No tools needed for this agent
        )
        self.shared_memory = shared_memory

    def execute(self):
        user_input = self.shared_memory.get("user_input")
        search_query = self.shared_memory.get("search_query")
        search_results = self.shared_memory.get("search_results")

        if not all([user_input, search_query, search_results]):
            raise ValueError("Missing required data in shared_memory for recommendation.")

        # Use Mistral to evaluate the search path
        evaluation = self.evaluate_search_path(user_input, search_query, search_results)

        # Store evaluation in shared memory
        self.shared_memory.set("evaluation", evaluation)

    def evaluate_search_path(self, user_input, search_query, search_results):
        # Limit search_results to first 500 characters
        short_results = str(search_results)[:500]

        # Construct prompt for Mistral
        prompt = f"""
        Evaluate if search query '{search_query}' matched user input '{user_input}' based on these results: '{short_results}'.
        Score from 1 to 10 and give a brief reason.
        """

        # Use Mistral to generate evaluation
        evaluation = self.run(prompt)

        return evaluation



from agno.models.openai import OpenAIChat

class SummarizerAgent(Agent):
    def __init__(self, shared_memory, model_id: str = "gpt-3.5-turbo"):
        super().__init__(
            model=OpenAIChat(id=model_id),
            description="You are a travel recommendation summarizer. You take outputs from the RecommenderAgent and generate concise summaries for users.",
            markdown=True
        )
        self.shared_memory = shared_memory

    def execute(self, style: str = "bullet"):
        recommendations_text = self.shared_memory.get("evaluation")
        if not recommendations_text:
            raise ValueError("No recommendations found in shared_memory to summarize.")
        prompt = self._build_prompt(recommendations_text, style)
        summary = self.run(prompt)
        self.shared_memory.set("summary", summary)

    def _build_prompt(self, text, style):
        """Build prompt based on the requested summary style."""
        if style == "bullet":
            return f"Summarize the following recommended NYC travel places into concise bullet points:\n\n{text}"
        elif style == "headline":
            return f"Create a short and catchy headline summarizing these recommended NYC travel places:\n\n{text}"
        else:
            return f"Summarize the following NYC travel recommendations into a short paragraph:\n\n{text}"

class TranslationAgent(Agent):
    def __init__(self, shared_memory, model_id: str = "gpt-3.5-turbo"):
        super().__init__(
            model=OpenAIChat(id=model_id),
            description="You are a translation agent. Translate the provided text into the target language accurately while keeping the original meaning.",
            markdown=True
        )
        self.shared_memory = shared_memory

    def execute(self):
        summary = self.shared_memory.get("summary")
        target_language = self.shared_memory.get("target_language")  # e.g., "French", "Spanish"

        if not summary:
            raise ValueError("No summary found in shared_memory to translate.")
        if not target_language:
            raise ValueError("No target language specified in shared_memory.")

        prompt = f"Translate the following text into {target_language}:\n\n{summary}"
        translated_text = self.run(prompt)
        self.shared_memory.set("translated_summary", translated_text)
