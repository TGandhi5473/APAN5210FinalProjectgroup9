import openai
import logging
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat
import spacy
# Configure logging
logging.basicConfig(level=logging.INFO)

class WebSearchAgent(Agent):
    """
    An agent that performs web searches based on user input.

    Attributes:
        shared_memory: A shared memory object for inter-agent data exchange.
    """

    def __init__(self, shared_memory):
        """
        Initializes the WebSearchAgent with a model and tools.

        Args:
            shared_memory: The shared memory object for data exchange between agents.
        """
        super().__init__(
            model=OpenAIChat(id="gpt-3.5-turbo"),
            tools=[DuckDuckGoTools()],
        )
        self.shared_memory = shared_memory
        self.nlp = spacy.load("en_core_web_sm")  # Load SpaCy's language model

    def execute(self):
        """
        Executes the agent by determining a search query from user input,
        validating it, performing a web search, and storing the results in shared memory.
        """
        logging.info("WebSearchAgent execution started.")
        user_input = self.shared_memory.get("user_input")
        if not user_input:
            raise ValueError("Missing 'user_input' in shared_memory for WebSearchAgent.")

        # Determine and validate search query
        search_query = self.determine_search_query(user_input)
        if not self.validate_search_query(search_query):
            raise ValueError(f"Invalid search query: '{search_query}'. Please provide a more specific input.")
        self.shared_memory.set("search_query", search_query)

        # Perform web search
        prompt = f"Use web search to find relevant travel information for: '{search_query}'"
        search_results = self.run(prompt)

        # Store search results in shared memory
        self.shared_memory.set("search_results", search_results)
        logging.info("WebSearchAgent execution completed successfully.")

    def determine_search_query(self, user_input: str) -> str:
        """
        Determines a suitable search query from the user input.

        Args:
            user_input: The input provided by the user.

        Returns:
            A search query string derived from the user input.
        """
        prompt = f"Given the following user input: '{user_input}', what is a suitable search query to find relevant travel information?"
        search_query = self.run(prompt)
        return search_query

    def validate_search_query(self, query: str) -> bool:
        """
        Validates the search query using Named Entity Recognition (NER) to ensure it contains relevant entities.

        Args:
            query: The search query to be validated.

        Returns:
            A boolean indicating whether the query is valid.
        """
        logging.info(f"Validating search query: '{query}'")
        doc = self.nlp(query)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:  # Check for location-related entities
                logging.info(f"Valid search query with entity: {ent.text} ({ent.label_})")
                return True
        logging.warning("No relevant entities found in the search query.")
        return False

class RecommenderAgent(Agent):
    """
    An agent that evaluates web search results and provides recommendations.

    Attributes:
        shared_memory: A shared memory object for inter-agent data exchange.
        cache: A dictionary to store cached evaluations for repeated queries.
    """

    def __init__(self, shared_memory):
        """
        Initializes the RecommenderAgent with a model and description.

        Args:
            shared_memory: The shared memory object for data exchange between agents.
        """
        super().__init__(
            model=OpenAIChat(id="gpt-3.5-turbo"),
            tools=[],
            description="You are an evaluator of search results, pick the best search results based on user input."
        )
        self.shared_memory = shared_memory
        self.cache = {}  # Initialize a local cache

    def execute(self):
        """
        Executes the agent by evaluating search results based on user input, user preferences,
        and storing the evaluation in shared memory. Includes caching and error handling.
        """
        logging.info("RecommenderAgent execution started.")
        try:
            # Retrieve required data from shared memory
            user_input = self.shared_memory.get("user_input")
            search_query = self.shared_memory.get("search_query")
            search_results = self.shared_memory.get("search_results")
            user_preferences = self.shared_memory.get("user_preferences", {})

            # Validate inputs
            if not user_input:
                raise ValueError("Missing 'user_input' in shared_memory for RecommenderAgent.")
            if not search_query:
                raise ValueError("Missing 'search_query' in shared_memory for RecommenderAgent.")
            if not search_results:
                raise ValueError("Missing 'search_results' in shared_memory for RecommenderAgent.")

            # Check if the result is already cached
            cache_key = hash((user_input, search_query))
            if cache_key in self.cache:
                logging.info("Using cached evaluation for the given input.")
                self.shared_memory.set("evaluation", self.cache[cache_key])
                return

            # Process top 5 results only
            top_results = search_results[:5] if isinstance(search_results, list) else search_results

            # Evaluate search results with user preferences
            evaluation = self.evaluate_search_path(user_input, search_query, top_results, user_preferences)

            # Cache the evaluation result
            self.cache[cache_key] = evaluation

            # Store evaluation in shared memory
            self.shared_memory.set("evaluation", evaluation)
            logging.info("RecommenderAgent execution completed successfully.")
        except Exception as e:
            logging.error(f"Error during RecommenderAgent execution: {e}")
            raise

    def evaluate_search_path(self, user_input: str, search_query: str, search_results, user_preferences: dict) -> str:
        """
        Evaluates search results and provides recommendations with explainability.

        Args:
            user_input: The input provided by the user.
            search_query: The search query derived from the user input.
            search_results: The top search results to be evaluated.
            user_preferences: A dictionary of user preferences for evaluation.

        Returns:
            A string containing the evaluation and recommendations with explanations.
        """
        # Extract user preferences
        preference_criteria = user_preferences.get("evaluation_criteria", "relevance")

        # Build the evaluation prompt
        prompt = f"""
        Based on the user input: '{user_input}' and the search query: '{search_query}', evaluate the following top results:
        {search_results}.
        Prioritize results based on '{preference_criteria}'. Provide a score from 1 to 10 for each result and explain why you assigned each score.
        """
        logging.debug(f"Generated evaluation prompt: {prompt}")

        # Run the evaluation
        evaluation = self.run(prompt)
        return evaluation
class SummarizerAgent(Agent):
    """
    An agent that summarizes travel recommendations based on evaluations.

    Attributes:
        shared_memory: A shared memory object for inter-agent data exchange.
    """

    def __init__(self, shared_memory, model_id: str = "gpt-3.5-turbo"):
        """
        Initializes the SummarizerAgent with a model and description.

        Args:
            shared_memory: The shared memory object for data exchange between agents.
            model_id: The ID of the model to be used for summarization.
        """
        logging.info("Initializing SummarizerAgent.")
        super().__init__(
            model=OpenAIChat(id=model_id),
            description="You are a travel recommendation summarizer. You take outputs from the RecommenderAgent and generate concise summaries for users.",
            markdown=True
        )
        self.shared_memory = shared_memory
        logging.info("SummarizerAgent initialized successfully.")

    def execute(self, style: str = "bullet"):
        """
        Executes the agent by summarizing recommendations into the specified style.

        Args:
            style: The style of the summary (e.g., bullet, headline, or paragraph).
        """
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
        """
        Builds a prompt for summarization based on the requested style.

        Args:
            text: The text to be summarized.
            style: The style of the summary (e.g., bullet, headline, or paragraph).

        Returns:
            A string prompt for the summarization.
        """
        logging.info(f"Building prompt for style: {style}")
        if style == "bullet":
            return f"Summarize the following recommended NYC travel places into concise bullet points:\n\n{text}"
        elif style == "headline":
            return f"Create a short and catchy headline summarizing these recommended NYC travel places:\n\n{text}"
        else:
            return f"Summarize the following NYC travel recommendations into a short paragraph:\n\n{text}"


class TranslationAgent(Agent):
    """
    An agent that translates text summaries into a specified target language.

    Attributes:
        shared_memory: A shared memory object for inter-agent data exchange.
    """

    def __init__(self, shared_memory, model_id: str = "gpt-3.5-turbo"):
        """
        Initializes the TranslationAgent with a model and description.

        Args:
            shared_memory: The shared memory object for data exchange between agents.
            model_id: The ID of the model to be used for translation.
        """
        logging.info("Initializing TranslationAgent.")
        super().__init__(
            model=OpenAIChat(id=model_id),
            description="You are a translation agent. Translate the provided text into the target language accurately while keeping the original meaning.",
            markdown=True
        )
        self.shared_memory = shared_memory
        logging.info("TranslationAgent initialized successfully.")

    def execute(self):
        """
        Executes the agent by translating the summary text into the target language.
        """
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
            raise
