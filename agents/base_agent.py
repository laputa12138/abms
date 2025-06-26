import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.llm_service import LLMService
from core.embedding_service import EmbeddingService
from core.reranker_service import RerankerService
from core.vector_store import VectorStore
# Import settings to allow agents to access global configs if necessary,
# though direct dependency on specific model names should be via service initialization.
from config import settings

# Configure logging for the agents module
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the RAG system.
    It provides a common structure and can hold shared resources
    like service instances.
    """

    def __init__(self,
                 agent_name: str,
                 llm_service: Optional[LLMService] = None,
                 embedding_service: Optional[EmbeddingService] = None,
                 reranker_service: Optional[RerankerService] = None,
                 vector_store: Optional[VectorStore] = None,
                 config: Optional[dict] = None):
        """
        Initializes the BaseAgent.

        Args:
            agent_name (str): The name of the agent, for logging and identification.
            llm_service (Optional[LLMService]): An instance of LLMService.
            embedding_service (Optional[EmbeddingService]): An instance of EmbeddingService.
            reranker_service (Optional[RerankerService]): An instance of RerankerService.
            vector_store (Optional[VectorStore]): An instance of VectorStore.
            config (Optional[dict]): Agent-specific configuration dictionary.
        """
        self.agent_name = agent_name
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.vector_store = vector_store
        self.config = config or {}

        # Global settings can be accessed via the imported `settings` module if needed,
        # e.g., settings.DEFAULT_LLM_MODEL_NAME, but typically services should handle these.
        # self.global_settings = settings

        logger.info(f"Agent '{self.agent_name}' initialized.")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        The main execution method for the agent.
        Subclasses must implement this method to define their specific logic.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the agent's execution.
        """
        logger.info(f"Agent '{self.agent_name}' starting run method.")
        pass

    def _log_input(self, *args: Any, **kwargs: Any):
        """Helper method to log input parameters."""
        # Truncate long inputs for cleaner logs
        truncated_args = [str(arg)[:200] + '...' if len(str(arg)) > 200 else str(arg) for arg in args]
        truncated_kwargs = {k: str(v)[:200] + '...' if len(str(v)) > 200 else str(v) for k, v in kwargs.items()}
        logger.debug(f"Agent '{self.agent_name}' received input - Args: {truncated_args}, Kwargs: {truncated_kwargs}")

    def _log_output(self, output: Any):
        """Helper method to log output."""
        truncated_output = str(output)[:500] + '...' if len(str(output)) > 500 else str(output)
        logger.debug(f"Agent '{self.agent_name}' produced output: {truncated_output}")

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the agent's configuration.

        Args:
            key (str): The configuration key.
            default (Any, optional): The default value if key is not found. Defaults to None.

        Returns:
            Any: The configuration value or default.
        """
        return self.config.get(key, default)

# Example of a concrete agent (for demonstration, will be moved/refined later)
class MyDummyAgent(BaseAgent):
    def __init__(self, llm_service: LLMService, config: Optional[dict]=None):
        super().__init__(agent_name="MyDummyAgent", llm_service=llm_service, config=config)

    def run(self, user_query: str) -> str:
        self._log_input(user_query=user_query)
        if not self.llm_service:
            logger.error(f"Agent '{self.agent_name}' requires LLMService but it was not provided.")
            return "Error: LLMService not available."

        prompt = f"User asked: {user_query}. Respond briefly."
        try:
            response = self.llm_service.chat(prompt, system_prompt="You are a dummy agent.")
            self._log_output(response)
            return response
        except Exception as e:
            logger.error(f"Agent '{self.agent_name}' encountered an error: {e}")
            return f"Error during processing: {e}"

if __name__ == '__main__':
    # This is an example of how to use the BaseAgent and a dummy implementation.
    # Requires a running Xinference server for LLMService to function fully.
    print("BaseAgent and MyDummyAgent Example")

    # Configure basic logging for the example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        # Attempt to initialize LLMService (will try to connect to Xinference)
        # from core.llm_service import LLMService, LLMServiceError
        # try:
        #     llm_service_instance = LLMService() # Uses defaults from settings
        #     print("Successfully initialized LLMService for the example.")
        # except LLMServiceError as e:
        #     print(f"Could not initialize LLMService (Xinference might not be running or model not available): {e}")
        #     print("Proceeding with None for LLMService in dummy agent (it will handle this).")
        #     llm_service_instance = None

        # Forcing None for llm_service_instance for this example run to avoid external dependency for automated test
        print("Using None for LLMService in dummy agent for this example run.")
        llm_service_instance = None


        # Create a dummy agent instance
        # If llm_service_instance is None, the dummy agent's run method should handle it.
        dummy_agent_config = {"temperature_override": 0.8}
        dummy_agent = MyDummyAgent(llm_service=llm_service_instance, config=dummy_agent_config)

        print(f"\nRunning {dummy_agent.agent_name}...")
        # query = "What is the purpose of a RAG system?"
        # result = dummy_agent.run(user_query=query)
        # print(f"Response from {dummy_agent.agent_name}: {result}")

        # query2 = "Tell me a joke."
        # result2 = dummy_agent.run(user_query=query2)
        # print(f"Response from {dummy_agent.agent_name}: {result2}")

        # Accessing config
        # temp_override = dummy_agent._get_config_value("temperature_override")
        # print(f"Dummy agent's temperature_override config: {temp_override}")
        # non_existent_config = dummy_agent._get_config_value("non_existent_key", "default_val")
        # print(f"Dummy agent's non_existent_key config: {non_existent_config}")

        print("\nNote: Actual LLM calls in MyDummyAgent are commented out if LLMService was None.")
        print("BaseAgent example finished.")

    except Exception as e:
        print(f"An unexpected error occurred in BaseAgent example: {e}")
        import traceback
        traceback.print_exc()
