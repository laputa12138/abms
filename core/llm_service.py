import logging
from xinference.client import Client
from config.settings import (
    XINFERENCE_API_URL,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_P,
    DEFAULT_LLM_ENABLE_THINKING,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_MIN_P
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMServiceError(Exception):
    """Custom exception for LLMService errors."""
    pass

class LLMService:
    """
    A service class to interact with a Large Language Model (LLM)
    deployed via Xinference.
    """
    def __init__(self, api_url: str = None, model_name: str = None):
        """
        Initializes the LLMService.

        Args:
            api_url (str, optional): The URL of the Xinference API.
                                     Defaults to XINFERENCE_API_URL from settings.
            model_name (str, optional): The name of the LLM model to use.
                                        Defaults to DEFAULT_LLM_MODEL_NAME from settings.
        """
        self.api_url = api_url or XINFERENCE_API_URL
        self.model_name = model_name or DEFAULT_LLM_MODEL_NAME

        try:
            self.client = Client(self.api_url)
            self.model = self.client.get_model(self.model_name)
            logger.info(f"Successfully connected to Xinference API at {self.api_url} and loaded model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Xinference client or load model {self.model_name} from {self.api_url}: {e}")
            raise LLMServiceError(f"Xinference client/model initialization failed: {e}")

    def chat(self,
             query: str,
             system_prompt: str = "You are a helpful assistant.",
             max_tokens: int = None,
             temperature: float = None,
             top_p: float = None,
             enable_thinking: bool = None,
             top_k: int = None,
             min_p: float = None) -> str:
        """
        Sends a chat message to the LLM and returns the assistant's response.

        Args:
            query (str): The user's query or message.
            system_prompt (str, optional): The system prompt to guide the LLM.
                                           Defaults to "You are a helpful assistant.".
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to settings.
            temperature (float, optional): Sampling temperature. Defaults to settings.
            top_p (float, optional): Nucleus sampling parameter. Defaults to settings.
            enable_thinking (bool, optional): Whether to enable thinking process. Defaults to settings.
            top_k (int, optional): Top-k sampling parameter. Defaults to settings.
            min_p (float, optional): Min-p sampling parameter. Defaults to settings.


        Returns:
            str: The LLM's response content.

        Raises:
            LLMServiceError: If the chat request fails or the response is malformed.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        generate_config = {
            "max_tokens": max_tokens if max_tokens is not None else DEFAULT_LLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else DEFAULT_LLM_TEMPERATURE,
            "top_p": top_p if top_p is not None else DEFAULT_LLM_TOP_P, # Corrected from TopP
            "enable_thinking": enable_thinking if enable_thinking is not None else DEFAULT_LLM_ENABLE_THINKING,
            "top_k": top_k if top_k is not None else DEFAULT_LLM_TOP_K, # Corrected from TopK
            "min_p": min_p if min_p is not None else DEFAULT_LLM_MIN_P,
        }

        # Detailed logging for LLM interaction
        logger.debug(f"LLM Request to model: {self.model_name}")
        logger.debug(f"System Prompt: {system_prompt}")

        # Log query (potentially truncated for brevity in standard logs, full in debug file if needed)
        log_query_display = query if len(query) < 500 else query[:500] + "... (truncated)"
        logger.debug(f"User Query (preview): {log_query_display}")
        if len(query) > 500: # For very long queries, consider writing to a temp file
            # For now, we'll rely on debug level being sufficient. If not, implement file logging for prompts.
            logger.debug(f"Full User Query (first 2000 chars for very long inputs): \n{query[:2000]}")

        logger.debug(f"Generation Config: {generate_config}")


        try:
            response = self.model.chat(
                messages=messages,
                generate_config=generate_config
            )

            # Detailed logging for LLM response
            logger.debug(f"Raw LLM Response object: {response}")

            if response and "choices" in response and len(response["choices"]) > 0:
                assistant_message = response["choices"][0].get("message", {}).get("content")
                if assistant_message:
                    log_assistant_message_display = assistant_message if len(assistant_message) < 500 else assistant_message[:500] + "... (truncated)"
                    logger.debug(f"LLM Assistant Message (preview): {log_assistant_message_display}")
                    if len(assistant_message) > 2000: # Log more for very long outputs if necessary
                        logger.debug(f"Full LLM Assistant Message (first 2000 chars for very long outputs): \n{assistant_message[:2000]}")
                    return assistant_message
                else:
                    logger.error(f"Malformed response from LLM: 'content' field missing. Full Response: {response}")
                    raise LLMServiceError("Malformed response from LLM: 'content' field missing.")
            else:
                logger.error(f"No valid choices in LLM response. Full Response: {response}")
                raise LLMServiceError("No valid choices in LLM response.")
        except Exception as e:
            logger.error(f"Error during LLM chat request to {self.model_name}: {e}", exc_info=True)
            raise LLMServiceError(f"LLM chat request failed: {e}")

if __name__ == '__main__':
    # This is an example of how to use the LLMService.
    # It requires a running Xinference server with the 'qwen3' model.
    # As per instructions, this will not be run during the automated process.
    print("LLMService Example (requires running Xinference server)")
    print("This part will not be executed by the agent but is for local testing.")

    # Mocking Xinference client for demonstration if direct call is not possible in this environment
    # In a real scenario, ensure Xinference is accessible.
    try:
        llm_service = LLMService() # Uses defaults from settings.py

        # Example 1: Simple query
        # user_query = "介绍美国的ABMS系统"
        # print(f"\nQuerying LLM about: {user_query}")
        # response_content = llm_service.chat(user_query)
        # print(f"LLM Response:\n{response_content}")

        # Example 2: Custom system prompt and parameters
        # user_query_translate = "Translate 'Hello, world!' to French."
        # system_prompt_translate = "You are a helpful translation assistant."
        # print(f"\nQuerying LLM for translation: {user_query_translate}")
        # response_translate = llm_service.chat(
        #     user_query_translate,
        #     system_prompt=system_prompt_translate,
        #     temperature=0.7
        # )
        # print(f"LLM Translation Response:\n{response_translate}")

        print("\nLLMService example finished. If no output, ensure Xinference server is running and configured.")
        print("Note: Actual API calls are commented out to prevent errors if server is not available.")

    except LLMServiceError as e:
        print(f"Error initializing or using LLMService: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
