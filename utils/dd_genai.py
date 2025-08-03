import google.generativeai as genai
import os

def get_gemini_response(user_prompt: str, model_name: str = 'gemini-pro') -> str:
    """
    Interacts with the Gemini API to generate a text response based on a user prompt.

    Args:
        user_prompt (str): The text prompt to send to the Gemini model.
        model_name (str, optional): The name of the Gemini model to use.
                                     Defaults to 'gemini-pro'. Other options include
                                     'gemini-pro-vision' for multimodal inputs.

    Returns:
        str: The generated text response from the Gemini model, or an error message
             if the API call fails.
    """

    try:
        # Configure the Gemini API key.
        # It's highly recommended to set your API key as an environment variable
        # named GEMINI_API_KEY. The SDK will automatically pick it up.
        # If you must hardcode it for quick testing (not recommended for production):
        # genai.configure(api_key="YOUR_API_KEY_HERE")
        # Otherwise, ensure the GEMINI_API_KEY environment variable is set.
        if os.getenv("GEMINI_API_KEY") is None:
            return "Error: GEMINI_API_KEY environment variable not set. Please set your Gemini API key."

        # Initialize the Generative Model with the specified model name.
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

        # Send the user prompt to the Gemini model.
        response = model.generate_content(user_prompt)

        # Return the generated text.
        return response.text

    except Exception as e:
        # Catch any exceptions that occur during the API call and return an error message.
        return f"An error occurred: {e}"