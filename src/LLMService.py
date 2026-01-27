import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class LLMService:
    """
    A universal wrapper for Gemini that allows per-instance 
    configuration of the model, temperature, and API key.
    """
    def __init__(self, model="gemini-2.0-flash", temperature=0, api_key=None):
        # Logic for API Key: Priority (Argument > .env > OS Environment)
        if not api_key:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("API Key not provided and 'GOOGLE_API_KEY' not found in .env")

        # Initialize the model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
        # Store config metadata for easy debugging
        self.model_info = {
            "model": model, 
            "temperature": temperature
        }

    def ask(self, system_msg, human_msg, variables, schema=None):
        """
        Universal method to handle both structured Pydantic outputs 
        and standard string responses.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", human_msg)
        ])
        
        # Determine if we need structured output (Pydantic) or raw text
        engine = self.llm.with_structured_output(schema) if schema else self.llm
        
        chain = prompt | engine
        return chain.invoke(variables)