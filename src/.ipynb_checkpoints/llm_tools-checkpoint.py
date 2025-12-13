import os
from dotenv import load_dotenv
from typing import List, Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- Schema 1: For files where we ALREADY know the section ---
class TitleOnly(BaseModel):
    title: str = Field(
        ..., 
        description="A concise, identity-focused title for this file's content (e.g., 'AI Researcher - Univ of Isfahan'). Do not include the section header."
    )

# --- Schema 2: For 'Uncategorized' files where we need BOTH ---
class TitleAndCategory(BaseModel):
    title: str = Field(
        ...,
        description="A concise, identity-focused title for this file's content."
    )
    category: Literal["CONTACT_INFORMATION","PROFILE", "WORK_EXPERIENCE", "EDUCATION", "PROJECTS", "SKILLS", "OTHER"] = Field(
        ...,
        description="The standardized category of this section determined from the content."
    )

def enrich_file_metadata(file_list: List[dict], gemini_api_key: str = None) -> List[dict]:
    """
    Analyzes the content of individual LaTeX files using the Gemini LLM 
    to extract a descriptive title and, if necessary, determine the file's 
    standardized category.

    This function uses conditional logic based on the 'section' key:
    1. If 'section' is 'Uncategorized', it calls Chain B (TitleAndCategory).
    2. Otherwise, it calls Chain A (TitleOnly).

    Args:
        file_list: A list of dictionaries, each containing at least 
                   'full_path' (str) and 'section' (str).
        gemini_api_key: Optional. The API key for the Gemini model. If None,
                        it attempts to load it from the GOOGLE_API_KEY environment variable.

    Returns:
        A new list of dictionaries, where each entry is updated with a 'title' 
        key and potentially a 'category' key.

    Raises:
        ValueError: If the Gemini API key cannot be found.
    """
    
    # 1. API Key Setup
    if not gemini_api_key: 
        # Loads variables from .env file
        load_dotenv()
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI API KEY not found")
            
    # 2. Initialize Model
    # Note: 'gemini-2.5-flash' is the standard lightweight model name. 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=gemini_api_key
    )
    
    # 3. Create TWO distinct chains for different output schemas
    
    # Chain A: For known sections (extracts Title only)
    structured_llm_title = llm.with_structured_output(TitleOnly)
    prompt_title = ChatPromptTemplate.from_messages([
        ("system", "You are a resume parser. Extract a concise title from the LaTeX content."),
        ("human", "Context: Section is known as '{section}'.\n\nRaw Content:\n{content}")
    ])
    chain_title = prompt_title | structured_llm_title

    # Chain B: For Uncategorized sections (extracts Title + Category)
    structured_llm_full = llm.with_structured_output(TitleAndCategory)
    prompt_full = ChatPromptTemplate.from_messages([
        ("system", "You are a resume parser. The section is unknown. Analyze content to determine the Category and Title."),
        ("human", "Context: Section is Uncategorized.\n\nRaw Content:\n{content}")
    ])
    chain_full = prompt_full | structured_llm_full

    enriched_list = []
    print(f"Processing {len(file_list)} files...")

    for entry in file_list:
        full_path = entry.get("full_path")
        current_section = entry.get("section", "Uncategorized")
        
        # --- Read Content ---
        if full_path and os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    # Read content for LLM; limit size to avoid excessive tokens
                    raw_content = f.read()
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
                raw_content = ""
        else:
            raw_content = ""
            
        if not raw_content.strip():
            entry["title"] = "Empty File"
            enriched_list.append(entry)
            continue

        # --- Conditional Logic (The core distinction) ---
        try:
            # Copy the entry to prevent modifying the original list item during iteration
            new_entry = entry.copy()

            if current_section == "Uncategorized":
                # PATH 1: Use Full Chain (Get Title + Category)
                response = chain_full.invoke({
                    "content": raw_content[:3000]
                })
                new_entry["title"] = response.title
                new_entry["category"] = response.category
                # Update the section key to the newly identified category for consistency
                new_entry["section"] = response.category 
                print(f"Filled Uncategorized -> [{response.category}] {response.title}")

            else:
                # PATH 2: Use Title Chain (Get Title Only)
                response = chain_title.invoke({
                    "section": current_section,
                    "content": raw_content[:3000]
                })
                new_entry["title"] = response.title
                print(f"Processed Known -> {response.title}")
            
            enriched_list.append(new_entry)
            
        except Exception as e:
            print(f"LLM Error on {full_path}: {e}")
            entry["error"] = str(e)
            enriched_list.append(entry)

    return enriched_list