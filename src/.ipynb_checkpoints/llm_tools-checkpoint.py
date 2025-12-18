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
    
    # API Key Setup
    if not gemini_api_key: 
        # Loads variables from .env file
        load_dotenv()
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI API KEY not found")
            
    # Initialize Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=gemini_api_key
    )
    
    # Create TWO distinct chains for different output schemas
    
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

##################################################################################

# --- 1. Revert to Simple Schema (The Judge's Final Output) ---
class JDExtraction(BaseModel):
    """Final, filtered list of keywords approved by the Judge."""
    technical_stack: List[str] = Field(..., description="Approved list of core tech stack.")
    tools_and_platforms: List[str] = Field(..., description="Approved list of tools/platforms.")
    domain_knowledge: List[str] = Field(..., description="Approved list of domain concepts.")
    soft_skills: List[str] = Field(..., description="Approved list of critical behavioral traits.")
    years_experience_min: int = Field(..., description="Validated minimum years of experience.")

# --- 2. Intermediate Schema (For the Miner - Raw List) ---
class RawExtraction(BaseModel):
    """Raw, unfiltered extraction from the first pass."""
    raw_keywords: List[str] = Field(..., description="A comprehensive list of ALL potential keywords found in the text.")
    raw_years: int = Field(..., description="Years of experience found.")

def extract_jd_features(jd_file_path: str, gemini_api_key: str = None) -> JDExtraction:
    
    if not gemini_api_key: 
        load_dotenv()
        gemini_api_key = os.getenv("GOOGLE_API_KEY")

    if not os.path.exists(jd_file_path):
        raise FileNotFoundError(f"File not found: {jd_file_path}")
        
    try:
        with open(jd_file_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file: {e}")

    # Use a stronger model for the Judge if possible, but Flash is fine for both.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=gemini_api_key)

    # ==========================================
    # STEP 1: THE MINER (High Recall)
    # ==========================================
    # We ask for a simple raw list first to ensure we don't miss anything.
    miner_llm = llm.with_structured_output(RawExtraction)
    
    miner_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a Job Description Scanner. Your job is to extract EVERY potential technical keyword, tool, soft skill, and concept mentioned in the text.
         
         RULES:
         1. Be aggressive. If it looks like a skill, extract it.
         2. Do not filter yet. We will filter later.
         3. Extract exact phrases from the text.
         """
        ),
        ("human", "Job Description:\n{jd_content}")
    ])
    
    miner_chain = miner_prompt | miner_llm

    print("Step 1: Mining keywords (High Recall)...")
    try:
        # Context management
        content_for_mining = jd_text[:12000]
        raw_data = miner_chain.invoke({"jd_content": content_for_mining})
        print(f"Minerd found {len(raw_data.raw_keywords)} potential keywords.")
    except Exception as e:
        print(f"Mining Error: {e}")
        return JDExtraction(technical_stack=[], tools_and_platforms=[], domain_knowledge=[], soft_skills=[], years_experience_min=0)

    # ==========================================
    # STEP 2: THE JUDGE (High Precision)
    # ==========================================
    # The Judge takes the raw list AND the original JD to make decisions.
    judge_llm = llm.with_structured_output(JDExtraction)

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a Senior Technical Recruiter acting as a Judge. 
         You will receive a list of 'Potential Keywords' extracted from a Job Description.
         
         YOUR TASK:
         Review the list against the original Job Description and produce a FINAL, CLEAN LIST.
         
         JUDGING RULES:
         1. **Relevance:** Remove keywords that are mentioned but not required (unless they are a strong 'plus').
         2. **Redundancy:** Deduplicate (e.g., if 'AWS' and 'Amazon Web Services' both exist, keep only 'AWS').
         3. **Specificity:** Remove generic terms like "Coding", "Software", "Computer" unless they refer to a specific certification.
         4. **Categorization:** Sort the remaining keywords strictly into the correct categories (Stack vs Tools vs Domain).
         5. **Soft Skills:** Keep only the top 3-5 most emphasized soft skills. Delete generic fluff like "Hard worker".
         """
        ),
        ("human", 
         """
         Original Job Description Context:
         {jd_context}
         
         ---
         Potential Keywords List (Draft):
         {raw_list}
         ---
         
         Produce the final approved list:
         """)
    ])
    
    judge_chain = judge_prompt | judge_llm

    print("Step 2: Judging keywords (High Precision)...")
    try:
        final_data = judge_chain.invoke({
            "jd_context": content_for_mining[:5000], # Summary context is usually enough
            "raw_list": str(raw_data.raw_keywords)
        })
        print("Judge has finished filtering.")
        return final_data
    except Exception as e:
        print(f"Judge Error: {e}")
        return JDExtraction(technical_stack=[], tools_and_platforms=[], domain_knowledge=[], soft_skills=[], years_experience_min=0)
