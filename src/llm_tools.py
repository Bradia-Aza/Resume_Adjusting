import os
import json
import re
from dotenv import load_dotenv
from typing import List, Literal, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field



####################################################################################################################################################
####################################################################################################################################################
###############################################################   ENRICH META DATA   ###############################################################


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
                # Update the section key to the newly identified category
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
    

####################################################################################################################################################
####################################################################################################################################################
#######################################################   EXTRACT JOB DESCRIPION KEY WORDS   #######################################################


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

def extract_jd_features(jd_text: str, gemini_api_key: str = None) -> JDExtraction:
    
    if not gemini_api_key: 
        load_dotenv()
        gemini_api_key = os.getenv("GOOGLE_API_KEY")

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

####################################################################################################################################################
####################################################################################################################################################
################################################   TAILOR PROFILE AND HIGHLIGHT OF QUALIFICATIONS   ################################################


# Helper to clean LLM output (LLMs often wrap JSON in markdown ```json ... ```)
def parse_llm_json(llm_output):
    try:
        # Remove markdown code blocks if present
        clean_text = re.sub(r'```json\n?|```', '', llm_output).strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        print("Error parsing JSON. Raw output:", llm_output)
        return None


def tailor_profile_and_highlights( resume_context, job_description, keywords, gemini_api_key:str = None ):
    """
    Args:
        llm: Your configured LangChain LLM object
        resume_context (str): The XML-like string built by your file loader
        job_description (str): Text of the target JD
        keywords (list): List of extracted keywords strings
        
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


    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert career strategist and professional writer. "
        "Your goal is to tailor resume sections to align with a job description while maintaining "
        "strict honesty and a natural, human narrative flow. "
        "You prioritize coherence and readability over keyword density."
    ),
    (
        "human",
        """
    ### INPUT DATA
    JOB DESCRIPTION:
    {job_description}

    TARGET KEYWORDS (Use only if applicable):
    {keywords}

    CURRENT RESUME CONTENT (Source of Truth):
    {resume_context}

    ### WRITING GUIDELINES
    1. **Truth Anchor:** All claims must be supported by evidence in the `CURRENT RESUME CONTENT`. Do not invent skills or experiences just to match the JD.
    2. **Selective Integration:** Do NOT try to use every keyword. Only include keywords from the list that genuinely match the user's existing experience. If a keyword doesn't fit naturally, discard it.
    3. **Human Tone:** Avoid buzzword-stuffing. Write in a confident, professional, yet conversational tone. Use varied sentence structures to avoid a robotic feel.

    ### INSTRUCTIONS
    1. Analyze the <component section='Profile'> and <component section='Highlights'> in the provided resume.
    2. Rewrite the **"Profile"** (3–4 sentences):
       - Create a narrative hook connecting the user's actual background to the JD's biggest pain point.
       - Focus on the *value* the user brings, rather than listing skills.
    3. Rewrite the **"Highlights"** (4–6 bullet points):
       - Select the strongest matches between the user's history and the JD.
       - Focus on outcomes and impact, using the keywords only where they enhance the description.
    4. **Strict Output:** Return ONLY a valid JSON object.
    
    ### STYLE GUARDRAILS (STRICT):
    1. **Ban Meta-Phrases:** Do not use phrases like "demonstrated history of," "proven track record," "showcasing expertise in," or "proficiency in." Just state the action (e.g., instead of "Demonstrated ability to code in Python," write "Coded in Python").
    2. **No Mission Statement Mimicry:** Do not copy the company's high-level mission (e.g., "societal benefit," "industry-defining") into the user's profile. Keep the profile focus on the *technical value* the user adds.
    3. **Simplify Connectors:** Avoid the word "leveraging." Use "using" or "utilizing."

    ### JSON SCHEMA
    {{
        "profile": "The humanized, coherent paragraph...",
        "highlights": [
            "Impact-driven bullet point 1...",
            "Impact-driven bullet point 2..."
        ]
    }}
    """
    )
    ])  
    # Format the prompt
    formatted_prompt = prompt.format_messages(
        job_description=job_description,
        keywords=", ".join(keywords),
        resume_context=resume_context
    )
    
    # Invoke LLM (adjust syntax depending on your LangChain version)
    response = llm.invoke(formatted_prompt)
    
    # Handle response (if your LLM returns an object, extract .content, otherwise use string)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Parse into Python Dictionary
    data = parse_llm_json(response_text)
    
    return datae


####################################################################################################



def rank_experiences(jd: str, experience_list: List[str], rsm_exp: str, gemini_api_key: str = None) -> List[str]:
    """
    Reorders a list of candidate experiences based on their relevance to a job description using Gemini.

    Args:
        jd (str): The full text of the Job Description.
        experience_list (List[str]): A list of experience titles (or identifiers) to be sorted.
        rsm_exp (str): The detailed text content of the candidate's projects/experiences.
        gemini_api_key (str, optional): Google API Key. If None, attempts to load from environment.

    Returns:
        List[str]: The 'experience_list' reordered by relevance. 
                   Returns the original list if JSON parsing fails.

    Raises:
        ValueError: If the Gemini API key is missing.
    """
    
    # --- 1. API Key Setup ---
    if not gemini_api_key: 
        # Loads variables from .env file if not passed explicitly
        load_dotenv()
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not gemini_api_key:
            raise ValueError("GEMINI API KEY not found. Please set it in .env or pass it as an argument.")

    # --- 2. Initialize Model ---
    # We use a temperature of 0 for deterministic (consistent) ranking results
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        api_key=gemini_api_key
    )

    # --- 3. Define the Prompt ---
    # We use `from_template` because we are passing a single string instructions block.
    template_text = """
    You are an expert technical recruiter. Reorder the candidate's projects based on relevance to the Job Description.

    JOB DESCRIPTION:
    {job_description}

    CANDIDATE'S PROJECTS (Detailed Context):
    {resume_exp}

    LIST OF TITLES TO SORT:
    {experience_list}

    INSTRUCTIONS:
    1. Read the content of each project (in CANDIDATE'S PROJECTS) to understand the actual skills used (e.g., Python, RAG, React).
    2. Compare these skills to the Job Description requirements.
    3. Rank the items in "LIST OF TITLES TO SORT" from MOST RELEVANT to LEAST RELEVANT.
    4. Return ONLY a valid JSON list of strings from the experience list.
    
    OUTPUT FORMAT:
    Return a raw JSON list of strings. Do not include markdown formatting.
    Example: ["Title B", "Title A", "Title C"]
    """
    
    prompt = ChatPromptTemplate.from_template(template_text)

    # --- 4. Chain Execution ---
    # We pipe the formatted prompt directly to the LLM
    chain = prompt | llm
    
    print(" Asking LLM to rank experiences based on context...")
    
    # invoke() automatically formats the prompt using the dictionary provided
    response = chain.invoke({
        "job_description": jd,
        "resume_exp": rsm_exp,
        "experience_list": json.dumps(experience_list) # Converting list to string for safer prompt injection
    })
    
    # --- 5. Output Parsing ---
    response_text = response.content
    
    try:
        # Clean up potential markdown code blocks (e.g., ```json ... ```)
        clean_text = re.sub(r'```json\n?|```', '', response_text).strip()
        
        # Parse the string into a Python list
        sorted_titles = json.loads(clean_text)
        
        # Validation: Ensure the output is actually a list
        if isinstance(sorted_titles, list):
            return sorted_titles
        else:
            print(f"Warning: LLM returned valid JSON but not a list. Type: {type(sorted_titles)}")
            return experience_list

    except json.JSONDecodeError:
        print(f"Error parsing JSON. Raw response: {response_text}")
        # Fallback: return original order so the pipeline doesn't crash
        return experience_list

