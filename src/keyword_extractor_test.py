"""
Test file for experimenting with different approaches to extracting
keywords from a job description.

This file is intended for development and testing purposes only.
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

#loading gemii api
load_dotenv()
google_api = os.getenv("GOOGLE_API_KEY")

# --- Define the Structured Output Schema ---
class JDExtraction(BaseModel):
    """Schema for extracting structured data from a Job Description."""
    hard_skills: List[str] = Field(
        ...,
        description="A list of specific technical skills, tools, programming languages, and frameworks required (e.g., 'Python', 'PyTorch', 'AWS', 'Docker'). Must be exact keywords."
    )
    soft_skills: List[str] = Field(
        ...,
        description="A list of required non-technical/behavioral skills and attributes (e.g., 'cross-functional collaboration', 'written and verbal communication', 'problem-solving'). Must be exact phrases."
    )
    domain_knowledge: List[str] = Field(
        ...,
        description="A list of specific industry knowledge or specialized fields required (e.g., 'Financial Services', 'LLMOps', 'Natural Language Processing')."
    )
    years_experience_min: int = Field(
        ...,
        description="The minimum number of years of professional experience explicitly stated in the JD. Return 0 if not specified."
    )


def extract_jd_features1(jd_file_path: str, gemini_api_key: str = None) -> JDExtraction:
    """
    Reads a Job Description from a file and uses the LLM to extract key features 
    (hard skills, soft skills, domain knowledge, experience).
    
    Args:
        jd_file_path (str): Path to the text file containing the Job Description.
        gemini_api_key (str): The Google API key for the LLM.
        
    Returns:
        JDExtraction: A Pydantic object containing the structured keywords.
    """
    if not gemini_api_key: 
        gemini_api_key = google_api
        if not gemini_api_key:
            raise ValueError("GEMINI API KEY not found")
    
    # 1. Read the Job Description Text
    if not os.path.exists(jd_file_path):
        raise FileNotFoundError(f"Job Description file not found at: {jd_file_path}")
        
    try:
        with open(jd_file_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {jd_file_path}: {e}")

    # 2. Initialize Model and Structured Chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        api_key=gemini_api_key
    )
    
    structured_llm = llm.with_structured_output(JDExtraction)

    # 3. Define the Precision Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert ATS (Applicant Tracking System) parser. "
         "Your sole task is to extract the ABSOLUTE KEY REQUIREMENTS from the job description and categorize them. "
         "Precision is critical. Use the exact wording from the text for all skill lists."
        ),
        ("human", 
         "Analyze the Job Description below and extract the requirements using the provided schema. "
         "Job Description:\n---\n{jd_content}\n---"
        )
    ])

    # 4. Create and Invoke the Chain
    chain = prompt | structured_llm
    
    # Truncate to a reasonable size if the JD is excessively long
    content_to_process = jd_text[:10000]
    
    print("Extracting keywords from Job Description...")

    try:
        # The invoke method directly returns the Pydantic object
        extracted_data = chain.invoke({"jd_content": content_to_process})
        print("Extraction complete.")
        return extracted_data
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        # Return an empty structure on failure
        return JDExtraction(hard_skills=[], soft_skills=[], domain_knowledge=[], years_experience_min=0)



##################################################################################################################################


# --- 1. Expanded Schema for Maximum Coverage ---
class DetailedJDExtraction(BaseModel):
    """Schema for exhaustive keyword extraction from Job Descriptions."""
    
    technical_stack: List[str] = Field(
        ...,
        description="Core programming languages, frameworks, and databases (e.g., Python, React, PostgreSQL). Include versions if specified (e.g., 'Python 3.8+')."
    )
    tools_and_platforms: List[str] = Field(
        ...,
        description="DevOps tools, cloud platforms, IDEs, and software mentioned anywhere in the text (e.g., AWS, Docker, JIRA, Figma, Linux, Git)."
    )
    methodologies_and_concepts: List[str] = Field(
        ...,
        description="processes, workflows, and concepts (e.g., Agile, Scrum, CI/CD, TDD, RESTful APIs, Microservices, LLM Fine-tuning)."
    )
    soft_skills_and_traits: List[str] = Field(
        ...,
        description="Behavioral traits and professional attributes (e.g., 'ownership', 'mentor junior devs', 'fast-paced environment')."
    )
    domain_keywords: List[str] = Field(
        ...,
        description="Industry-specific terms (e.g., 'FinTech', 'HIPAA compliance', 'E-commerce', 'SaaS')."
    )

def extract_jd_features2(jd_file_path: str, gemini_api_key: str = None) -> DetailedJDExtraction:
    
    # Check file existence
    if not os.path.exists(jd_file_path):
        raise FileNotFoundError(f"Job Description file not found at: {jd_file_path}")
        
    try:
        with open(jd_file_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {jd_file_path}: {e}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        api_key=gemini_api_key
    )
    
    structured_llm = llm.with_structured_output(DetailedJDExtraction)

    # --- 2. The "Deep Scan" Prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a ruthless ATS keyword extractor. Your goal is to maximize keyword coverage.
         
         RULES FOR EXTRACTION:
         1. SCAN EVERY SECTION: Do not limit yourself to the 'Requirements' or 'Skills' section. Keywords are often hidden in 'Responsibilities', 'About Us', or 'Day to Day' paragraphs.
         2. EXTRACT TOOLS: If the text says 'You will use JIRA to track tickets', extract 'JIRA'.
         3. CAPTURE CONCEPTS: If the text says 'build scalable APIs', extract 'Scalable APIs' and 'API Design'.
         4. NO HALLUCINATIONS: Only extract what is explicitly written or strongly implied by a specific task.
         5. KEEP IT GRANULAR: Split 'Python/Django' into 'Python' and 'Django'.
         """
        ),
        ("human", 
         "Analyze this Job Description and extract every relevant keyword:\n\n---\n{jd_content}\n---"
        )
    ])

    chain = prompt | structured_llm
    
    # Limit context if necessary, but 10k chars is usually fine for JDs
    content_to_process = jd_text[:12000]
    
    print("Performing deep keyword scan on Job Description...")

    try:
        extracted_data = chain.invoke({"jd_content": content_to_process})
        
        # Optional: Quick debug print to see the volume of keywords found
        total_keywords = len(extracted_data.technical_stack) + len(extracted_data.tools_and_platforms)
        print(f"Extraction complete. Found {total_keywords} technical keywords.")
        
        return extracted_data
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        # Return empty object on failure to prevent crash
        return DetailedJDExtraction(
            technical_stack=[], tools_and_platforms=[], methodologies_and_concepts=[], 
            soft_skills_and_traits=[], domain_keywords=[]
        )



##################################################################################################################################

# --- Schema (Unchanged) ---
class DetailedJDExtraction1(BaseModel):
    technical_stack: List[str] = Field(..., description="Core programming languages, frameworks, and databases.")
    tools_and_platforms: List[str] = Field(..., description="DevOps tools, cloud platforms, IDEs, and software.")
    methodologies_and_concepts: List[str] = Field(..., description="Processes, workflows, and technical concepts.")
    soft_skills_and_traits: List[str] = Field(..., description="Behavioral traits and professional attributes.")
    domain_keywords: List[str] = Field(..., description="Industry-specific terms.")

def extract_jd_features(jd_file_path: str, gemini_api_key: str = None) -> DetailedJDExtraction:
    
    if not os.path.exists(jd_file_path):
        raise FileNotFoundError(f"File not found: {jd_file_path}")
        
    try:
        with open(jd_file_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file: {e}")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=gemini_api_key)
    structured_llm = llm.with_structured_output(DetailedJDExtraction1)

    # --- UPDATED PROMPT WITH FEW-SHOT & CONSTRAINTS ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an expert Technical Recruiter and ATS Parser. 
         Your goal is to extract a precise, high-signal list of keywords from the Job Description.

         ### NEGATIVE CONSTRAINTS (DO NOT EXTRACT):
         1. DO NOT extract generic headers or filler words (e.g., "Responsibilities", "Requirements", "Duties", "Candidate", "Role").
         2. DO NOT extract benefit perks (e.g., "401k", "Remote work", "Salary", "Ping pong table").
         3. DO NOT extract vague verbs alone (e.g., "Working", "Building", "Trying").
         4. DO NOT extract educational degrees unless specifically required as a certification (e.g., ignore "Bachelor's degree").

         ### FEW-SHOT EXAMPLES (FOLLOW THIS LOGIC):
         
         **Example 1: Granularity & Splitting**
         *Input:* "Experience with Python/Django and CI/CD pipelines using Jenkins or GitLab."
         *Output:* - technical_stack: ["Python", "Django"]
            - tools_and_platforms: ["Jenkins", "GitLab"]
            - methodologies_and_concepts: ["CI/CD"]
            (Note: "Python/Django" was split. "pipelines" was ignored as generic context.)

         **Example 2: Implied Skills (Inference)**
         *Input:* "You will containerize applications and orchestrate them in the cloud."
         *Output:* - methodologies_and_concepts: ["Containerization", "Cloud Orchestration"]
            - tools_and_platforms: ["Docker", "Kubernetes"] 
            (Note: Extracted "Docker" and "Kubernetes" even though they weren't explicitly named, because they are the industry standard for this task.)

         **Example 3: Acronym Handling**
         *Input:* "Knowledge of NLP and Large Language Models."
         *Output:* - domain_keywords: ["NLP", "Natural Language Processing", "Large Language Models", "LLMs"]
            (Note: Captured both the acronym and the full form for maximum ATS coverage.)
         """
        ),
        ("human", 
         "Analyze this Job Description and extract the key requirements:\n\n---\n{jd_content}\n---"
        )
    ])

    chain = prompt | structured_llm
    
    # Context window management
    content_to_process = jd_text[:12000] 
    
    try:
        return chain.invoke({"jd_content": content_to_process})
    except Exception as e:
        print(f"Extraction Error: {e}")
        return DetailedJDExtraction1(
            technical_stack=[], tools_and_platforms=[], methodologies_and_concepts=[], 
            soft_skills_and_traits=[], domain_keywords=[]
        )






######################################################################################################


# --- 1. New Nested Model for Granular Control ---
class Keyword(BaseModel):
    name: str = Field(
        ..., 
        description="The exact technical keyword or skill (e.g., 'Python', 'Agile')."
    )
    importance: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description=(
            "CRITICAL: Must-have requirement / Dealbreaker. "
            "HIGH: Important requirement. "
            "MEDIUM: Preferred / Bonus skill. "
            "LOW: Minor mention or soft preference."
        )
    )

# --- 2. Updated Parent Schema ---
class JDExtraction1(BaseModel):
    """Schema for extracting structured data from a Job Description."""
    
    # We now use List[Keyword] instead of List[str]
    technical_stack: List[Keyword] = Field(
        ...,
        description="Core programming languages, frameworks, and databases with importance scoring."
    )
    tools_and_platforms: List[Keyword] = Field(
        ...,
        description="DevOps tools, cloud platforms, and software with importance scoring."
    )
    domain_knowledge: List[Keyword] = Field(
        ...,
        description="Industry-specific knowledge with importance scoring."
    )
    soft_skills: List[Keyword] = Field(
        ...,
        description="Behavioral skills and attributes with importance scoring."
    )
    years_experience_min: int = Field(
        ...,
        description="Minimum years of experience required (0 if not specified)."
    )

def extract_jd_features(jd_file_path: str, gemini_api_key: str = None) -> JDExtraction1:
    """
    Reads a JD and uses the LLM to extract scored keywords.
    """
    if not gemini_api_key: 
        load_dotenv()
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI API KEY not found")
            
    if not os.path.exists(jd_file_path):
        raise FileNotFoundError(f"File not found: {jd_file_path}")
        
    try:
        with open(jd_file_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {jd_file_path}: {e}")

    # Initialize Model (Using gemini-2.0-flash as requested, or fallback to 1.5)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0,
        api_key=gemini_api_key
    )
    
    structured_llm = llm.with_structured_output(JDExtraction)

    # --- 3. Prompt with Few-Shot Examples & Scoring Logic ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an expert Technical Recruiter. Your goal is to extract keywords and SCORE their importance.

         ### SCORING RULES:
         - **CRITICAL**: The JD says "Must have", "Required", or "Essential".
         - **HIGH**: The JD lists it in the main requirements bullet points.
         - **MEDIUM**: The JD says "Nice to have", "Plus", or "Preferred".
         - **LOW**: Mentioned in "Day to Day" or general context but not a hard requirement.

         ### NEGATIVE CONSTRAINTS:
         1. DO NOT extract generic headers (e.g., "Responsibilities", "Duties").
         2. DO NOT extract benefits (e.g., "401k", "Remote").
         3. DO NOT extract vague verbs (e.g., "Working", "Building").

         ### FEW-SHOT EXAMPLES:
         
         **Input:** "Required: Expert in Python and Django. Experience with AWS is a plus."
         **Output:** - technical_stack: [
               {{"name": "Python", "importance": "CRITICAL"}}, 
               {{"name": "Django", "importance": "CRITICAL"}}
             ]
           - tools_and_platforms: [
               {{"name": "AWS", "importance": "MEDIUM"}}
             ]

         **Input:** "We are looking for a team player who can communicate effectively."
         **Output:**
           - soft_skills: [
               {{"name": "Team Player", "importance": "HIGH"}},
               {{"name": "Communication", "importance": "HIGH"}}
             ]
         """
        ),
        ("human", 
         "Analyze this Job Description and extract keywords with scores:\n\n---\n{jd_content}\n---"
        )
    ])

    chain = prompt | structured_llm
    
    # Truncate to avoid context limit issues
    content_to_process = jd_text[:12000]
    
    print("Extracting scored keywords from Job Description...")

    try:
        return chain.invoke({"jd_content": content_to_process})
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        return JDExtraction1(
            technical_stack=[], tools_and_platforms=[], 
            domain_knowledge=[], soft_skills=[], years_experience_min=0
        )


################################################################################################

# --- 1. Revert to Simple Schema (The Judge's Final Output) ---
class JDExtraction2(BaseModel):
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

def extract_jd_features(jd_file_path: str, gemini_api_key: str = None) -> JDExtraction2:
    
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
        return JDExtraction2(technical_stack=[], tools_and_platforms=[], domain_knowledge=[], soft_skills=[], years_experience_min=0)

    # ==========================================
    # STEP 2: THE JUDGE (High Precision)
    # ==========================================
    # The Judge takes the raw list AND the original JD to make decisions.
    judge_llm = llm.with_structured_output(JDExtraction2)

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
        return JDExtraction2(technical_stack=[], tools_and_platforms=[], domain_knowledge=[], soft_skills=[], years_experience_min=0)

