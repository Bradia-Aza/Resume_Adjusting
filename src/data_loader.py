import os
import re
import copy
import pathlib as Path
import pickle




####################################################################################################################################################
####################################################################################################################################################
#############################################################  EXTRACT DEPENDENCIES   ##############################################################

def extract_latex_dependencies(main_file_path: str) -> list[dict] | dict:
    """
    Parses a main LaTeX file to identify imported sub-files and the 
    corresponding section they belong to.

    It extracts files linked by commands like \\input, \\include, and 
    \\subimport, categorizing them under the most recently encountered 
    \\resumesection command.

    Args:
        main_file_path: The absolute or relative path to the main .tex file.

    Returns:
        A list of dictionaries, where each dictionary represents a dependency 
        and contains the keys 'section' (str) and 'full_path' (str).
        Returns a dictionary with an 'error' key if the main file is not found.
    """
    
    dependencies = []
    # Determine the base directory for resolving relative paths
    base_dir = os.path.dirname(main_file_path)
    
    try:
        with open(main_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return {"error": f"File not found at {main_file_path}"}

    # Regex captures: (1) command, (2) first argument, (3) optional second argument
    pattern = re.compile(r'\\(input|include|subimport|resumesection)\{([^}]+)\}(?:\{([^}]+)\})?')
    
    current_section = "Uncategorized" # State variable for section context

    for match in pattern.finditer(content):
        command = match.group(1)
        arg1 = match.group(2).strip()
        arg2 = match.group(3).strip() if match.group(3) else None

        # If it's a section header, update the current context
        if command == 'resumesection':
            current_section = arg1
        # Otherwise, process it as a file dependency
        else:
            entry = {
                "section": current_section,
                "full_path": None
            }

            if command == 'subimport' and arg2:
                # Handle \subimport{dir}{file}
                combined_path = os.path.join(base_dir, arg1, arg2)
            else:
                # Handle \input{file} or \include{file}
                combined_path = os.path.join(base_dir, arg1)

            # Resolve and normalize the final path
            entry["full_path"] = os.path.normpath(combined_path)

            # Ensure the dependency has the .tex extension
            if not entry["full_path"].endswith('.tex'):
                entry["full_path"] += ".tex"
                
            dependencies.append(entry)

    return dependencies


####################################################################################################################################################
####################################################################################################################################################
############################################################  ADJUST DEPENDENCIES PATH  ############################################################


def rebase_dependency_paths(dependency_list, new_project_root, old_project_root=None):
    """
    Updates the 'full_path' in the dependency list to point to the new directory.
    
    Args:
        dependency_list (list): Your list of dicts.
        new_project_root (str): The absolute path to your COPIED resume folder.
        old_project_root (str, optional): The old root. If None, it attempts to 
                                          auto-detect it based on the common prefix.
    """
    # Create a deep copy so we don't mess up the original list in memory
    updated_list = copy.deepcopy(dependency_list)
    
    # 1. Auto-detect old root if not provided
    if old_project_root is None:
        # Get all paths
        all_paths = [item['full_path'] for item in dependency_list]
        # Find the longest common folder
        common_prefix = os.path.commonpath(all_paths)
        
        # Assumption:structure is likely project_root/components/...
        # So we want the parent of the common prefix
        if "components" in common_prefix:
            # Go up one level 
            old_project_root = os.path.dirname(common_prefix)
        else:
            old_project_root = common_prefix

    print(f"Rebasing paths...\nFROM: {old_project_root}\nTO:   {new_project_root}\n")

    # 2. Update every path
    for item in updated_list:
        old_path = item['full_path']
        
        # Calculate the relative path 
        # This strips the old_root part off the front
        relative_path = os.path.relpath(old_path, start=old_project_root)
        
        # Join it to the new root
        new_path = os.path.join(new_project_root, relative_path)
        
        # Update the dictionary
        item['full_path'] = new_path

    return updated_list


####################################################################################################################################################
####################################################################################################################################################
########################################################  BUILD RESUME USING DEPENDENCIES  #########################################################


def build_resume_context(dependency_list):
    """
    Takes the dependency list and creates a single tagged string
    containing the entire resume content.
    """
    full_context_text = ""

    for item in dependency_list:
        path = item['full_path']
        section = item['section']
        title = item.get('title', 'Untitled') # Use .get() in case title is missing
        
        # 1. Read the file content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except FileNotFoundError:
            content = f"[ERROR: File not found at {path}]"

        # 2. Create Semantic Tags
        # We include the section and title as attributes so the LLM understands the context
        # e.g., <component section="TECHNICAL EXPERIENCE" title="AI Researcher">
        entry = (
            f"<component section='{section}' title='{title}'>\n"
            f"{content}\n"
            f"</component>\n\n"
        )
        
        full_context_text += entry

    return full_context_text


####################################################################################################################################################
####################################################################################################################################################
#########################################################  FLATTEN KEY WORDS PYDANTIC OBJ  #########################################################


def flatten_pydantic(extraction_obj, target_keys):
    """
    Converts JDExtraction object into a single list of strings
    based on the specific keys provided.
    
    Args:
        extraction_obj: The Pydantic model, class, or dict containing the data.
        target_keys (list): List of strings representing the keys to extract 
                            (e.g. ['technical_stack', 'tools_and_platforms'])
    """
    # --- Standardize Input to Dictionary ---
    if hasattr(extraction_obj, 'model_dump'):
        data = extraction_obj.model_dump()
    elif isinstance(extraction_obj, dict):
        data = extraction_obj
    elif isinstance(extraction_obj, list):
        # If it's already a list, we assume it's already flat and return it
        return extraction_obj
    else:
        print("Can not covert the object to dictionary")
        return []

    # --- Dynamic Extraction ---
    flat_list = []
    for key in target_keys:
        # data.get(key, []) ensures we don't crash if a key is missing
        # extend() adds the items from the list to our flat_list
        items = data.get(key, [])
        if isinstance(items, list):
            flat_list.extend(items)
            
    # Remove duplicates while preserving order
    return list(dict.fromkeys(flat_list))


####################################################################################################################################################
####################################################################################################################################################
#################################################  WRITE LATEX CODE FOR PROFILE AND QUALIFICATION ##################################################


def convert_profile_latex(tailored_data):
    """
    Takes the pure text dictionary(for profile and highlight of qualifications) and wraps it in LaTeX syntax.
    """
    if not tailored_data:
        return None, None

    # Format Profile
    # Assuming profile.tex just needs the text, or \cvsection{Profile} + text
    profile_latex = tailored_data['profile'] 

    # Format Highlights
    # We construct the itemize environment programmatically
    highlights_items = "\n".join([f"\\item {item}" for item in tailored_data['highlights']])
    highlights_latex = (
        "\\begin{itemize}\n"
        f"{highlights_items}\n"
        "\\end{itemize}"
    )
    
    return profile_latex, highlights_latex




    