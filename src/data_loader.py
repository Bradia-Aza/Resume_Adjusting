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
########################################################  REFORMAT DEPENDENCIES LIST  #########################################################


def create_dep_map(dependency_list):
    """
    Transforms the list of dicts into a nested dictionary for fast lookup.
    Structure: { 'SECTION_NAME': { 'Title Name': 'full/path/to/file.tex' } }
    """
    file_map = {}
    
    for entry in dependency_list:
        section = entry['section']
        title = entry['title']
        full_path = entry['full_path']
        
        # If section doesn't exist, create it
        if section not in file_map:
            file_map[section] = {}
            
        # Add the title and path
        file_map[section][title] = full_path
        
    return file_map



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
        title = item.get('title', 'Untitled') 
        
        # Read the file content
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


def convert_to_latex(content):
    """
    Converts raw text or a list of strings into valid LaTeX syntax.

    Args:
        content (str or list): The raw content to format.
                               - If str: Returns the string as-is (or safe for LaTeX).
                               - If list: Wraps items in a LaTeX 'itemize' environment.

    Returns:
        str: The formatted LaTeX code, or None if input is empty/invalid.
    """
    # Safety Check: Return None if data is empty
    if not content:
        return None

    # Handle String Input (e.g., Profile Summary)
    if isinstance(content, str):
        # Return the string directly. 
        # (Optional: You could add a helper here to escape special chars like % or &)
        return content

    # Handle List Input (e.g., Highlights of Qualifications)
    elif isinstance(content, list):
        # Convert each list item into a \item line
        # We use a list comprehension for efficiency
        items_latex = "\n    ".join([f"\\item {item}" for item in content])
        
        # Wrap the items in the itemize environment
        latex_block = (
            "\\begin{itemize}\n"
            f"    {items_latex}\n"
            "\\end{itemize}"
        )
        return latex_block

    # Handle Unsupported Types
    else:
        print(f"Warning: Unsupported type {type(content)} passed to convert_to_latex.")
        return None


####################################################################################################################################################
####################################################################################################################################################
###################################################   WRITE THE LATEX CODE IN THE DESIRED FILE   ###################################################


def write_section_content(file_map, section, title, content_data):
    """
    Writes string content to a specific resume file identified by section and title.
    """
    
    # 1. Locate the file path using the map
    try:
        target_path = file_map[section][title]
    except KeyError:
        print(f"Error: Could not find file path for Section: '{section}', Title: '{title}'")
        return False

    # 2. Check if content is String AND Path exists
    # We use 'and' instead of '&', and ensure the logic flows correctly
    if isinstance(content_data, str) and os.path.exists(target_path):
        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content_data)
            print(f"Updated: {title}")
            return True
        except Exception as e:
            print(f"Error writing to {target_path}: {e}")
            return False
    else:
        print(f"Error: Content must be 'str' and path must exist. Got type: {type(content_data)}")
        return False

    