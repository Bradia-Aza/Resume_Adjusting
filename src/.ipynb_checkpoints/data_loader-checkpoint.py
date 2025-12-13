import os
import re
import pathlib as Path
import pickle

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


