import os
import re
import copy
import yaml
import pathlib as Path
import pickle
import subprocess
from fpdf import FPDF
from datetime import datetime

####################################################################################################################################################
####################################################################################################################################################
##################################################################  FILE LOADER   ##################################################################

def load_file(file_path, file_type='tex'):
    """
    Efficiently validates and loads tex, yml, or pkl files.
    """
    # 1. Validation
    if not os.path.isfile(file_path):
        print(f" Invalid file path: {file_path}")
        return None

    # 2. Loading Logic
    try:
        # Pickle requires binary mode ('rb'), others use text mode ('r')
        mode = 'rb' if file_type == 'pkl' else 'r'
        encoding = None if file_type == 'pkl' else 'utf-8'

        with open(file_path, mode, encoding=encoding) as f:
            if file_type == 'yml':
                return yaml.safe_load(f)
            elif file_type == 'pkl':
                return pickle.load(f)
            else:
                return f.read()

    except Exception as e:
        print(f" Error loading {file_type} file at {file_path}: {e}")
        return None


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
    Converts raw text, lists, or dictionaries into valid LaTeX syntax.
    Escapes LaTeX reserved characters and applies specific formatting based on type.

    Args:
        content (str, list, or dict): 
            - str: Returns escaped string.
            - list: Returns escaped items in an itemize environment.
            - dict: Returns \item \textbf{key:} value in an itemize environment.

    Returns:
        str: Formatted LaTeX code, or None if input is empty/invalid.
    """
    if not content:
        return None

    # Create a Translation Table for LaTeX reserved characters
    latex_escapes = str.maketrans({
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    })

    # Handle String Input (e.g., Professional Summary)
    if isinstance(content, str):
        return content.translate(latex_escapes)

    # Handle List Input (e.g., General Highlights)
    elif isinstance(content, list):
        items = [f"\\item {str(item).translate(latex_escapes)}" for item in content]
        return (
            "\\begin{itemize}\n"
            f"    {'\n    '.join(items)}\n"
            "\\end{itemize}"
        )

    # Handle Dictionary Input (e.g., Technical Skills)
    elif isinstance(content, dict):
        items = []
        for key, value in content.items():
            # Clean the key
            clean_key = str(key).translate(latex_escapes)
            
            # Check if the value is a list; if so, join with commas
            if isinstance(value, list):
                processed_val = ", ".join(str(v) for v in value)
            else:
                processed_val = str(value)
            
            # Clean the resulting value string
            clean_val = processed_val.translate(latex_escapes)
            
            items.append(f"\\item \\textbf{{{clean_key}:}} {clean_val}")
        
        return (
            "\\begin{itemize}\n"
            f"    {'\n    '.join(items)}\n"
            "\\end{itemize}"
        )

    return None


####################################################################################################################################################
####################################################################################################################################################
###################################################   WRITE THE LATEX CODE IN THE DESIRED FILE   ###################################################


def write_section_content(file_map, section, title, content_data):
    """
    Writes content to a LaTeX file. If the content is an itemized list, it 
    identifies the itemize environment in the existing file and replaces 
    only that block, preserving external headers or metadata.

    Args:
        file_map (dict): Mapping of {section: {title: path}}.
        section (str): The category key for lookup.
        title (str): The specific item title for lookup.
        content_data (str): The new LaTeX content to be written.

    Returns:
        bool: True if write was successful, False otherwise.
    """
    #  Resolve file path from the mapping
    try:
        target_path = file_map[section][title]
    except KeyError:
        print(f"Error: Key mismatch for section '{section}' and title '{title}'")
        return False

    #  Validate content type and file existence
    if not isinstance(content_data, str) or not os.path.exists(target_path):
        print(f"Error: Invalid content type or path does not exist: {target_path}")
        return False

    try:
        #  Read existing file content
        with open(target_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        #  Perform targeted replacement for itemized lists
        # We look for \begin{itemize} to determine if this is a list update
        if "\\begin{itemize}" in content_data:
            pattern = r"\\begin\{itemize\}.*?\\end\{itemize\}"
            
            # If the original file contains a list, replace only that segment
            if re.search(pattern, original_content, flags=re.DOTALL):
                # We extract the clean list from the new data
                new_list = re.search(pattern, content_data, flags=re.DOTALL).group(0)
                
                # Replace the old list with the new one. 
                # .replace("\\", "\\\\") handles backslashes for the regex engine.
                final_output = re.sub(
                    pattern, 
                    new_list.replace("\\", "\\\\"), 
                    original_content, 
                    flags=re.DOTALL
                )
            else:
                # No existing list found in target file; proceed with full overwrite
                final_output = content_data
        else:
            # For non-list sections (e.g., Profile), replace entire file content
            final_output = content_data

        #  Commit changes to disk
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(final_output)
            
        print(f"Successfully updated: {title}")
        return True

    except Exception as e:
        print(f"Error occurred while writing to {target_path}: {e}")
        return False


####################################################################################################################################################
####################################################################################################################################################
##########################################################   Reorder Experience Section   ##########################################################

def exp_reorder(file_path, ranked_keys, bridge_dict):
    """
    Replaces the LaTeX content between two %exp comment anchors 
    with a new order of subimports.
    """
    # Create the new string and escape backslashes for the Regex engine
    new_experience_block = "\n".join([bridge_dict[key] for key in ranked_keys])
    safe_experience_block = new_experience_block.replace("\\", "\\\\")
    
    #Read the main file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    #Define the Pattern to look for %exp anchors
    # % is a special char in some contexts, but here it matches literally.
    # We look for %exp ... some content ... %exp
    pattern = r"(%exp)(.*?)(%exp)"
    
    #Check if the anchors exist before proceeding
    if not re.search(pattern, content, flags=re.DOTALL):
        print("Error: The comment anchors '%exp' were not found in the file.")
        print("Instruction: Please wrap your experience subimports in your main.tex like this:")
        print("\n%exp\n\\subimport{...}{...}\n%exp\n")
        return False

    #Perform the substitution
    # \1 is the first %exp, \3 is the second %exp
    new_content = re.sub(
        pattern, 
        rf"\1\n{safe_experience_block}\n\3", 
        content, 
        flags=re.DOTALL
    )

    #Write back to the file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully reordered {len(ranked_keys)} experiences in {file_path}")
        return True
    except Exception as e:
        print(f"Failed to write to file: {e}")
        return False

####################################################################################################################################################
####################################################################################################################################################
##############################################################   Compile Latex File   ##############################################################


def compile_latex(main_file_path, output_dir=None):
    """
    Refined LuaLaTeX compiler that handles complex paths by 
    isolating the working directory.
    """
    # 1. Expand to absolute path to avoid any ambiguity
    abs_main_path = os.path.abspath(main_file_path)
    
    if not os.path.isfile(abs_main_path):
        print(f"Error: File not found at {abs_main_path}")
        return False

    # 2. Split path into directory and filename
    # This allows us to 'cd' into the folder and run the file by name
    file_dir = os.path.dirname(abs_main_path)
    file_name = os.path.basename(abs_main_path)
    
    target_output = os.path.abspath(output_dir) if output_dir else file_dir

    # 3. Construct the command
    # We use the file_name only because we will set the cwd to file_dir
    command = [
        "lualatex",
        "-interaction=nonstopmode",
        f"-output-directory={target_output}",
        file_name
    ]

    try:
        for pass_count in range(1, 3):
            print(f"Executing LuaLaTeX pass {pass_count}...")
            
            result = subprocess.run(
                command,
                cwd=file_dir,  # Crucial: Change context to the folder itself
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Error: Compilation failed during pass {pass_count}")
                # Log the bottom of the stdout to see the specific LaTeX error
                log_lines = result.stdout.splitlines()
                print("\n".join(log_lines[-20:]))
                return False

        print(f"Successfully compiled: {file_name}")
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


####################################################################################################################################################
####################################################################################################################################################
##########################################################   BUILD PDF FILE FROM STRING   ##########################################################


def convert_str_to_pdf(content_str: str, output_dir: str, output_filename: str = "Cover_Letter.pdf"):
    """
    Converts a plain text string into a professional PDF cover letter using FPDF2.
    Saves the file to the specified output directory.
    """

    # Sanitize text for standard PDF fonts (Times does not support Unicode – dashes)
    replacements = {
        "\u2013": "-", 
        "\u2014": "--", 
        "\u2018": "'", 
        "\u2019": "'", 
        "\u201c": '"', 
        "\u201d": '"', 
        "\u2026": "..."
    }
    
    for char, replacement in replacements.items():
        content_str = content_str.replace(char, replacement)
    
    # Ensure the directory exists
    if not os.path.exists(output_dir):
        print(f"Output dir not found: {output_dir}")
        return False

    # Construct the full file path
    full_path = os.path.join(output_dir, output_filename)

    # Initialize PDF (A4 size)
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=25.4, top=25.4, right=25.4) 
    pdf.add_page()
    
    # Set Font
    pdf.set_font("Times", size=12)
    
    # Add Date (Top Right)
    date_str = datetime.now().strftime("%B %d, %Y")
    pdf.set_xy(150, 20)
    pdf.cell(0, 10, date_str, align='R')
    
    # Move to Content Area
    pdf.set_y(40)
    
    # Handle the main text content
    paragraphs = content_str.split('\n\n')
    for para in paragraphs:
        if para.strip():
            pdf.multi_cell(w=0, h=6, text=para.strip(), align='L')
            pdf.ln(5) 

    # 3. Output the file to the full path
    try:
        pdf.output(full_path)
        print(f"Success! PDF saved at: {full_path}")
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

