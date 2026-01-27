import os
import re
import copy
import yaml
import re
from pathlib import Path
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


def component_registry(components_path: str) -> Dict[str, Dict[str, str]]:
    """
    Crawls the components directory to build a map of available LaTeX files.
    
    Structure:
    registry[folder_name][file_name_lowercase] = full_path
    
    Example:
    registry['projects']['housing'] = 'components/projects/Housing.tex'
    """
    registry = {}
    base_path = Path(components_path)

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        
        # We only care about folders that actually contain .tex files
        # The folder name becomes our 'category'
        category = root_path.name
        
        # Skip the base directory itself if it has no category name
        if category == base_path.name:
            continue

        for file in files:
            if file.endswith(".tex"):
                # Clean the filename to use as a key (e.g., 'Housing.tex' -> 'housing')
                project_id = file.replace(".tex", "").lower()
                
                if category not in registry:
                    registry[category] = {}
                
                # Store the absolute or relative path for later reading/writing
                registry[category][project_id] = str(root_path / file)

    return registry


####################################################################################################################################################
####################################################################################################################################################
############################################################  ADJUST DEPENDENCIES PATH  ############################################################


def rebase_dependencies(registry, new_project_root, old_project_root=None):
    """
    Updates all file paths in the nested registry dictionary to point to a new directory.
    
    Args:
        registry (dict): The nested dict from component_registry.
        new_project_root (str): The path to the new destination folder.
        old_project_root (str, optional): The original root to strip away. 
                                          If None, it auto-detects from the first path found.
    """
    # 1. Deep copy to avoid modifying the original registry during execution
    updated_registry = copy.deepcopy(registry)
    
    # 2. Auto-detect old root if not provided
    if old_project_root is None:
        # Grab the first actual path string found in the nested structure
        # (Goes into the first category, then the first file_id)
        first_cat = next(iter(registry.values()))
        sample_path = next(iter(first_cat.values()))
        
        # We assume the root is everything before '/components/'
        if "components" in sample_path:
            old_project_root = sample_path.split("components")[0].rstrip(os.sep)
        else:
            # Fallback: just use the directory of the sample path
            old_project_root = os.path.dirname(os.path.dirname(sample_path))

    print(f"Rebasing Registry Paths...\nFROM: {old_project_root}\nTO:   {new_project_root}")

    # 3. Traverse and Update
    for category, files in updated_registry.items():
        for file_id, old_path in files.items():
            # Using LibPath (pathlib) for clean manipulation
            # .relative_to() is the professional way to strip the old root
            rel_path = os.path.relpath(old_path, start=old_project_root)
            
            # Combine new root with the relative path
            new_path = Path(new_project_root) / rel_path
            
            # Update the registry with the string version of the new path
            updated_registry[category][file_id] = str(new_path)

    return updated_registry



####################################################################################################################################################
####################################################################################################################################################
########################################################  BUILD RESUME USING DEPENDENCIES  #########################################################


def build_resume_context(registry):
    """
    Traverses the nested registry dictionary to create a single tagged string
    containing the entire resume content for LLM context.
    
    Args:
        registry (dict): The nested dictionary {category: {file_id: path}}
    """
    full_context_text = ""

    # Iterate through categories (folders)
    for category, files in registry.items():
        # Iterate through specific files in that category
        for file_id, path in files.items():
            
            # 1. Read the file content
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except Exception as e:
                content = f"[ERROR: Could not read file at {path}. Details: {e}]"

            # 2. Create Semantic Tags
            # We use 'category' and 'id' as attributes. 
            # This matches your LLM_TO_RESUME_BRIDGE structure perfectly.
            entry = (
                f"<component category='{category}' id='{file_id}'>\n"
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
    r"""
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

        
        # Look for \begin{itemize} to determine if this is a list update
        if "\\begin{itemize}" in content_data:
            pattern = r"\\begin\{itemize\}.*?\\end\{itemize\}"
            
            # If the original file contains a list, replace only that segment
            if re.search(pattern, original_content, flags=re.DOTALL):

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


def exp_reorder(file_path, bridge_dict, primary_list, secondary_list=None, pro_exp_sep=True):
    r"""
    Replaces the content between %tech exp anchors in main.tex.
    
    Args:
        file_path (str): Path to main.tex
        bridge_dict (dict): Map of {llm_key: r"\subimport{...}{...}"}
        primary_list (list): 
            - If pro_exp_sep=True: The list of PROJECT keys.
            - If pro_exp_sep=False: The SINGLE MIXED list of all keys.
        secondary_list (list, optional): 
            - If pro_exp_sep=True: The list of EXPERIENCE keys.
            - If pro_exp_sep=False: Ignored (can be None).
        pro_exp_sep (bool): Toggle between split sections vs. single section.
    """
    
    # Build the Primary Block (Always used)
    # This represents 'Projects' in split mode OR 'Everything' in mixed mode
    primary_block = "\n".join([bridge_dict.get(k, f"% Missing Bridge: {k}") for k in primary_list])

    # Logic: Construct the Final LaTeX Block
    if pro_exp_sep:
        # --- Mode A: Two Separate Sections (Projects + Experience) ---
        if not secondary_list:
            print("Warning: Split mode is True but no secondary_list (Experience) was provided.")
            secondary_block = ""
        else:
            secondary_block = "\n".join([bridge_dict.get(k, f"% Missing Bridge: {k}") for k in secondary_list])

        new_content_block = (
            f"\\resumesection{{Projects}}\n"
            f"{primary_block}\n\n"
            f"\\resumesection{{Experience}}\n"
            f"{secondary_block}"
        )
    else:
        # --- Mode B: Single Combined Section (Mixed Order) ---
        # We only use the primary_list, which now contains the mixed/ranked items
        new_content_block = (
            f"\\resumesection{{Technical Experience}}\n"
            f"{primary_block}"
        )

    # Escape backslashes for Regex safety
    safe_content_block = new_content_block.replace("\\", "\\\\")

    # Read the File
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return False


    # Pattern: Matches everything between %tech exp and %tech exp ends
    pattern = r"(%tech exp)(.*?)(%tech exp ends)"
    
    if not re.search(pattern, content, flags=re.DOTALL):
        print(f"Error: Anchors '%tech exp' and '%tech exp ends' not found in {file_path}")
        return False

    new_content = re.sub(
        pattern, 
        rf"\1\n{safe_content_block}\n\3", 
        content, 
        flags=re.DOTALL
    )

    # Write Back
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully reordered resume. (Split Mode: {pro_exp_sep})")
        return True
    except Exception as e:
        print(f"Failed to write file: {e}")
        return False

####################################################################################################################################################
####################################################################################################################################################
##############################################################   Compile Latex File   ##############################################################


def compile_latex(main_file_path, output_dir=None):
    """
    Refined LuaLaTeX compiler that handles complex paths by 
    isolating the working directory.
    """
    #  Expand to absolute path to avoid any ambiguity
    abs_main_path = os.path.abspath(main_file_path)
    
    if not os.path.isfile(abs_main_path):
        print(f"Error: File not found at {abs_main_path}")
        return False

    # Split path into directory and filename
    # This allows us to 'cd' into the folder and run the file by name
    file_dir = os.path.dirname(abs_main_path)
    file_name = os.path.basename(abs_main_path)
    
    target_output = os.path.abspath(output_dir) if output_dir else file_dir

    # Construct the command
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
                print(f"Error: Compilation might have failed during pass!!!{pass_count}")
                # Log the bottom of the stdout to see the specific LaTeX error
                ## For Debugging purpose only
                # log_lines = result.stdout.splitlines()
                # print("\n".join(log_lines[-20:]))
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
        #print(f"Error generating PDF: {e}")
        return False

