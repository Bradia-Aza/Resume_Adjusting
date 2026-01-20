import os
import shutil
import re
import pickle
from src.data_loader import load_file, extract_latex_dependencies, build_resume_context, flatten_pydantic , compile_latex
from src.data_loader import rebase_dependency_paths, convert_to_latex, create_dep_map, write_section_content, exp_reorder, convert_str_to_pdf
from src.llm_tools import enrich_file_metadata, extract_jd_features, tailor_profile_and_highlights
from src.llm_tools import rank_experiences, refined_exp_bullets, generate_skills, generate_cover_letter

##Manually saving the component list(cause not all the files are imorted in the main text)

# component_list = [{'section': 'CONTACT_INFORMATION',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/background.tex',
#   'title': 'Contact Information'},
#  {'section': 'PROFILE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/Profile.tex',
#   'title': 'Professional Profile'},
#  {'section': 'HIGHLIGHT OF QUALIFICATIONS',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/Qualifications/QualificationsHighlight.tex',
#   'title': 'Qualifications Summary'},
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/experiences/BSS.tex',
#   'title': 'Computer Vision Engineer - Behyar Sanaat Sepahan'},
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/experiences/Researcher-UI.tex',
#   'title': 'AI Researcher - Univ of Isfahan'},
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/experiences/NT.tex',
#   'title': 'Electronic Engineer - Noavarihaye Tak'},
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/Projects/Housing.tex',
#   'title': 'Machine Learning Engineer - Ottawa Housing Demand Analysis'},
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/Projects/NLP.tex',
#   'title': 'AI Agent Developer - Algonquin College'},
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/Projects/pistachio.tex',
#   'title': 'Machine Learning Engineer - Algonquin College'}, 
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/Projects/incident.tex',
#   'title': 'Machine Learning Engineer - Algonquin College'},
#  {'section': 'TECHNICAL EXPERIENCE',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/Projects/ROS2.tex',
#   'title': 'Robotics Engineer - Algonquin College'},
#  {'section': 'TECHNICAL SKILLS',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/skills.tex',
#   'title': 'Technical Skills'},
#  {'section': 'EDUCATION',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/education/Education.tex',
#   'title': 'AI & Software Developer - Algonquin College'},
#  {'section': 'AWARDS',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/awards/DataDays.tex',
#   'title': 'First place at Sharif DataDays 2022'},
#  {'section': 'AWARDS',
#   'full_path': 'Rsm/main/Resume_Bardia_Azami/components/awards/Torob.tex',
#   'title': 'Third place at Torob Data Challenge 2023'}]

# with open(cache_path, 'wb') as f: 
#     pickle.dump(component_list, f)
#     print("The proccesed dependencies saved")


# Create a mpping function from the llm output keys to the extracted dependencies keys 
LLM_TO_RESUME_BRIDGE = {
    # -- Summaries --
    "profile": ("PROFILE", "Professional Profile"), # Updated to match component_list title
    "highlights": ("HIGHLIGHT OF QUALIFICATIONS", "Qualifications Summary"), 

    # -- Experience (Jobs) --
    "behyar_job": ("TECHNICAL EXPERIENCE", "Computer Vision Engineer - Behyar Sanaat Sepahan"),
    "ui_research": ("TECHNICAL EXPERIENCE", "AI Researcher - Univ of Isfahan"),
    "nt_job": ("TECHNICAL EXPERIENCE", "Electronic Engineer - Noavarihaye Tak"), 

    # -- Projects --
    "housing_project": ("TECHNICAL EXPERIENCE", "Machine Learning Engineer - Ottawa Housing Demand Analysis"),
    "nlp_project": ("TECHNICAL EXPERIENCE", "AI Agent Developer - Algonquin College"), # Updated to match component_list title
    "pistachio_project": ("TECHNICAL EXPERIENCE", "Machine Learning Engineer - Algonquin College"), 
    "incident_project": ("TECHNICAL EXPERIENCE", "Machine Learning Engineer - Algonquin College"), 
    "ros2_project": ("TECHNICAL EXPERIENCE", "Robotics Engineer - Algonquin College"),

    # -- Skills --
    "skills": ("TECHNICAL SKILLS", "Technical Skills")
}


exp_path_bridge = {
    # -- Experiences (Jobs) --
    "behyar_job": r"\subimport{../components/experiences/}{BSS.tex}",
    "ui_research": r"\subimport{../components/experiences/}{Researcher-UI.tex}",
    "nt_job": r"\subimport{../components/experiences/}{NT.tex}",
    # -- Projects --
    "housing_project": r"\subimport{../components/Projects}{Housing.tex}",
    "nlp_project": r"\subimport{../components/Projects}{NLP.tex}",
    "pistachio_project": r"\subimport{../components/Projects}{pistachio.tex}",
    "incident_project": r"\subimport{../components/Projects}{incident.tex}",
    "ros2_project": r"\subimport{../components/Projects}{ROS2.tex}"
}


exp_des_path = {
    "behyar_job": "./Rsm/project_des/behyar_job.yml",
    "nt_job": "./Rsm/project_des/nt_job.yml",
    "ui_research": "./Rsm/project_des/ui_research.yml",
    "housing_project": "./Rsm/project_des/housing_project.yml",
    "nlp_project": "./Rsm/project_des/nlp_project.yml",
    "pistachio_project": "./Rsm/project_des/pistachio_analysis.yml",
    "incident_project": "./Rsm/project_des/incident_analysis.yml",
    "ros2_project": "./Rsm/project_des/ros2_project.yml"
}


# check for the cache file containing all the dependencies and load it if it exists
cache_dir = "./cache"
# check the validity of the cache directory
if not os.path.exists(cache_dir):
    raise FileNotFoundError(f"The directory was not found: ' {cache_dir} ' ")

cache_file_name = "resume_metadata.pkl"
cache_path = os.path.join(cache_dir, cache_file_name)

if os.path.exists(cache_path): 
    dep_list = load_file(cache_path, "pkl")
#else proccess the latex main file for the dependencies list
else: 
    rsm_main_path = "./Rsm/main/Resume_Bardia_Azami/resume-general/Bardia-Azami-Resume.tex"
    #check the validity of file path 
    if not os.path.exists(rsm_main_path): 
        raise FileNotFoundError(f"The directory was not found: ' {rsm_main_path} ' ")
    #unproccesed dependencies list
    dep_unprc = extract_latex_dependencies(rsm_main_path)
    #enrich the dependencies list 
    dep_list = enrich_file_metadata(dep_unprc)
    print("dependencies extracted")
    #save the list 
    with open(cache_path, 'wb') as f: 
        pickle.dump(dep_list, f)
        print("The proccesed dependencies saved")

# Define the directories
jd_dir = "./Rsm/jd"
rsm_dir = "./Rsm/main/Resume_Bardia_Azami"
rsm_tailored_path ="./Rsm/tailored"
for path in [jd_dir, rsm_dir, rsm_tailored_path]:
    if not os.path.exists(path):
        raise FileExistsError(f"could not find the job description file:{jd_dir} ")

# list all .tex files in jd directory
tex_files = [f for f in os.listdir(jd_dir) if f.endswith(".txt")]

# create destination folders for tailored resumes and copy the main resume for making adjusments
for tex_file in tex_files: 
    target_dir = os.path.join( rsm_tailored_path , os.path.splitext(tex_file)[0] )
    os.makedirs(target_dir, exist_ok=True)

    destination = os.path.join(target_dir, os.path.basename(rsm_dir))
    if not os.path.exists(destination):
        shutil.copytree(rsm_dir, destination)
    #destination main file path
    main_path = os.path.join(destination, 'resume-general/Bardia-Azami-Resume.tex')

    #update dep list for the tailored resume path 
    new_dep_list = rebase_dependency_paths(dep_list , destination, rsm_dir)
    dep_map = create_dep_map(new_dep_list)

    #load job description
    jd_path = os.path.join(jd_dir, tex_file)
    jd = load_file(jd_path, "tex")

    #extract key words
    jd_kw = extract_jd_features(jd)

    #make keywords list
    kw_ls = flatten_pydantic(jd_kw , ['domain_knowledge', 'technical_stack', 'tools_and_platforms', 'soft_skills'])
    kw_tech_ls = flatten_pydantic(jd_kw , ['domain_knowledge', 'technical_stack', 'tools_and_platforms'])
    print(f"Number of extracted key words: {len(kw_ls)}")

    # Reorde experiences 
    dep_list_exp = [dep for dep in dep_list if dep['section'] == 'TECHNICAL EXPERIENCE']
    rsm_exp = build_resume_context(dep_list_exp)

    exp_key_list = [key for key, (section, title) in LLM_TO_RESUME_BRIDGE.items() if section == "TECHNICAL EXPERIENCE"]
    exp_ranked = rank_experiences(jd, exp_key_list, rsm_exp)

    #delete least 2 related project
    del exp_ranked[-2:]
    
    exp_reorder(main_path, exp_ranked, exp_path_bridge)

    #Regenerate experience Bullet Points(generate 4 for the top related ones and 3 for rest)
    i = 0
    for job in exp_ranked:
        if i <2:
            num_bullets = 4
        else:
            num_bullets = 3
            
        project_des = load_file(exp_des_path[job], "yml")
        bp = refined_exp_bullets(  project_data= project_des, jd= jd, keywords= kw_ls, num_bullets = num_bullets)
        # Convert tge bullet points to Latex code
        bp_latex = convert_to_latex(bp)
        # Write it in the file
        section, title = LLM_TO_RESUME_BRIDGE[job]
        write_section_content(dep_map, section , title , bp_latex)
        i+=1

    # Generate Skills 
    rsm = build_resume_context(dep_list)  
    skills = generate_skills(jd = jd, keywords= kw_tech_ls, resume_text = rsm)
    skills_latex = convert_to_latex(skills)
    section, title = LLM_TO_RESUME_BRIDGE["skills"]
    write_section_content(dep_map, section , title , skills_latex)

    #Tailor profile and highlight of qualifications
   
    data = tailor_profile_and_highlights(rsm, jd, kw_ls)
    for keys in data.keys():
        latex_data = convert_to_latex( data[keys] )
        section, title = LLM_TO_RESUME_BRIDGE[keys]
        write_section_content(dep_map, section , title , latex_data)

    # Compile Latex
    compile_latex(main_path, destination)

    # Cover Letter
    cover_letter = generate_cover_letter(jd= jd, keywords=kw_ls, resume_text= rsm )
    convert_str_to_pdf(content_str=cover_letter, output_dir= destination)

    # Delete job description file
    os.remove(jd_path)
    
    