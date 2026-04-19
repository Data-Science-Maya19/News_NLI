import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
PROJECT_NAME = "news_nli"

# Complete file structure tree
list_of_files = [
    ".github/workflows/.gitkeep",  # gitkeep if any empty folder it identifies that
    f"src/{PROJECT_NAME}/__init__.py",  # constructor file
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/utils/common.py",
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/entity/config_entity.py",
    f"src/{PROJECT_NAME}/constants/__init__.py",
    f"src/{PROJECT_NAME}/logging/__init__.py",
    "logs/.gitkeep",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",  # for experimentation purpose
    "templates/index.html",
    "README.md",
    ".gitignore",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    # separate folder name and file names
    filedir, filename = os.path.split(filepath)

    # check if folder is not empty
    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"Creating directory {filedir} for the file {filename}")
    
    # Check if the file is present and what if the filesize (like if any content exists in file)
    if (not (os.path.exists(filename)) or (os.path.getsize(filename) == 0)):
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating empty file {filepath}")
    else:
        logging.info(f"The file {filename} already exists")
