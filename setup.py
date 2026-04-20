import setuptools

with open("README.md","r",encoding='utf-8') as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "News_NLI"
AUTHOR_USER_NAME = "Data-Science-Maya19"
SRC_REPO = "news_nli"
AUTHOR_EMAIL = "mayura19.zadane@gmail.com" 

setuptools.setup(
    name  = SRC_REPO,
    version=__version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description= "SemEval-2022 Task 8 Cross-Lingual NLI — Capstone Project",
    long_description= long_description,
    long_description_content = "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src")
)
