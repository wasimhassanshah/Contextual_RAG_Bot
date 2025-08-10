from setuptools import find_packages, setup
from typing import List


def get_requirements() ->List[str]:

    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []

    """
    Write a code to read requirements.txt file and append each requirements in requirement_list variable.
    """
    return requirement_list

setup(
    name = "Contextual_Retrieval_and_Generation",
    version= "0.0.1",
    author="Wasim-Shah",
    author_email="sayedwasimshahkazmi@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)