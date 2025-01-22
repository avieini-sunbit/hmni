from setuptools import setup, find_packages

setup(
    name="hmni",
    version="0.1.9",
    author="Christopher Thornton",
    author_email="christopher_thornton@outlook.com",
    description="Fuzzy Name Matching with Machine Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Christopher-Thornton/hmni",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "joblib",
        "unidecode",
        "fuzzywuzzy",
        "editdistance",
        "abydos @ git+https://github.com/IlnarSelimcan/abydos"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 