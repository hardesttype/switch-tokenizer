from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="switch-tokenizer",
    version="0.1.0",
    author="",
    author_email="",
    description="A multilingual tokenizer implementation using shared 64k vocabulary space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hardesttype/switch-tokenizer",
    project_urls={
        "Bug Tracker": "https://github.com/hardesttype/switch-tokenizer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "switch-tokenizer=src.__main__:main",
        ],
    },
) 