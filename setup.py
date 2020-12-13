from setuptools import setup, find_packages


setup(
    name="rnnencdec",
    version="1.0.0",
    author="Hoang Nghia Tuyen",
    author_email="hnt4499@gmail.com",
    url="https://github.com/hnt4499/seq2seq",
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.md"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="A package containing several sequence to sequence models for "
                "machine translation",
    keywords=["deep learning", "natural language processing", "NLP",
              "machine translation"]
)
