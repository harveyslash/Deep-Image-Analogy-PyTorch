Deep-Image-Analogy
==============================

Unofficial,PyTorch version of Deep Image Analogy.https://arxiv.org/abs/1705.01088. This project focuses on documentation of the project , and simplifying the structure. A blog post on it is coming soon. 

Project Organization
------------
    ├── data
    │   ├── outputs <-- folder to store outputs
    │   └── raw <-- folder to store inputs
    ├── LICENSE.md
    ├── notebooks
    │   ├── Deep Image Analogy.ipynb Full Pipeline in a step by step manner
    │   ├── PatchMatch-Demo.ipynb Raw Patchmatch demo
    │   └── WLS.ipynb Weighted Least Squares Implementation Demo (currently not being used by this project)
    ├── README.md 
    ├── requirements.txt <-- Project requirements. 
    └── src
        ├── Deep-Img-Analogy.py <-- End to end executable with command line interface.
        ├── __init__.py
        ├── models
        │   ├── __init__.py
        │   └── VGG19.py <-- modified VGG19 with support for deconvolution, and other things. 
        ├── PatchMatch
        │   ├── __init__.py
        │   └── PatchMatchOrig.py <-- CPU version of PatchMatch. GPU version may come in the future.
        ├── Utils.py <-- Helper Utilities
        └── WLS.py <-- Weighted Least Squares.
 

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
