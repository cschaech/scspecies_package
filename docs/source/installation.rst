
Installation
============

**scSpecies** requires **Python 3.11** or higher.

Virtual environment
-------------------

It’s best to install scSpecies in an isolated environment:

.. code-block:: bash

    # Using Python’s built-in venv
    python3 -m venv scspecies-env
    source scspecies-env/bin/activate

    # Using conda
    conda create -n scspecies-env python=3.11
    conda activate scspecies-env

Install from PyPI
-----------------

To install the latest stable release of scSpecies run the following command.

.. code-block:: bash

    pip install scspecies

After installation, confirm that scSpecies loads:

.. code-block:: bash

    python -c "import scspecies; print(scspecies.__version__)"


Tutorial notebooks
------------------

Start with running the tutorial notebooks. They can be downloaded from the `GitHub repository <https://github.com/cschaech/scspecies_package/tree/main/docs/source/tutorials>`_.
