
Installation
============

**scSpecies** requires **Python 3.10** or higher.

Virtual environment
-------------------

It’s best to install scSpecies in an isolated environment:

.. code-block:: bash

    # Using Python’s built-in venv
    python3 -m venv scspecies-env
    source scspecies-env/bin/activate

    # Or using conda
    conda create -n scspecies-env python=3.10
    conda activate scspecies-env

Install from PyPI
-----------------

To install the latest stable release of scSpecies run one of the following commands.
scSpecies defines two extras required to run the tutorial notebooks, **plotting** and **notebooks**.

.. code-block:: bash

    # Without extras
    pip install scspecies

    # Plotting tools
    pip install scspecies[plotting]

    # Notebook support
    pip install scspecies[notebooks]

    # All extras
    pip install scspecies[plotting,notebooks]


After installation, confirm that scSpecies loads:

.. code-block:: bash

    python -c "import scspecies; print(scspecies.__version__)"


Tutorial
-------------------

Start with the tutorial notebooks:
