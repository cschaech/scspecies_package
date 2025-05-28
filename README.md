# scspecies

scSpecies is a deep learning model designed to align network architectures for single-cell RNA sequencing datasets across multiple species. 
The model builds on scVI and transfer learning to establish a direct correspondence between cells of different datasets.

## Install from PyPI

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