# scspecies

**scSpecies** is a deep‐learning framework for aligning single‐cell RNA-seq datasets across species.  
Built on top of scVI and transfer-learning principles, it learns a shared embedding space that directly matches cell populations from different organisms.

## Installation

To install the latest stable release of scSpecies run one of the following commands.
scSpecies defines two extras required to run the tutorial notebooks, **plotting** and **notebooks**.

.. code-block:: bash

    pip install scspecies

After installing,  confirm that scSpecies loads:

.. code-block:: bash

    python -c "import scspecies; print(scspecies.__version__)"
    
## Documentation 

Full API docs, tutorials, and examples are available at:
[scSpecies Documentation (Read the Docs)](https://scspecies.readthedocs.io/en/latest/)

## Tutorial Notebooks

Start with running the tutorial notebooks. They can be downloaded from the [GitHub repository](https://github.com/cschaech/scspecies_package/tree/main/docs/source/tutorials).