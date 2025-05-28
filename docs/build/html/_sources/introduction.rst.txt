.. scpecies_package documentation master file, created by
   sphinx-quickstart on Thu May  8 13:35:00 2025.


Introduction
============

The workflow and applications of **scSpecies** are demonstrated in five tutorial notebooks, covering dataset preprocessing, architecture alignment, annotation transfer, differential gene expression analysis, and atlas creation across species.


.. figure:: _static/scSpecies_model_architecture.png
   :alt: Architecture
   :align: left
   :width: 750px

   scSpecies model architecture and workflow.

Key functionalities of scSpecies
--------------------------------

1) **Align and visualize latent representations:**  
   
   scSpecies aligns latent representations of datasets across species. The influence of experimental batch effects on the latent representation is removed internally. 

2) **Compute cell similarity scores:**  
   
   scSpecies defines a similarity metric that establishes a direct correspondence between cells of different datasets and ranks cells by similarity across species. 

3) **Transfer cell annotation between species:**  
   
   Cell type labels or other annotation can be transferred via a neighbor search on the aligned latent space.

4) **Match cell type annotation:**  
   
   scSpecies helps to identify homologous cell types and can match label annotation. 

5) **Differential gene expression analysis:**  
   
   scSpecies aids in identifying differentially expressed genes among biologically similar cells across datasets. 

6) **Aligned cell atlas creation:**  
   
   scSpecies can create an aligned cell atlas that spans multiple species. 

.. figure:: _static/multiple_species.jpeg
   :alt: Atlas
   :align: left
   :width: 700px

   Aligned latent representations of liver cell samples across multiple species.


Requirements for training scSpecies
-----------------------------------

To train **scSpecies**, two scRNA-seq datasets are required:

- A **context dataset** for pre-training.  
- A **target dataset** for fine-tuning.  

Datasets must meet the following criteria:

- Both must contain **raw count data** as a `Compressed Sparse Row` sparse matrix of dtype 'float32' in the `.h5ad` format.
- Both must store gene identifiers as **gene symbols** (e.g., *Sox17*, *ISG15*) in `.var_names`, following species-specific naming conventions.
- Both must include **experimental batch labels** for both datasets in the `obs` layer.
- The **context dataset** must include **cell type annotations** in the `obs` layer.


Preprint
--------

The preprint describing the methodology of **scSpecies** is available at:  

Clemens Schächter, Martin Treppner, Maren Hackenberg, Hanne Raum, Joschka Boedecker, Harald Binder (2024).  
*Enhancement of Network Architecture Alignment in Comparative Single-Cell Studies*. Qeios.  
`doi:10.32388/D37AFF <https://doi.org/10.32388/D37AFF>`_


Funding
--------

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project-ID 499552394 - SFB 1597 Small Data.
For more information, refer to the `Small Data homepage <https://www.smalldata-initiative.de/>`_.
