from typing import Union, Optional, List
from pathlib import Path
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import muon as mu
from collections import Counter
import io, os, requests, contextlib, torch, random
from mygene import MyGeneInfo
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def map_homologs(
        gene_list: list[str], 
        target_NCBI_Taxon_ID: int, 
        context_NCBI_Taxon_ID: int
    ) -> list[str]:
    
    """
    Maps a list of gene symbols from the target species to their homologous symbols
    of the context species using MyGeneInfo.

    Parameters
    ----------
    gene_list : list[str]
        Gene symbols in the target species to be translated.
    target_NCBI_Taxon_ID : int
        NCBI Taxonomy ID of the target species.
    context_NCBI_Taxon_ID : int
        NCBI Taxonomy ID of the source (context) species.

    Returns
    -------
    list[str]
        Homologous gene symbols in the target species, with 'non_hom_<i>' for non homologous genes.
    """

    mg = MyGeneInfo()
    
    results = mg.querymany(
        gene_list,
        scopes='symbol',
        species=target_NCBI_Taxon_ID,
        fields='homologene',
        as_dataframe=False
    )

    homolog_ids = {}       
    ids_to_lookup = set()  
    for res in results:
        src = res['query']
        if res.get('notfound') or 'homologene' not in res:
            homolog_ids[src] = []
        else:
            hits = [g[1] for g in res['homologene']['genes'] if g[0] == context_NCBI_Taxon_ID]
            homolog_ids[src] = hits
            ids_to_lookup.update(hits)

    id_to_symbol = {}
    if ids_to_lookup:
        lookup = mg.querymany(
            list(ids_to_lookup),
            scopes='entrezgene',
            fields='symbol',
            species=context_NCBI_Taxon_ID,
            as_dataframe=False
        )
        
        for hit in lookup:
            try:
                eid = int(hit['query'])
                if 'symbol' in hit:
                    id_to_symbol[eid] = hit['symbol']
            except (KeyError, ValueError):
                continue

    mapped = []

    for i,g in enumerate(gene_list):
        syms = [id_to_symbol[eid] for eid in homolog_ids.get(g, []) if eid in id_to_symbol]

        if not syms:
            mapped.append('non_hom_'+str(i))

        else:
            mapped.append(syms[0])

    return mapped

def map_homologs_silent(
        gene_list: list[str], 
        target_NCBI_Taxon_ID: int, 
        context_NCBI_Taxon_ID: int
    ) -> list[str]:
    """
    Same as `map_homologs` but suppresses all console output 
    as map_homologs outputs a print statement for each gene.

    Parameters
    ----------
    gene_list : list[str]
        Gene symbols in the target species to be translated.
    target_NCBI_Taxon_ID : int
        NCBI Taxonomy ID of the target species.
    context_NCBI_Taxon_ID : int
        NCBI Taxonomy ID of the source (context) species.

    Returns
    -------
    list[str]
        Homologous gene symbols in the target species, with 'non_hom_<i>' for non homologous genes.
    """

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = map_homologs(gene_list, target_NCBI_Taxon_ID, context_NCBI_Taxon_ID)

    return result

def get_key(
        gene, 
        homology_targsp_df, 
        homology_context_df, 
        i
    ) -> str: 
    
    """
    Retrieve the homologous context gene symbol for a given target gene using homology tables from
    informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt
    Can only be used for mouse, rat, human, zebrafish context-target dataset pairs,

    Parameters
    ----------
    gene : str
        Gene symbol in the ‘from’ DataFrame.
    homology_targsp_df : pandas.DataFrame
        Homology table for the target species (columns include 'Symbol' and 'DB Class Key').
    homology_context_df : pandas.DataFrame
        Homology table for the context species (same key column).
    i : int
        Index of the gene in the original list, used to name unmapped genes.

    Returns
    -------
    str
        Context‐species gene symbol if found, otherwise 'non_hom_<i>'.
    """

    targ_gene_names = 'non_hom_'+str(i)

    if gene in homology_targsp_df['Symbol'].unique():
        key = homology_targsp_df[homology_targsp_df['Symbol'] == gene]['DB Class Key'].values[0] 

        if key in homology_context_df['DB Class Key'].unique():
            targ_gene_names = homology_context_df[homology_context_df['DB Class Key'] == key]['Symbol'].values[0]

    return targ_gene_names      


def download_datasets():
    """
    Download liver cell .h5ad datasets into ./data directory.
    Downloads each file and skips files already present.
    
    Raises
    ------
    requests.HTTPError
        If any of the dataset URLs returns a bad status.
    """
    data_urls = {
        "human_liver.h5ad":   "https://zenodo.org/records/15522251/files/human_liver.h5ad?download=1",
        "mouse_liver.h5ad":   "https://zenodo.org/records/15522251/files/mouse_liver.h5ad?download=1",
        "hamster_liver.h5ad": "https://zenodo.org/records/15522251/files/hamster_liver.h5ad?download=1",
    }

    data_path = Path("data")
    data_path.mkdir(parents=True, exist_ok=True)

    for fname, url in data_urls.items():
        out_file = data_path / fname
        if out_file.exists():
            print(f"{fname} already exists. Skipping download.")
            continue

        print(f"Downloading {fname} …")
        resp = requests.get(url)  # no stream=True
        resp.raise_for_status()

        with open(out_file, "wb") as f:
            f.write(resp.content)

        size_mb = out_file.stat().st_size / 1024 / 1024
        print(f"{fname} downloaded. Size: {size_mb:.2f} MB")

    print("All datasets have been downloaded to the ./data directory.")

def set_random_seed(
        seed: int
        ):
    
    """
    Fix all relevant RNG seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use for Python, NumPy, random, and PyTorch.
    """


    os.environ["PYTHONHASHSEED"] = str(seed) 
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

class create_mdata():
    """
    Builder for MuData container that is used by scSpecies to align context & target AnnData datasets.

    Handles downloading a gene-translation table from the mouse to human genome, 
    preprocessing a “context” AnnData, and “target” AnnData from potentially multiple species, and saving the
    final MuData object.
    """

    def __init__(
            self,        
            adata: ad.AnnData,
            batch_key: str,
            cell_key: str,      
            dataset_name: str = 'mouse',
            NCBI_Taxon_ID: int = 10090,
            n_top_genes: Union[int, None] = None,                   
            min_non_zero_genes: float = 0.025, 
            min_cell_type_size: int = 20,     
            min_batch_size: int = 20,                
            ): 

        """
        Initialize and preprocess the context dataset.

        Steps:

        1. Onehot-encode experimental batchs.

        2. Calculate library size encoder prior parameters for scVI

        3. Subset to top HVGs and filter out cells with low expression patterns as well as rare cell types and batches (optionally).

        Parameters
        ----------
        adata : ad.AnnData
            AnnData used as a context in scSpecies.
        batch_key : str
            Observation key for experimental batch labels.
        cell_key : str
            Observation key for cell-type annotation.
        dataset_name : str, optional
            Tag for the context dataset (default 'mouse').
        NCBI_Taxon_ID : int, optional
            Taxonomy ID of the context species (default mouse - 10090).
        n_top_genes : int or None, optional
            Number of HVGs to retain (None to skip) (default None).
        min_non_zero_genes : float, optional
            Min fraction of nonzero genes per cell (default 0.025).
        min_cell_type_size : int, optional
            Min cells per cell-type, cell types with fewer samples are removed (default 20).
        min_batch_size : int, optional
            Min cells per batch for encoding, batch with fewer samples are removed (default 20).

        Effects
        -------
        - Ensures a `data/` directory exists.
        - Annotates `adata.uns['metadata']` with context dataset info.
        - One-hot encodes batch labels, dropping any batches smaller than `min_batch_size`.
        - Computes per-batch library size prior parameters.
        - Subsets to top highly variable genes if `n_top_genes` is not None.
        - Filters out cells with low gene detection and rare cell-types.
        - Stores the processed AnnData in `self.dataset_collection`.
        """        
        adata = adata.copy()

        mu.set_options(pull_on_update=False)

        self.min_non_zero_genes = min_non_zero_genes
        self.min_cell_type_size = min_cell_type_size  
        self.min_batch_size = min_batch_size   
        
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.context_dataset_name = dataset_name
        self.context_cell_key = cell_key
        self.context_NCBI_Taxon_ID = NCBI_Taxon_ID
        
        if adata.isbacked:
            adata = adata.copy()

        adata.uns['metadata'] = {
            'name': dataset_name,            
            'batch_key': batch_key, 
            'cell_key': cell_key,
            'NCBI_Taxon_ID': NCBI_Taxon_ID,
            'function': 'context',            
        }

        adata.obs['dataset'] = dataset_name
        adata.obs.index = adata.obs.index.astype(str) + f"_{dataset_name}"        

        adata = self.encode_batch_labels(adata, self.min_batch_size)
        adata = self.compute_lib_prior_params(adata)     
        
        if n_top_genes != None:
            adata = self.subset_to_hvg(adata, n_top_genes)

        adata = self.filter_cells(adata, self.min_non_zero_genes, self.min_cell_type_size)

        self.dataset_collection = {dataset_name: adata}
        
        adata.obs_names_make_unique()
        adata.X = csr_matrix(adata.X)
        print('Done!\n'+'-'*90)

    def setup_target_adata(self,
            adata: ad.AnnData,
            batch_key: str,
            cell_key: Union[str, None] = None,  
            eval_nns_keys: Union[List[str], None] = None,
            dataset_name: str = 'human',
            NCBI_Taxon_ID: int = 9606,
            n_top_genes: Union[int, None] = None,               
            compute_log1p: bool = True,                   
            nn_kwargs: Optional[dict] = None,     
            ):   

        """
        Preprocess and align a target AnnData against the context.

        Steps:

        1. Onehot-encode experimental batchs.

        2. Calculate library size encoder prior parameters for scVI

        3. Subset to top HVGs and filter out cells with low expression patterns as well as rare cell types and batches (optionally). 

        4. Translate target gene symbols to context homologs.

        5. Compute and evaluate data-level nearest neighbors on the shared homologous gene set.

        Parameters
        ----------
        adata : ad.AnnData
            Target dataset.
        batch_key : str
            Observation key for experimental batch labels.
        cell_key : str or None
            Observation key for cell types (None if unkown).
        eval_nns_keys : List of str or None
            List of context dataset `obs` keys that should be transferred by scSpecies. Defaults to [cell_key]. 
        dataset_name : str, optional
            Defaults to 'human'.
        NCBI_Taxon_ID : int, optional
            Taxonomy ID for the target species (default human - 9606).
        n_top_genes : int or None, optional
            Number of HVGs to keep (None to skip) (default None).
        compute_log1p : bool, optional
            Use log1p counts for neighbor search if True (default True).
        nn_kwargs : dict, optional
            Args for sklearn.neighbors.NearestNeighbors.
            Defaults to `{'n_neighbors': 250, 'metric': 'cosine'}`.       

        Effects
        -------
        - Updates `adata.uns['metadata']` with target dataset info.
        - Filters and one-hot encodes batch (and cell-type, if provided).
        - Computes library size prior parameters.
        - Calls `translate_gene_list` to add translated gene symbols in the context genome to `var_names_transl`.
        - Subsets to HVGs if `n_top_genes` is not None.
        - Filters out low-coverage cells and rare cell-types.
        - Identifies intersecting homologous genes with the context and performs a nearest-neighbor search on log1p (or raw) counts.
        - Stores neighbor indices in `adata.obsm['ind_neigh_nns']`.
        - Calculates the percentage of neighbor label agreement and transfers labels based on the data-level nearest neighbor search.
        - Inserts the processed AnnData into `self.dataset_collection`.        
        """

        adata = adata.copy()

        adata.uns['metadata'] = {
            'name': dataset_name,            
            'batch_key': batch_key,
            'cell_key': cell_key, 
            'NCBI_Taxon_ID': NCBI_Taxon_ID,
            'function': 'target',                  
        }

        if cell_key == None:   
            adata.uns['metadata']['cell_key'] = 'unknown'

        adata.obs['dataset'] = dataset_name
        adata.obs.index = adata.obs.index.astype(str) + f"_{dataset_name}"        

        adata = self.encode_batch_labels(adata, self.min_batch_size)
        adata = self.compute_lib_prior_params(adata)           

        if n_top_genes != None:
            adata = self.subset_to_hvg(adata, n_top_genes)

        adata = self.filter_cells(adata, self.min_non_zero_genes, self.min_cell_type_size)

        adata = self.translate_gene_list(adata)
        _, context_ind, target_ind = np.intersect1d(self.dataset_collection[self.context_dataset_name].var_names.to_numpy(), adata.var['var_names_transl'], return_indices=True)
        
        if nn_kwargs is None:
            nn_kwargs = {}
        if "n_neighbors" not in nn_kwargs:
            nn_kwargs["n_neighbors"] = 250
        if "metric" not in nn_kwargs:
            nn_kwargs["metric"] = "cosine"

        if len(context_ind) == 0:
            raise ValueError("No homologous genes found. scSpecies cannot be used.")
        elif len(context_ind) < 250:
            raise Warning("Only \033[35m{}\033[0m homologous genes found. Data-level neighbor search may yield noisy results.".format(str(len(context_ind))))      
        else:
            print("Found \033[35m{}\033[0m shared homologous genes between context and target dataset".format(str(len(context_ind))))

        print('Perform the data-level nearest neigbor search on the homologous gene set.')    

        if compute_log1p:
            context_neigh = np.log1p(self.dataset_collection[self.context_dataset_name].X.toarray()[:, context_ind])
            target_neigh = np.log1p(adata.X.toarray()[:, target_ind])

        else:    
            context_neigh = self.dataset_collection[self.context_dataset_name].X.toarray()[:, context_ind]
            target_neigh = adata.X.toarray()[:, target_ind]

        neigh = NearestNeighbors(**nn_kwargs)
        neigh.fit(context_neigh)

        _, indices_whole = neigh.kneighbors(target_neigh)
        adata.obsm['ind_neigh_nns'] = np.squeeze(indices_whole).astype(np.int32)

        if eval_nns_keys == None:
            eval_nns_keys = [self.context_cell_key]
        adata = self.pred_labels_nns_hom_genes(adata, eval_nns_keys)

        self.dataset_collection[dataset_name] = adata
        adata.X = csr_matrix(adata.X)
        print('Done!\n'+'-'*90)

    def translate_gene_list(
            self, 
            adata: ad.AnnData
        ) -> ad.AnnData:
        
        """
        Translate gene symbols in var_names of a target AnnData to homologous context-species symbols.

        Will download a HOM_AllOrganism.rpt if not present if context-target species pair consits of 
        human, mouse, rat or zebrafish. Will fallback to `map_homologs_silent` for unsupported species pairs.

        Parameters
        ----------
        adata : anndata.AnnData
            Target AnnData whose var_names will be translated.

        Effects
        -------
        - Prints a status message about which datasets are being translated.
        - Downloads and saves `HOM_AllOrganism.rpt` if not already present.
        - Reads the homology report into a DataFrame.
        - Filters the table to context and target species.
        - Computes a translated gene list via `get_key` or falls back to `map_homologs_silent` if species is not human, mouse, rat or zebrafish.
        - Sets `adata.var['var_names_transl']` to the mapped names.            
        """

        print('Translating homologous gene names between {} context and {} target dataset.'.format(self.context_dataset_name, adata.uns['metadata']['name']))
        gene_list = adata.var_names
        NCBI_Taxon_ID = adata.uns['metadata']['NCBI_Taxon_ID']

        if self.context_NCBI_Taxon_ID == NCBI_Taxon_ID:
            transl_gene_list = gene_list

        elif self.context_NCBI_Taxon_ID in (9606, 10090, 10116, 7955) and NCBI_Taxon_ID in (9606, 10090, 10116, 7955): #False:# s
            out_dir = Path("data")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "HOM_AllOrganism.rpt"

            if not out_file.exists():
                print(f"Downloading gene translation dictionary.")
                homology_df = pd.read_csv("https://www.informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt", sep="\t")
                homology_df.to_csv(out_file, sep="\t", index=False)
                print('Gene translation dictionary saved to data/HOM_AllOrganism.rpt'.format(out_file))

            else:    
                homology_df = pd.read_csv(out_file, sep="\t")

            homology_contsp_df = homology_df[homology_df['NCBI Taxon ID'] == self.context_NCBI_Taxon_ID]
            homology_targsp_df = homology_df[homology_df['NCBI Taxon ID'] == NCBI_Taxon_ID]
            transl_gene_list = [get_key(gene, homology_targsp_df, homology_contsp_df, i) for i,gene in enumerate(gene_list)]

        else:
            transl_gene_list = map_homologs_silent(list(gene_list), NCBI_Taxon_ID, self.context_NCBI_Taxon_ID)


        num_hom_genes = len([gene for gene in transl_gene_list if 'non_hom' not in gene])
        print('Could map \033[33m{}\033[0m of {} target gene symbols to context species gene symbols'.format(str(num_hom_genes), str(len(transl_gene_list))))    

        adata.var['var_names_transl'] = transl_gene_list
        return adata


    @staticmethod
    def filter_cells(
            adata: ad.AnnData, 
            min_non_zero_genes: float, 
            min_cell_type_size: int
        ) -> ad.AnnData:
        
        """
        Filter cells based on minimum non-zero gene fraction and cell‐type size.

        Parameters
        ----------
        adata : anndata.AnnData
            The annotated data matrix to filter.
        min_non_zero_genes : float
            Minimum fraction of genes that must have nonzero counts in a cell.
        min_cell_type_size : int
            Minimum number of cells required to retain any given cell‐type.

        Effects
        -------
        - Removes cells with fewer than `min_non_zero_genes * n_vars` detected genes.
        - If a cell‐type key is set in `adata.uns['metadata']['cell_key']`, discards
        any cell‐types with fewer than `min_cell_type_size` cells.
        """

        old_n_obs = adata.n_obs
        cell_key = adata.uns['metadata']['cell_key']        
        sc.pp.filter_cells(adata, min_genes=adata.n_vars*min_non_zero_genes)

        if cell_key != 'unknown':
            cell_type_counts = adata.obs[cell_key].value_counts()>min_cell_type_size
            cell_type_counts = cell_type_counts[cell_type_counts==True].index
            adata = adata[adata.obs[cell_key].isin(cell_type_counts)]    

        print('Filtering cells. Kept {}, removed {}.'.format(str(adata.n_obs), str(int(old_n_obs-adata.n_obs))))
        return adata        

    @staticmethod
    def compute_lib_prior_params(
            adata: ad.AnnData
        ) -> ad.AnnData:
        
        """
        Compute scVI library size prior parameters for each cell.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix with raw counts in `adata.X`.

        Effects
        -------
        - Within each batch (from `adata.uns['metadata']['batch_key']`),
        calculates the mean and standard deviation of log-total counts.
        - Stores values in `adata.obs['library_log_mean']` and
        `adata.obs['library_log_std']` as float32 columns.
        """

        print('Compute prior parameters for the library encoder.')
        batch_key = adata.uns['metadata']['batch_key'] 

        library_log_mean = np.zeros(shape=(adata.n_obs, 1))
        library_log_std = np.ones(shape=(adata.n_obs, 1))  
        log_sum = np.log(adata.X.sum(axis=1))

        for batch in np.unique(adata.obs[batch_key]):
            ind = np.where(adata.obs[batch_key] == batch)[0]
            library_log_mean[ind]  = np.mean(log_sum[ind])
            library_log_std[ind] = np.std(log_sum[ind])   

        adata.obs['library_log_mean'] = library_log_mean.astype(np.float32) 
        adata.obs['library_log_std'] = library_log_std.astype(np.float32) 
        return adata


    def pred_labels_nns_hom_genes(
            self, 
            adata: ad.AnnData, 
            context_label_keys: List[str] = None,
            k: int = 25,
        ) -> ad.AnnData:
        
        """
        Predicts target cell-type labels using data-level k-nearest neighbor search 
        results over homologous genes shared with the context dataset.
        Additionaly calculates the uncertainty score that will be used by scSpecies to decide
        which cells are aligned during fine-tuning.

        Parameters
        ----------
        adata : anndata.AnnData
            Target dataset that contains the neighbor indices in `adata.obsm['ind_neigh_nns']`.
        context_label_keys : list of str
            Keys in the context dataset's `obs` corresponding to categorical labels to be transferred 
            (e.g., cell-type, tissue-type). 
        k : int
            Amount of neighbort to consider for majority voting

        Effects
        -------
        - For each key in `context_label_keys`, assigns:
            - `adata.obs['pred_nns_<label_key>']`: predicted label (most frequent among neighbors).
            - `adata.obs['top_percent_<label_key>']`: confidence score based on relative neighbor rank.
        """

        context_adata = self.dataset_collection[self.context_dataset_name]

        for context_label_key in context_label_keys:
            print('Evaluating data level NNS and calculating cells with the highest agreement for context labels key {}.'.format(context_label_key))

            ind_neigh_topk = adata.obsm['ind_neigh_nns'][:,:k]
            candidate_labels = context_adata.obs[context_label_key].to_numpy()

            label_counts = [dict(Counter(candidate_labels[ind_neigh_topk[i]])) for i in range(adata.n_obs)]
            label_counts = [max(label_counts[i].items(), key=lambda x: x[1]) + (i, ) for i in range(adata.n_obs)]

            top_dict = {c: [] for c in np.unique(candidate_labels)}
            for i in range(len(label_counts)):
                top_dict[label_counts[i][0]] += [label_counts[i]]

            for key in top_dict.keys():
                top_dict[key] = sorted(top_dict[key], key=lambda x: x[1])
                num_samples = len(top_dict[key])
                top_dict[key] = [top_dict[key][i]+(1-(i+1)/num_samples,) for i in range(len(top_dict[key]))] 

            label_counts = sorted([item for sublist in top_dict.values() for item in sublist], key=lambda x: x[-2]) 

            adata.obs['top_percent_'+context_label_key] = np.array([label_counts[i][-1] for i in range(len(label_counts))])
            adata.obs['pred_nns_'+context_label_key] = np.array([label_counts[i][0] for i in range(len(label_counts))])

        return adata    


    @staticmethod
    def encode_batch_labels(
            adata: ad.AnnData, 
            min_batch_size: Union[int, None] = None
        ) -> ad.AnnData:    
         
        """
        One‐hot encode experimental batch labels, excluding small batches.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix with batch labels in `adata.obs[...]`.
        min_batch_size : int
            Smallest batch size to keep; batches with fewer cells are removed, must be >= 0.

        Effects
        -------
        - Drops any batch categories with fewer than `min_batch_size` cells.
        - Fits a OneHotEncoder to remaining batch labels.
        - Saves the encoded batch matrix to `adata.obsm['batch_label_enc']`.
        - Builds `adata.uns[batch_dict]`, mapping each cell‐type (and 'unknown')
        to batch labels in which they have samples.
        """
        if min_batch_size == None:
            min_batch_size = 0

        batch_key = adata.uns['metadata']['batch_key'] 
        cell_key = adata.uns['metadata']['cell_key'] 
        name = adata.uns['metadata']['name'] 

        batch_counts = adata.obs[batch_key].value_counts()
        to_remove   = batch_counts[batch_counts < min_batch_size].index
        adata       = adata[~adata.obs[batch_key].isin(to_remove)]
        batch_labels = adata.obs[batch_key].to_numpy().reshape(-1, 1)
        
        print('Registering experimental batches for the {} dataset. Kept {}, removed {}.'.format(
            name, str(len(np.unique(batch_labels))), str(len(batch_counts))))

        enc = OneHotEncoder()
        enc.fit(batch_labels)

        adata.obsm['batch_label_enc'] = enc.transform(batch_labels).toarray().astype(np.float32) 

        if cell_key == 'unknown':
            batch_dict = {'unknown': enc.transform(np.unique(batch_labels).reshape(-1, 1)).toarray().astype(np.float32)}
        else:
            cell_types = adata.obs[cell_key].cat.categories.to_numpy()
            batch_dict = {c: adata[adata.obs[cell_key] == c].obs[batch_key].value_counts() > 3 for c in cell_types}
            batch_dict = {c : batch_dict[c][batch_dict[c]].index.to_numpy() for c in cell_types}
            batch_dict = {c : enc.transform(batch_dict[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in cell_types}
            batch_dict['unknown'] = enc.transform(np.unique(batch_labels).reshape(-1, 1)).toarray().astype(np.float32)

        adata.uns['batch_dict'] = batch_dict

        return adata


    @staticmethod
    def subset_to_hvg(
            adata: ad.AnnData,                                       
            n_top_genes: int,
        ) -> ad.AnnData:

        """
        Subset dataset to the top highly variable genes using the Seurat method.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix to subset.
        n_top_genes : int
            Number of top highly variable genes to select.

        Effects
        -------
        - Subsets `adata` to the top `n_top_genes` hvg genes.
        """        
        print('Subsetting the {} dataset to the {} most highly variable genes using seurat.'.format(adata.uns['metadata']['name'], str(n_top_genes)))

        batch_key = adata.uns['metadata']['batch_key'] 

        adata.layers["raw_counts"] = adata.X.copy() 
        sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(
            adata,
            batch_key=batch_key,
            n_top_genes=n_top_genes,
            subset=True,
            flavor='seurat_v3',
        )

        adata.X = adata.layers['raw_counts'].copy()
        del adata.layers['raw_counts']        

        return adata

    def return_mdata(self, 
            return_mdata: bool = True, 
            save: bool = True, 
            save_path: Path = Path("data"), 
            save_name: str = 'mudata'
        ) -> mu.MuData:
        
        """
        Optionally save and/or return the assembled MuData object.

        Parameters
        ----------
        return_mdata : bool, optional
            If True, return the MuData object at the end (default True).
        save : bool, optional
            If True, write the MuData object to disk (default True).
        save_path : pathlib.Path, optional
            Directory in which to save the file; created if missing (default Path("data")).
        save_name : str, optional
            Filename stem for the .h5mu file; '.h5mu' is appended (default 'mudata').

        Effects
        -------
        - If `save` is True:
            - Ensures that `save_path` exists, creating it if necessary.
            - Writes the MuData assembled from `self.dataset_collection` to
            `save_path/<save_name>.h5mu`.
            - Prints messages about directory creation and file saving.
        - If `return_mdata` is True:
            - Returns the MuData object constructed from `self.dataset_collection`.
        """

        if save:
            save_path = Path(save_path)
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"\nCreated directory '{save_path}'.")

            mdata = mu.MuData(self.dataset_collection)
            file_path = save_path / f"{save_name}.h5mu" 

            mdata.write(str(file_path))
            print(f"Saved mdata to {file_path}.")
        if return_mdata:
            return mdata