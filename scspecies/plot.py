import glasbey
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import muon as mu
import os
from typing import Union, List, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import anndata as ad
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import scanpy as sc


def progressive_moving_average(
        y: np.ndarray,
        max_window: int = 6000
    ) -> np.ndarray:

    """
    Compute a moving average over a 1D array with a window
    that grows linearly (capped by max_window) to smooth early iterations 
    more strongly and later ones less.
    Helper for `plot_prototype_sim_history`.

    Parameters
    ----------
    y : np.ndarray
        Input 1D array of values (e.g., losses or metrics over iterations).
    max_window : int, default=6000
        Maximum size of the moving window.

    Returns
    -------
    np.ndarray
        Smoothed values of the same shape as `y`.
    """

    n = len(y)
    return np.array([
        np.mean(y[max(0, k - min(int(5 + min(k, 5) + k * 0.1), max_window) + 1): k + 1])
        for k in range(n)
    ])


def plot_lfc(
        lfc_dict: pd.DataFrame,    
        prob_delta: float = 0.9,
        save_key: Optional[str] = None,         
    ):

    """
    Scatter-plot Log2-Fold-change versus probability for each cell type,
    highlighting and annotating top up- and down-regulated genes.

    Parameters
    ----------
    lfc_dict : list
        List of LFC dataframes.
    prob_delta : float, default=0.9
        Probability threshold for calling significant LFC.
    save_key : str or None, default=None
        If a string, the plot will be saved to `figures/{save_key}.png`. If None, it will only be displayed.   
    """

    cell_types = list([key for key in lfc_dict.keys() if key != 'lfc_delta'])
    df_lfc = pd.DataFrame({ct: lfc_dict[ct]['lfc'] for ct in cell_types})
    df_prob = pd.DataFrame({ct: lfc_dict[ct]['p'] for ct in cell_types})
    lfc_delta = lfc_dict['lfc_delta']
    cell_types = df_prob.columns

    n_cell_types = len(cell_types)
    n_cols = 4  
    n_rows = int(np.ceil(n_cell_types / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)
    axs = axs.flatten()

    ge_one = []
    ge_prob = []
    up_reg = []
    down_reg = []


    for i, cell in enumerate(cell_types):
        greater_than_one = np.round(((df_lfc[cell].abs() > lfc_delta).mean()*100), 1)
        greater_than_one_prob = np.round(((df_lfc[cell].abs() >lfc_delta) & (df_prob[cell].abs() > prob_delta)).mean()*100, 1)
        up = np.round(((df_lfc[cell] > lfc_delta) & (df_prob[cell].abs() > prob_delta)).mean()*100, 1)
        down = np.round(((df_lfc[cell] < -lfc_delta) & (df_prob[cell].abs() > prob_delta)).mean()*100, 1)
        
        ge_one.append(greater_than_one)
        ge_prob.append(greater_than_one_prob)
        up_reg.append(up)
        down_reg.append(down)
        
        ax = axs[i]

        colors = []
        for l, p in zip(df_lfc[cell], df_prob[cell]):
            if abs(l) <= 1:
                colors.append('grey')
            else:
                if l > 1:
                    colors.append('red' if p > prob_delta else 'lightcoral')
                elif l < -1:
                    colors.append('blue' if p > prob_delta else 'lightblue')
        
        ax.scatter(df_lfc[cell], df_prob[cell], c=colors, s=12, edgecolor='k')
        ax.set_xlabel('Log Fold Change')
        ax.set_ylabel('Probability')
        ax.set_title(f"{cell} |LFC|>1, with p>0.9: {greater_than_one_prob}", pad=10)

        ax.axhline(prob_delta, color='black', linestyle='--', linewidth=1.2)
        ax.axvline(-1, color='black', linestyle='--', linewidth=1.2)
        ax.axvline(1, color='black', linestyle='--', linewidth=1.2)

        up_subset = df_lfc[cell][(df_lfc[cell] > lfc_delta) & (df_prob[cell].abs() > prob_delta)]
        down_subset = df_lfc[cell][(df_lfc[cell] < -lfc_delta) & (df_prob[cell].abs() > prob_delta)]
        
        top_up = up_subset.sort_values(ascending=False).head(5)
        top_down = down_subset.sort_values(ascending=True).head(5)

        up_text = "Upregulated:\n" + "\n".join([f"{j+1}. {gene}" for j, gene in enumerate(top_up.index)])
        down_text = "Downregulated:\n" + "\n".join([f"{j+1}. {gene}" for j, gene in enumerate(top_down.index)])

        ax.text(0.025, 0.54, up_text, transform=ax.transAxes, verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle="round", alpha=0.3, facecolor="white"))
        ax.text(0.98, 0.54, down_text, transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.3, facecolor="white"))

    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f'Differential gene expression analysis on the normalized decoder parameter space\n' 
                f'Median |LFC|>1: {np.round(np.mean(ge_one),1)}%, with p>0.9: {np.round(np.mean(ge_prob),1)}. Up regulated: {np.round(np.mean(up_reg),1)}%, down regulated: {np.round(np.mean(down_reg),1)}%.\n', y=0.99, fontsize=16)

    if save_key:
        os.makedirs("figures", exist_ok=True)
        out_path = os.path.join("figures", f"{save_key}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_prototype_sim_heatmap(
        df: pd.DataFrame,
        save_key: Optional[str] = None,         
    ):

    """
    Heatmap of prototype-similarity between target (rows) and context (columns)
    cell types, with top-2 matches annotated by rank.

    Parameters
    ----------
    df : pd.DataFrame
        Similarity matrix (target cell types × context cell types).
    save_key : str or None, default=None
        If a string, the plot will be saved to `figures/{save_key}.png`. If None, it will only be displayed.   
    """

    cell_types_context = np.array(df.columns)
    cell_types_target = np.array(df.index)

    cell_types_context_reordered = np.concat((np.intersect1d(cell_types_context, cell_types_target), np.setdiff1d(cell_types_context, cell_types_target)))
    cell_types_target_reordered = np.concat((np.intersect1d(cell_types_context, cell_types_target), np.setdiff1d(cell_types_target, cell_types_context)))

    df = df.loc[cell_types_target_reordered, cell_types_context_reordered]

    scale = False
    if df.values.min() < 1000 or df.values.max() > 1000:
        scale = True
        df = -(-df.clip(upper=2000, lower=-2000))**0.5

    cols_liver = ['Central Vein ECs', 'LSECs', 'Portal Vein ECs', 'Lymphatic ECs', 'Cholangiocytes', 'Hepatocytes', 'Mesothelial Cells', 'Capsule Fibroblasts', 'Fibroblast 1', 'Fibroblast 2', 'Stellate Cells', 'B Cells', 'CD8 Eff. Memory T', 'Cytotoxic CD8+', 'Naive CD8+ T', 'ILCs', 'NK Cells', 'NKT Cells', 'Naive CD4+ T', 'Regulatory T', 'Th 1', 'Th 17', 'Mig. DCs', 'cDCs 1', 'cDCs 2', 'pDCs', 'Basophils', 'Neutrophils',  'MoMac1', 'MoMac2', 'Peritoneal Macs', 'KCs', 'Monocytes', 'Pat. Monocytes', 'Trans. Monocytes', 'Trans. Monocytes 2']
    index_liver = ['Central Vein ECs', 'LSECs', 'Portal Vein ECs', 'Cholangiocytes', 'Hepatocytes', 'Fibroblasts', 'Stellate Cells', 'Plasma', 'B Cells', 'Circ. Eff. Memory T', 'Cytotoxic CD8+', 'RM CD8+ T cells', 'Circ. NK', 'NKT Cells', 'Tissue Resident NK', 'Gamma-Delta T', 'CD4+ KLRB1 Th', 'Naive/CM CD4+ T', 'Regulatory T', 'Mig. DCs', 'cDCs 1', 'cDCs 2', 'pDCs', 'Basophils', 'Neutrophils', 'MoMac1', 'immLAMs', 'matLAMs', 'KCs', 'Monocytes', 'Pat. Monocytes', 'Pre-moKCs and moKCs']

    if set(index_liver) == set(df.index) and set(cols_liver) == set(df.columns):
        df = df.reindex(index=index_liver, columns=cols_liver)

    top_2_indices = df.apply(lambda row: row.nlargest(2).index, axis=1)
    col_positions = [[df.columns.get_loc(col) for col in row] for row in top_2_indices]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(df, 
                annot=False, 
                cmap=mpl.colormaps.get_cmap("terrain").reversed(),
                yticklabels=df.index, 
                xticklabels=df.columns,
                ax=ax)
    
    for i, positions in enumerate(col_positions):
        for rank, j in enumerate(positions):  
            ax.text(j + 0.5, i + 0.5, str(rank + 1), color="black", fontsize=12, 
                    ha='center', va='center', fontweight='bold')

    cbar = ax.collections[0].colorbar
    if scale == True:
        orig_ticks = np.array([-5.0, -10.0, -15.82, -22.37, -31.63, -44.73])
        sqrt_ticks = orig_ticks
        orig_vals = -sqrt_ticks**2
        cbar.set_ticks(sqrt_ticks.astype(int))
        cbar.set_ticklabels([f"{int(v)}" for v in orig_vals])
    cbar.set_label("Similarity value", labelpad=10)

    plt.title("Target and context cell prototype similarity scores", fontsize=14)
    plt.xlabel("Cell types (context)", fontsize=14)
    plt.ylabel("Cell types (target)", fontsize=14)

    if save_key:
        os.makedirs("figures", exist_ok=True)
        out_path = os.path.join("figures", f"{save_key}.png")
        fig.savefig(out_path, dpi=300)
    plt.show()

def plot_similarity(
        adata_concat: mu.MuData,
        df_neigbor: pd.DataFrame,
        human_ind: int,
        rep_key: str = 'X_umap',
        plot_annot: str = 'cell_type_fine',
        context_species: str = 'mouse',
        target_species: str = 'human',
        save_key: Optional[str] = None,         
    ):

    """
    Scatter dataset representation of context vs. target in 2D (e.g., UMAP) colored by similarity to a specified target cell.

    Parameters
    ----------
    adata_concat : MuData
        Combined MuData with `.obsm[rep_key]` for both species.
    df_neigbor : pd.DataFrame
        DataFrame with columns ['index','similarity_score'] for a single target cell.
    human_ind : int
        Index of the target cell in `adata_concat`.
    rep_key : str, default='X_umap'
        Key in `.obsm` for 2D coordinates.
    plot_annot : str, default='cell_type_fine'
        Observation key for labeling the target cell.
    context_species : str, default='mouse'
    target_species : str, default='human'
    save_key : str or None, default=None
        If a string, the plot will be saved to `figures/{save_key}.png`. If None, it will only be displayed.   
    """

    umap_context = adata_concat.obsm[rep_key][adata_concat.obs.dataset == context_species]
    umap_target = adata_concat.obsm[rep_key][adata_concat.obs.dataset == target_species]
    c_t = adata_concat.obs[plot_annot][adata_concat.obs.dataset == target_species].iloc[human_ind]
    similarities = df_neigbor['similarity_score'].iloc[np.argsort(df_neigbor['index'])]

    if np.shape(umap_context)[1] != 2 or np.shape(umap_target)[1] != 2:
        raise ValueError("Representation must have exactly 2 dimensions (e.g., UMAP 2D coordinates)")

    fig, ax = plt.subplots(1, 1, figsize=(6,5), constrained_layout=True, dpi=250) 

    ax.scatter(umap_target[:,0], umap_target[:,1], s=40000/np.shape(umap_target)[0], c='lightgray')
    scatter = ax.scatter(umap_context[:,0], umap_context[:,1], s=40000/np.shape(umap_context)[0], c=similarities, cmap='RdYlGn_r')

    ax.set_xlim(np.min(np.concatenate((umap_context[:, 0], umap_target[:, 0])))*1.04, np.max(np.concatenate((umap_context[:, 0], umap_target[:, 0])))*1.04)
    ax.set_ylim(np.min(np.concatenate((umap_context[:, 1], umap_target[:, 1])))*1.04, np.max(np.concatenate((umap_context[:, 1], umap_target[:, 1])))*1.04)
    ax.scatter(umap_target[human_ind,0], umap_target[human_ind,1], s=50, c='black', marker='x', linewidths=2.5)

    plt.colorbar(scatter, ax=ax, label='Similarity scores')
    ax.set_title("Similarity scores of {} cell (Index {}) target cell with context cells.".format(c_t, str(human_ind)))

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if save_key:
        os.makedirs("figures", exist_ok=True)
        out_path = os.path.join("figures", f"{save_key}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_2D_representation(
        adata_concat: mu.MuData,
        rep_key: str = 'X_umap',
        plot_annot: str = 'cell_type_fine',
        context_species: str = 'mouse',
        target_species: str = 'human',
        save_key: Optional[str] = None,         
    ):

    """
    Scatter dataset representation of context vs. target in 2D (e.g., UMAP) with shared color mapping based on labels.

    Parameters
    ----------
    adata_concat : MuData
        Combined MuData with `.obsm[rep_key]` for both species.
    rep_key : str, default='X_umap'
        Key in `.obsm` for 2D coordinates.
    plot_annot : str, default='cell_type_fine'
        Observation key for the categorical annotation.
    context_species : str, default='mouse'
    target_species : str, default='human'
    save_key : str or None, default=None
        If a string, the plot will be saved to `figures/{save_key}.png`. If None, it will only be displayed.   
    """

    umap_context = adata_concat.obsm[rep_key][adata_concat.obs.dataset == context_species]
    umap_target = adata_concat.obsm[rep_key][adata_concat.obs.dataset == target_species]

    if np.shape(umap_context)[1] != 2 or np.shape(umap_target)[1] != 2:
        raise ValueError("Representation must have exactly 2 dimensions (e.g., UMAP 2D coordinates)")

    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(11,5), constrained_layout=True, dpi=250) 

    col_dict = return_palette(adata_concat.obs[plot_annot])

    colors_context = [col_dict[ct] for ct in adata_concat.obs[plot_annot][adata_concat.obs.dataset == context_species]]
    colors_target = [col_dict[ct] for ct in adata_concat.obs[plot_annot][adata_concat.obs.dataset == target_species]]

    perm_context = np.random.permutation(len(colors_context))
    perm_target = np.random.permutation(len(colors_target))

    ax_B.scatter(umap_context[:,0], umap_context[:,1], s=40000/len(colors_context), c='lightgray')
    ax_A.scatter(umap_target[:,0], umap_target[:,1], s=40000/len(colors_target), c='lightgray')

    ax_A.scatter(umap_context[perm_context,0], umap_context[perm_context,1], s=40000/len(colors_context), c=np.array(colors_context)[perm_context])
    ax_B.scatter(umap_target[perm_target,0], umap_target[perm_target,1], s=40000/len(colors_target), c=np.array(colors_target)[perm_target])

    ax_B.set_xlim(np.min(np.concatenate((umap_context[:, 0], umap_target[:, 0])))*1.04, np.max(np.concatenate((umap_context[:, 0], umap_target[:, 0])))*1.04)
    ax_B.set_ylim(np.min(np.concatenate((umap_context[:, 1], umap_target[:, 1])))*1.04, np.max(np.concatenate((umap_context[:, 1], umap_target[:, 1])))*1.04)

    ax_A.set_xlim(np.min(np.concatenate((umap_context[:, 0], umap_target[:, 0])))*1.04, np.max(np.concatenate((umap_context[:, 0], umap_target[:, 0])))*1.04)
    ax_A.set_ylim(np.min(np.concatenate((umap_context[:, 1], umap_target[:, 1])))*1.04, np.max(np.concatenate((umap_context[:, 1], umap_target[:, 1])))*1.04)

    ax_A.set_title(f"{context_species.capitalize()} Representation (Context)")
    ax_B.set_title(f"{target_species.capitalize()} Representation (Target)")

    ax_A.set_xticks([])
    ax_A.set_yticks([])
    ax_B.set_xticks([])
    ax_B.set_yticks([])

    ax_A.spines['top'].set_visible(False)
    ax_A.spines['right'].set_visible(False)
    ax_A.spines['bottom'].set_visible(False)
    ax_A.spines['left'].set_visible(False)

    ax_B.spines['top'].set_visible(False)
    ax_B.spines['right'].set_visible(False)
    ax_B.spines['bottom'].set_visible(False)
    ax_B.spines['left'].set_visible(False)

    legend = fig.legend(handles=[mpatches.Patch(color=col_dict[label], label=label) for label in col_dict.keys()], 
                        loc='upper left', 
                        bbox_to_anchor=(1.04, 1.02), 
                        labelspacing=0.20, 
                        handlelength=2, 
                        handletextpad=0.5, 
                        markerscale=2
                        )
    legend.get_frame().set_linewidth(1)  
    legend.get_frame().set_edgecolor('black')    

    if save_key:
        os.makedirs("figures", exist_ok=True)
        out_path = os.path.join("figures", f"{save_key}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def ret_sign(
    number: float
    ) -> str:

    """
    Return '+' if number ≥ 0, else '-'.
    Helper for `label_transfer_acc`.

    Parameters
    ----------
    number : float

    Returns
    -------
    str
        '+' or '-'
    """

    if number >= 0:
        sign_str = "+"
    else:
        sign_str = "-"
    return  sign_str  

def is_bright(
        hex_color: str
    ) -> str:

    """
    Determine whether a hex RGB color is “bright” based on luminance,
    to choose black or white text for readability.
    Helper for `label_transfer_acc`.

    Parameters
    ----------
    hex_color : str
        Hex code (e.g. '#RRGGBB').

    Returns
    -------
    str
        'black' if background is light, 'white' otherwise.
    """

    hex_color = hex_color.lstrip('#')
    R, G, B = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    threshold = 100 
    if luminance < threshold:
        return 'white'
    else:
        return 'black'   

def label_transfer_acc(
        df_nns: pd.DataFrame,
        df_sim: pd.DataFrame,
        save_key: Optional[str] = None,         
    ):

    """
    Compare balanced-accuracy of label-transfer by data-level NNs vs. scSpecies similarity-based label transfer
    and plot horizontal bar stacks of top-k context votes.

    Parameters
    ----------
    df_nns : pd.DataFrame
        Confusion-matrix-based accuracy of kNN transfers.
    df_sim : pd.DataFrame
        Confusion-matrix-based accuracy using scSpecies similarity.
    save_key : str or None, default=None
        If a string, the plot will be saved to `figures/{save_key}.png`. If None, it will only be displayed.   
    """

    context_cell_types = np.array(df_sim.columns)
    target_cell_types = np.array(df_sim.index)
    all_labels = np.unique(np.concatenate((context_cell_types, target_cell_types)))
    common_labels, _, c_ind_b = np.intersect1d(context_cell_types, target_cell_types, return_indices=True)

    figsize=(12,0.9+len(target_cell_types)*11/16)

    fig, ax = plt.subplots(figsize=figsize, dpi=250)
    add = 0.01
    fs=20

    palette = return_palette(all_labels)

    legend = fig.legend(handles=[mpatches.Patch(color=palette[cell], label=str(i+1)+': '+cell) for i,cell in enumerate(all_labels) if cell in context_cell_types], loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=fs*0.8, ncol=4, columnspacing=0.1)
    legend.get_frame().set_linewidth(1)  
    legend.get_frame().set_edgecolor('black') 

    improvement = np.array([df_sim.loc[cell_type, cell_type] - df_nns.loc[cell_type, cell_type] for cell_type in common_labels])

    cells_old = np.array([df_nns.loc[key].sort_values(ascending=False).index for key in df_nns.index])
    cells_new = np.array([df_sim.loc[key].sort_values(ascending=False).index for key in df_sim.index])
    values_old = np.array([df_nns.loc[key].sort_values(ascending=False) for key in df_nns.index])
    values_new = np.array([df_sim.loc[key].sort_values(ascending=False) for key in df_sim.index])

    str_old = [str(i)+' - '+str(j) for (i,j) in zip(np.array([np.where(i == all_labels)[0][0]+1 for i in cells_old[:,0]]), np.array(round(df_nns.max(1),1)))]
    str_new = [str(i)+' - '+str(j) for (i,j) in zip(np.array([np.where(i == all_labels)[0][0]+1 for i in cells_new[:,0]]), np.array(round(df_sim.max(1),1)))]

    labels = [' 'for i in target_cell_types]
    for j, ind in enumerate(c_ind_b):
        labels[ind] = ret_sign(improvement[j])+str(round(abs(improvement[j]),1))

    tick_colors = []
    for numb in labels:
        if numb == ' ':
            tick_colors += ['#212121'] 
        elif float(numb) < -20:
            tick_colors += ['#b71c1c']     
        elif float(numb) < -10:
            tick_colors += ['#c62828'] 
        elif float(numb) < -4:
            tick_colors += ['#e53935'] 
        elif float(numb) < 0:
            tick_colors += ['#e57373'] 
        elif float(numb) == 0.0:
            tick_colors += ['#212121'] 
        elif float(numb) < 4:
            tick_colors += ['#66bb6a'] 
        elif float(numb) < 10:
            tick_colors += ['#4caf50']         
        elif float(numb) < 20:
            tick_colors += ['#43a047']    
        elif float(numb) < 100:
            tick_colors += ['#388e3c']   
    tick_colors = np.array(tick_colors)    

    x = np.arange(len(df_nns.index))

    for i in range(len(target_cell_types)):
        n=0
        x_positions = np.flip(x+(0.5*n)-1.0/2)
        ax.axhline(y=i+0.5, color='black', linestyle='solid')
        p = ax.barh(x_positions, width=values_new[:,i], left=np.cumsum(np.insert(values_new, 0, 0, axis=1), axis=1)[:,i], edgecolor="white", height=0.5, align='edge', color=[palette[c] for c in cells_new[:,i]])
        n=1
        x_positions = np.flip(x+(0.5*n)-1.0/2)
        p_2 = ax.barh(x_positions, width=values_old[:,i], left=np.cumsum(np.insert(values_old, 0, 0, axis=1), axis=1)[:,i], edgecolor="white", height=0.5, align='edge', color=[palette[c] for c in cells_old[:,i]])    

    ax.set_yticks(x)
    ax.set_ylim(-0.5, len(df_nns.index)-0.5)
    ax.set_yticklabels(np.array([str(np.where(np.array(all_labels)==s)[0][0]+1)+': '+s for s in np.flip(df_nns.index)]), rotation=50, ha='right', fontsize=fs*0.95)#
    ax_t = ax.secondary_yaxis('right')
    ax_t.set_yticks(x)
    ax_t.tick_params(axis='y', direction='inout', length=10)
    ax_t.set_yticklabels(np.flip(labels), ha='left', fontsize=fs)#

    for i, color in enumerate(np.flip(tick_colors)):
        ax_t.get_yticklabels()[i].set_color(color)

    ax.set_xticks(np.arange(0,110,10))
    ax.set_xticklabels(np.array([i.astype(int).astype(str) for i in np.arange(0,110,10).astype(int).astype(str)]), fontsize=fs)#
    ax.xaxis.set_tick_params(length=10)

    ax_f = ax.secondary_xaxis('top')
    ax_f.set_xticks(np.arange(0,110,10))
    ax_f.set_xticklabels(np.array([i.astype(int).astype(str) for i in np.arange(0,110,10)]), fontsize=fs)#
    ax_f.xaxis.set_tick_params(length=10)

    [ax.axvline(x=i, color='lightgray', linestyle='dashed', lw=1.25) for i in np.arange(0,110,10)]

    for j,x_pos in enumerate(x_positions):
        ax.text(values_new[j,0]/2-5.5, x_pos-(0.45-add), str_new[j], fontsize=fs, color=is_bright(palette[cells_new[j,0]]))       
        ax.text(values_old[j,0]/2-5.5, x_pos+(0.05+add), str_old[j], fontsize=fs, color=is_bright(palette[cells_old[j,0]]))        

    pos_old = np.cumsum(np.insert(values_old, 0, 0, axis=1), axis=1) + np.insert(values_old, -1, 0, axis=1)/2 - 1 
    pos_new = np.cumsum(np.insert(values_new, 0, 0, axis=1), axis=1) + np.insert(values_new, -1, 0, axis=1)/2 - 1 

    def ret_val(value):
        if np.abs(value) >= 10:
            return 0.15
        else:
            return -0.25

    for i in range(1, 6):
        for j,x_pos in enumerate(x_positions):
            if values_new[j,i] > 4:
                ind = np.where(cells_new[j,i] == all_labels)[0][0]+1
                ax.text(pos_new[j,i]-ret_val(ind), x_pos-(0.45-add), str(ind), fontsize=fs, color=is_bright(palette[cells_new[j,i]]))

            if values_old[j,i] > 4:
                ind = np.where(cells_old[j,i] == all_labels)[0][0]+1            
                ax.text(pos_old[j,i]-ret_val(ind), x_pos+(0.05+add), str(ind), fontsize=fs, color=is_bright(palette[cells_old[j,i]]))        

    nn_old = round(np.mean([df_nns.loc[cell_type, cell_type] for cell_type in common_labels]),1)
    nn_new = round(np.mean([df_sim.loc[cell_type, cell_type] for cell_type in common_labels]),1)

    ax.set_title('Data-level neighbor search. Label transfer balanced accuracy: '+str(nn_old)+'%.\n '
                'scSpecies similarity measure. Label transfer balanced accuracy: '+str(nn_new)+'%.\n ', fontsize=fs*1.25)

    if save_key:
        os.makedirs("figures", exist_ok=True)
        out_path = os.path.join("figures", f"{save_key}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_lfc_comparison(
        model, 
        lfc_dict: list,
        save_key: Optional[str] = None,         
    ):

    """
    Generate and display a grid of scatter plots comparing log₂‐fold changes
    estimated by scSpecies against LFC computed directly from the data.

    Parameters
    ----------
    model : scSpecies
        A trained and evaluated scSpecies model instance.
    lfc_dict : dict of {str: pandas.DataFrame}
        List of LFC dataframes.
    save_key : str or None, default=None
        If a string, the plot will be saved to `figures/{save_key}.png`. If None, it will only be displayed.   
    """                        
                        

    target_ind = np.array(model.target_config['homologous_genes'])
    target_gene_names = model.mdata.mod['human'].var_names.to_numpy()[target_ind]
    cell_types = list(lfc_dict.keys())[:-1]

    df_lfc_dat = pd.DataFrame(0, index=target_gene_names, columns=cell_types)

    spear = {}
    pear = {}
    kend = {}

    for ct in cell_types:
        adata_target = model.mdata.mod['human'][:, np.array(model.target_config['homologous_genes'])]
        adata_context = model.mdata.mod['mouse'][:, np.array(model.context_config['homologous_genes'])]

        adata_target = adata_target[adata_target.obs['cell_type_fine'] == ct].copy()
        adata_context = adata_context[adata_context.obs['cell_type_fine'] == ct].copy() 

        sc.pp.normalize_total(adata_context, target_sum=1e6, inplace=True)
        sc.pp.normalize_total(adata_target, target_sum=1e6, inplace=True)

        adata_context = np.mean(adata_context.X.toarray(), axis=0)
        adata_target = np.mean(adata_target.X.toarray(), axis=0)

        lfc = np.log2(adata_target+1) - np.log2(adata_context+1) 
        df_lfc_dat[ct] = lfc

        sort_data = np.argsort(lfc)
        sort_model = np.argsort(lfc_dict[ct]['lfc'])

        spear[ct] = spearmanr(lfc, lfc_dict[ct]['lfc']).statistic
        pear[ct] = pearsonr(lfc, lfc_dict[ct]['lfc']).statistic
        kend[ct] = kendalltau(np.arange(len(lfc)), np.argsort(lfc[sort_model])).statistic   

    df_lfc   = pd.DataFrame({ct: lfc_dict[ct]['lfc'] for ct in cell_types})

    n_cell_types = len(cell_types)
    n_cols = 4  
    n_rows = int(np.ceil(n_cell_types / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)
    axs = axs.flatten()

    for i, cell_type in enumerate(cell_types):
        ax = axs[i]

        sorted_idx = np.argsort(df_lfc[cell_type].values)

        x = df_lfc_dat[cell_type].iloc[sorted_idx].values
        y = df_lfc[cell_type].iloc[sorted_idx].values

        ax.scatter(x, y, c='darkgrey', s=12, edgecolor='k')
        ax.set_xlabel('LFC data-level')
        ax.set_ylabel('LFC scSpecies')
        ax.set_title(
            f"{cell_type}\n"
            f"ρ={spear[cell_type]:.2f}, "
            f"PCC={pear[cell_type]:.2f}, "
            f"τ={kend[cell_type]:.2f}",
            pad=10
        )

        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, '--', color='red', linewidth=1)

    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(
        "LFC: scSpecies vs data-level\n"
        f"Avg ρ={np.mean(list(spear.values())):.2f}, "
        f"Avg PCC={np.mean(list(pear.values())):.2f}, "
        f"Avg τ={np.mean(list(kend.values())):.2f}"
    )

    if save_key:
        os.makedirs("figures", exist_ok=True)
        out_path = os.path.join("figures", f"{save_key}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()    

def return_palette(
        names: Union[pd.Index, np.ndarray, List[str]],
        col_dict: dict = {}
    ) -> dict[str, str]:

    """
    Build a color mapping for a list of labels, using predefined overrides
    and extending with Glasbey palette for unknowns.

    Parameters
    ----------
    names : sequence of str
        Labels to assign colors.
    col_dict : dict, optional
        Predefined name→hex mappings.

    Returns
    -------
    dict[str, str]
        Mapping from each unique name in `names` to a hex color code.
    """

    col_dict = {**col_dict, **{
        'human':                    '#9467bd',
        'mouse':                    '#ff7f0e',
        'hamster':                  '#2ca02c',

        'Stromal Cells':            '#ff4500',
        'Endothelial':              '#ffbf00', 
        'Mesothelial Cells':        '#E0115F',
        'Mono/Mono Derived':        '#4f86f7',
        'NK/NKT':                   '#839B17', 
        'NK/NKT cells':            '#839B17',         

        'CD4+ KLRB1 Th':            '#74C365',
        'Circ. NK':                 '#8F9779',
        'Gamma-Delta T':            '#9acd32', 
        'Mesothelial Cells':        '#E0115F',
        'Mig. DCs':                 '#8f00ff', 
        'Mig. cDCs':                '#8f00ff',        
        'NK Cells':                 '#568203',      
        'NKT Cells':                '#043927', 
        'Pat. Monocytes':           '#008ECC', 
        'Peritoneal Macs':          '#6960EC',
        'Pre-moKCs and moKCs':      '#38ACEC',
        'RM CD8+ T cells':          '#D0F0C0',
        'Stellate Cells':           '#ff033e',  
        'Th 1':                     '#50C878',
        'Th 17':                    '#00A86B', 
        'Tissue Resident NK':       '#708238',
        'Trans. Monocytes 2':       '#6495ED',
        'cDCs 1':                   '#6F2DA8', 
        'cDCs 2':                   '#81007F', 
        'immLAMs':                  '#000080',
        'matLAMs':                  '#1035AC',

        'B Cells':                  '#964b00', 
        'Basophils':                '#000000',
        'cDC1s':                    '#6F2DA8', 
        'cDC2s':                    '#81007F', 
        'pDCs':                     '#D891EF',        
        'Plasma':                   '#c19a6b',
        'NK':                       '#568203',         
        'NKT':                      '#043927', 
        'Gamma delta T Cells':      '#9acd32', 
        'Fibroblasts':              '#ff3800', 
        'Hepatocytes':              '#ff0090',
        'Stellate':                 '#ff033e',  
        'Regulatory T':             '#0B6623',	
        'Monocytes':                '#87CDEE', 
        'Neutrophils':              '#8c8784', 
        'Circulating NK':           '#8F9779',
        'Cholangiocytes':           '#c90016', 
        'LSECs':                    '#ffa700', 
        'KCs':                      '#00bfff', 
        'Kupffer cells':            '#00bfff',         
        'Migratory cDCs':           '#8f00ff', 
        'Imm. LAMs':                '#000080',
        'Mat. LAMs':                '#1035AC',
        'MoMac1':                   '#0020C2', 
        'Cytotoxic CD8+':           '#A7F432', 
        'Patrolling Mono.':         '#008ECC', 
        'Tissue Res. NK':           '#708238',
        'Circ. Eff. Memory T':      '#98FB98', 
        'Portal Vein ECs':          '#ffd700',
        'Pre-moKCs/moKCs':          '#38ACEC',
        'Central Vein ECs':         '#fcc200', 
        'Naive/CM CD4+ T':          '#2E8B57', 
        'CD4+ KLRB1 T':             '#74C365',
        'Tissue Res. CD8+ T':       '#D0F0C0',
        'Trans. Mono. 1':           '#b5d7fd',  
        'Trans. Mono. 2':           '#82EEFD',
        'Th17s':                    '#00A86B', 
        'Th1s':                     '#50C878',
        'Mesothelial':              '#E0115F',
        'Peritoneal Mac.':          '#6960EC',

        'B/Plasma':                 '#6F4E37',
        'Naive CD8+ T':             '#A8E4A0', 
        'RM CD8+ T Cells':          '#D0F0C0',
        'CD8 Eff. Memory T':        '#D1FFBD',
        'T Cells':                  '#4CBB17', 
        'Naive CD4+ T':             '#48A860',
        'CD4+ T helper':            '#50a88b', 
        'NK/NKT Cells':             '#839B17', 
        'ILCs':                     '#7fff00',
        'Prol. TAM':                '#93FFE8', 
        'Bile-duct LAMs':           '#5865F2',
        'MoMac2':                   '#0041C2', 
        'CV/Capsule Cd207+ Macs':   '#1D2951',
        'Macrophages':              '#101D6B', 
        'Monocytes 1':              '#1974D2', 
        'Monocytes 2':              '#0909FF', 
        'Mono/mono-derived':        '#4f86f7',
        'Trans. Monocytes':         '#6495ED',
        'Mast Cells':               '#00CC99', 
        'DCs':                      '#80008B', 
        'DCs 3':                    '#342D7E',
        'cDCs':                     '#80008B', 
        'pre-DC':                   '#36013F',
        'Endothelials':             '#ffbf00', 
        'Lymphatic ECs':            '#FCE205', 
        'Lymphatic ECs 1':          '#C35817', 
        'Lymphatic ECs 2':          '#C04000', 
        'Endometrium':              '#FF8C00', 
        'Capsule Fibroblasts':      '#cf1020', 
        'Fibroblast 1':             '#f08080',
        'Fibroblast 2':             '#ff4500', 
        'Stromal':                  '#ff4500', 
        'Adipocytes':               '#FF2400', 
        'Pericytes':                '#FE5BAC', 
        'Myo Epithelials':          '#98AFC7',
        'Platelets':                '#837E7C',
    }}

    name_list = np.unique(names)
    lenght = 10 + len(name_list)
    colors = glasbey.extend_palette("tab10", palette_size=lenght)

    palette = {}

    j=0
    for name in name_list:       
        if name in col_dict.keys():
            palette[name] = col_dict[name]
        else: 
            print('\nAssingning color to unknown cell type: '+name)
            palette[name] = (colors[j], )
            j+=1
            
    return palette


def load_and_filter_pathways(
        gmt_path: str, 
        adata: ad.AnnData, 
        min_genes: int = 5,
        ) -> dict:
    
    """
    Load pathway gene sets from a GMT file and filter to those
    with at least `min_genes` overlapping with adata.var_names.

    Parameters
    ----------
    gmt_path : str
        Path to the .gmt file.
    adata : AnnData
        AnnData object with .var_names (genes).
    min_genes : int
        Minimum number of overlapping genes to keep a pathway.

    Returns
    -------
    filtered_pathways : dict
        Mapping of pathway names to lists of overlapping gene symbols.
    """

    pathways = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0]
            gene_list = parts[2:]
            pathways[name] = gene_list

    var_set = set(adata.var_names)
    filtered = {}
    for name, genes in pathways.items():
        overlap = var_set.intersection(genes)
        if len(overlap) >= min_genes:
            filtered[name] = list(overlap)

    return filtered