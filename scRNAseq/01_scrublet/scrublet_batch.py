import matplotlib.pyplot as plt
import tqdm as notebook_tqdm
from datetime import date
import scrublet as scr
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
import scipy
import sys
import os
import re
from statsmodels.sandbox.stats.multicomp import multipletests

f"Last execution: {date.today()}"

# -- Set base directory
base_dir = '/nfs/team292/rs40/projects/Aneuploid_screen_v2'
sys.path.insert(1, base_dir)
os.chdir(base_dir)

# -- Outputs
#datafiles
output_dir = base_dir+'/processed_data/scrublet'
path = os.path.join(base_dir, output_dir)

if not os.path.exists(path):
    os.makedirs(path)
    

out_annotation = 'scrublet/'

samples = ['Embryo-Imp15265938', 
           'Embryo-Imp15265939',
           'Embryo-Imp15265940',
           'Embryo-Imp15265941',
           'Embryo-Imp15265942',
           'Embryo-Imp15265943',
           'Embryo-Imp15265944',
           'Embryo-Imp15265945',
           'Embryo-Imp15265946',
           'Embryo-Imp15265947']


for sample in list(samples):
    #directory with mtx files
        input_directory = '/lustre/scratch126/cellgen/vento/rs40/from_iRODs/cellranger900_count_49831_' + sample +'_GRCh38-2024-A/filtered_feature_bc_matrix.h5'
        
        out_file = output_dir+'/'+sample+'_scrublet.csv'
        
        print(f'Calculating scrublet: {sample}')

        #read the data file
        adata_sample = sc.read_10x_h5(input_directory)
        adata_sample.var_names_make_unique()

        # -- Rename Sample barcode by adding sample id
        #adata_sample.obs_names = [sample+'_'+i for i in adata_sample.obs_names]
        
        min_genes = 100
        min_cells = 3
        rnd_seed = 88

        # -- Basic initial filtering
        sc.pp.filter_cells(adata_sample,
                           min_genes = min_genes)

        sc.pp.filter_genes(adata_sample,
                           min_cells = min_cells)

        sc.pp.calculate_qc_metrics(adata_sample,
                                   inplace = True,
                                   percent_top = None)

        # -- Calculate doublet score with scrublet
        np.random.seed(rnd_seed)

        scrub = scr.Scrublet(adata_sample.X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose = False)
        adata_sample.obs['scrublet_score'] = doublet_scores
        adata_sample.obs['scrublet_prediction'] = predicted_doublets

        # -- Plot scrublet distribution
        #sns_plot = sns.displot(adata_sample.obs['scrublet_score']).set(title = sample)
        #sns_plot.figure.savefig(figure_dir+'/' + sample + '_.pdf')
        
        scrub.plot_histogram();
        plt.savefig(output_dir +'/'+ sample + '_hist.pdf')
        plt.close('all')
        
        # -- Code as used by Luz (for small cluster based doublet detection) 
        # -- overcluster prep. run turbo basic scanpy pipeline
        sc.pp.normalize_per_cell(adata_sample,
                                 counts_per_cell_after = 1e4)
        sc.pp.log1p(adata_sample)
        sc.pp.highly_variable_genes(adata_sample,
                                    min_mean = 0.0125,
                                    max_mean = 3,
                                    min_disp = 0.5)

        adata_sample = adata_sample[:, adata_sample.var['highly_variable']]

        sc.pp.scale(adata_sample,
                    max_value = 10)

        sc.tl.pca(adata_sample,
                  svd_solver='arpack')

        sc.pp.neighbors(adata_sample)

        # -- overclustering proper - do basic clustering first, then cluster each cluster
        sc.tl.leiden(adata_sample)
        adata_sample.obs['leiden'] = [str(i) for i in adata_sample.obs['leiden']]

        for clus in np.unique(adata_sample.obs['leiden']):

            adata_sub = adata_sample[adata_sample.obs['leiden']==clus].copy()
            sc.tl.leiden(adata_sub)
            adata_sub.obs['leiden'] = [clus+','+i for i in adata_sub.obs['leiden']]
            adata_sample.obs.loc[adata_sub.obs_names,'leiden'] = adata_sub.obs['leiden']

        #compute the cluster scores - the median of Scrublet scores per overclustered cluster
        for clus in np.unique(adata_sample.obs['leiden']):

            results = np.median(adata_sample.obs.loc[adata_sample.obs['leiden']==clus, 'scrublet_score'])
            adata_sample.obs.loc[adata_sample.obs['leiden']==clus, 'scrublet_cluster_score'] = results

        #now compute doublet p-values. figure out the median and mad (from above-median values) for the distribution
        med = np.median(adata_sample.obs['scrublet_cluster_score'])
        mask = adata_sample.obs['scrublet_cluster_score'] > med
        mad = np.median(adata_sample.obs['scrublet_cluster_score'][mask]-med)

        #let's do a one-sided test. the Bertie write-up does not address this but it makes sense
        zscores = (adata_sample.obs['scrublet_cluster_score'].values - med) / (1.4826 * mad)
        adata_sample.obs['scrublet_zscore'] = zscores
        pvals = 1-scipy.stats.norm.cdf(zscores)
        adata_sample.obs['scrublet_bh_pval'] = multipletests(pvals, alpha=.05, method='bonferroni')[1]
        adata_sample.obs['scrublet_bonf_pval'] = multipletests(pvals, alpha=.05, method='fdr_bh')[1]

        # -- Extract scrublet annotation
        idx = [ bool(re.match('scrublet', i)) for i in adata_sample.obs.columns ]
        scrublet_sample = adata_sample.obs.loc[:, idx]

        # -- Write results
        scrublet_sample.to_csv(out_file)
    
print('Done')