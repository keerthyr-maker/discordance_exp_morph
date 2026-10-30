import pandas as pd
import numpy as np


def add_prefixes(expr_pcs, scalar_metrics, pcs_embeddings):
    expr_pcs = expr_pcs.add_prefix('expr_')
    scalar_metrics = scalar_metrics.add_prefix('morph_')
    pcs_embeddings = pcs_embeddings.add_prefix('morph_')
    return expr_pcs,scalar_metrics, pcs_embeddings

def get_id_map(data):
    link = data[['mask_label', 'target_cell']].drop_duplicates()
    id_map = (link.assign(cell_id=link['target_cell'].astype(str))[['cell_id', 'mask_label']]
                  .drop_duplicates())
    return id_map

def tension_map(tension_data, id_map):
    tensionmap_data = tension_data[tension_data['celltype'] != 'Low quality']

    tm_m = (tm.reset_index().rename(columns={'Unnamed: 0': 'cell_id'})
            .merge(id_map, on='cell_id', how='inner')
            .set_index('mask_label'))
    cx, cy = tm_m['centroid_x'], tm_m['centroid_y']
    tm_m['spatial_block'] = ((cx > cx.median()).astype(int) * 2 + (cy > cy.median()).astype(int))
    tm_m = tm_m.add_prefix('tm_')
    return tm_m

def join(exp_pcs, scalar_data, pcs_embeddings, tm_m):
    expr_pcs = expr_pcs.add_prefix('expr_')
    scalar_metrics = scalar_metrics.add_prefix('morph_')
    pcs_embeddings = pcs_embeddings.add_prefix('morph_')
    morph_data = scalar_metrics.join(pcs_embeddings, how='inner')
    final_exp_df = expr_pcs.join(tm_m[['tm_spatial_block', 'tm_celltype', 'tm_boundary_annotation', 'tm_cell_id','tm_centroid_x','tm_centroid_y']],how='inner')
    return final_exp_df.join(morph_data, how='inner')





