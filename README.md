# Morphology and Expression Discordance in Mouse Embryo Brain Cortex

Our project examines different ways ways to integrate cell morphology and gene expression modalities to find discordant cells and their specific properties. For our
expression data, we retrieved data from Lohoff et. al, where they performed SeqFish , a spatial molecular imaging technique, to get gene expression data of mouse embryos in the E8.5 region. 
We got three datasets from this work, each with different tissue regions in the brain. For getting our cell morphology metrics, we used a pretrained Vision Transformer applied to each cell’s segmentation mask from the datasets and reduced our embeddings with PCA.  We also computed additional scalar metrics (area, perimeter, circularity, etc) 
to add to our embeddings as an overall morphology metrics set. We measured the relationship between gene expression and morphology with a canonical correlational analysis, which gives us different CC components representing the shared variance between the two modalities.  
We compared this to a null dataset in which the morphology metric rows in our dataset were shuffled, while the expression data and covariates were fixed. For each cell, we generated an abnormality score for expression and one for morphology  with the negative loglikelihood of values under a Gaussian Density model. 
We also generated two discordance scores of the relationship between gene expression and morphology through the predicted gene expression from morphology and vice versa using the negative loglikelihood of values under a Ridge Regression model. All scores were z-scored within cell type and boundary groups. 
We assigned each cell to one of the five categories: “Morphology Outlier”, “Gene Expression Outlier”, “Consistent Relationship but Abnormality Outlier”, “Outlier in all scores”, or most importantly, “Hidden Discordance”. 


<img width="717" height="447" alt="Screenshot 2026-10-19 at 2 45 44 PM" src="https://github.com/user-attachments/assets/ba69f3bc-7ac5-4ef1-8aa6-e3eb5e3df16a" />


## Key Results

| Dataset | Cells | Observed CC1 | Null mean | Null 95th pct |
|---|---|---|---|---|
| 1: Neural Crest and FMH* | 1,039 | 0.25 | 0.14 | 0.21 | 
| 2: Cranial Mesoderm and FMH* | 859 | 0.39 | 0.01 | 0.07 | 
| 3: Midbrain and Hindbrain | 866 | 0.23 | 0.05 | -0.02 | 

*FMH: Forebrain/Midbrain/Hindbrain

| Dataset | Typical Cells | Discordant Cells | Morphology Outlier | Expression Outlier |
|---|---|---|---|---|
| 1: Neural Crest and FMH* | 812 | 100 | 46 | 41 | 
| 2: Cranial Mesoderm and FMH* | 675 | 84 | 43 | 33 | 
| 3: Midbrain and Hindbrain | 681 | 84 | 41 | 35 | 

## Repository Layout

- src/discordance/:   reusable code 
- notebooks/:         analysis notebooks, run in numeric order
- configs/:           dataset paths and parameters
- results/:           small derived tables
- figures/:           final figures
- datasets/:          extracted data used for analysis


## Data
- Lohoff et al. seqFISH, E8.5 mouse embryo 
- Hallou et al. segmentation masks and TensionMap 

## Installation
```bash
git clone https://github.com/keerthyr-maker/discordance_exp_morph.git
cd discordance_exp_morph
conda env create -f environment.yml
conda activate discordance_exp_morph
```

## Usage
Run the notebooks in order:
1. `FinalDFCompilation.ipynb`: builds one table per dataset (1,039 / 859 / 866 cells)
2. `NullCCAs.ipynb`: held-out CCA and 200-permutation nulls (Figure 2)
3. `DiscordantCells.ipynb`: abnormality and discordance scores (Figure 3)



