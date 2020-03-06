# LatentSpaceOptimization
Latent Space Optimization over AtlasNet Point Cloud Generation

# Steps to Run Ellipsoid Experiments
1. Generate the ellipsoid data set with the command
`python AtlasNet/auxiliary/ellipsoid_dataset.py`
This will populate the directory 'AtlasNet/data/ellipsoid_points/' with 6250 elliposoid point clouds drawn from the distributions illustrated in Figure 5 of the paper. 

2. Run LSO with the command
`python multi_run.py --lam_array 0.4 0.2 0.1`
This will run 5 trials of LSO for the Naive method and SP-LSO with each of the above lambda values and 35 Differential Evolution iterations per trial. The parameter num_params should be left at its default, as it must match the latent space size of the pre-trained AtlasNet model included in AtlasNet/trained_model. This is left as an optional parameter for a user who would like to train their own AtlasNet model with a different latent space dimensionality.
