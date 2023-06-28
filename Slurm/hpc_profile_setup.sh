# If using OzStar, the module to load is apptainer/latest. Petrichor uses singularity. 

# OzStar:
export APPTAINER_BINDPATH="/fred/oz237/kluken/redshift_pipeline_adacs/"
module load apptainer/latest

# Singularity:
# export SINGULARITY_BINDPATH="/some/location/"
# module load singularity/3.7.3

# Setup paths to scripts and containers
export container_path="/fred/oz237/kluken/redshift_pipeline_adacs/Containers"
export script_path="/fred/oz237/kluken/redshift_pipeline_adacs/Scripts"