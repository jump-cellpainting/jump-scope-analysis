# Running evalzoo

1. Pull image with `docker pull shntnu/evalzoo:latest`
2. Run RStudio server from docker with `docker run -it --rm -v ~/Desktop/input:/input -p 127.0.0.1:8787:8787 -e DISABLE_AUTH=true shntnu/evalzoo:latest`
   1. Login at [127.0.0.1:8787:8787](127.0.0.1:8787:8787)
3. Git pull the evalzoo repo, if required
4. File > Open Project and open `Evalzoo.Rproj`
5. Prior to running evalzoo, create the required param files for each batch of plates. This is done to preserve the folder structure controlled by `match_rep_df.csv`.
   1. Run `make_params` in `create_evalzoo_params.ipynb`
      1. Add the created param files **and profiles** (in their original folder structure) to `~/Desktop/input`
      2. The structure should be:
```
├── input
│   ├── params
│   ├── profiles
```
6. In RStudio server, run `source("input/run_eval.R")
   1. This script finds the `.yml` param files and runs evalzoo
   2. Results will be saved to `~/Desktop/input/results`
7. Run `postprocess_evalzoo` in `create_evalzoo_params.ipynb` to remove unnecessary files and rename the parent folder from a hash to the batch the folder contains


