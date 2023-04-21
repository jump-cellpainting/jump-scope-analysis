# Each batch requires its own config file, despite the same settings being used 
# for all profiles. This is because the config file itself defines the profile 
# paths that should be loaded.

library("magrittr")
library("purrr")

setwd("~/evalzoo/matric/")

source("run_param.R")

param_files <- list.files(path = "/input/params/", recursive = TRUE, full.names = TRUE)

results_dir <- "/input/results"

c(param_files) %>%
  walk(function(i)
    run_param(
      param_file = i,
      results_root_dir = results_dir
    ))