#!/usr/bin/env Rscript
# Author: Zhenhua Zhang
# E-mail: zhenhua.zhang217@gmail.com
# Created: Mar 15, 2022
# Updated Mar 15, 2022

suppressPackageStartupMessages({
  library(data.table)
  library(tidyverse)
  library(magrittr)
  library(Seurat)
})

proj_dir <- "~/Documents/projects/wp_codex"
out_dir <- file.path(proj_dir, "outputs")
in_dir <- file.path(proj_dir, "inputs")

mhh50_sobj_file <- file.path(in_dir, "scRNA-seq/pbmc.filter.annotev5.rds")
mhh50_sobj <- readRDS(mhh50_sobj_file)


tar_cols <- c(
  "SampleID", "patient", "gender", "Age", "Days.post.convalescent", "WHO_score",
  "Severity"
)

force_save <- TRUE
sample_info <- mhh50_sobj@meta.data %>%
  select(one_of(tar_cols)) %>%
  unique() %>%
  group_by(patient, gender, Age) %>%
  arrange(Days.post.convalescent) %>%
  filter(n() > 1) %>%
  (function(e) {
    g <- e %>%
      mutate(patient = as.factor(patient)) %>%
      ggplot(aes(
        x = Days.post.convalescent, y = patient, color = gender,
        label = WHO_score
      )) +
      geom_label(alpha = 0.5) +
      geom_vline(xintercept = 0, linetype = "dashed") +
      theme_classic()

    save_to <- str_glue("{out_dir}/mhh50_sample_info_summary.pdf")
    if (!file.exists(save_to) || force_save) {
      ggsave(save_to, plot = g)
    }

    return(e)
  }) %>%
  summarise(
    sample_id = paste(SampleID, collapse = ","),
    sampling_days = paste(Days.post.convalescent, collapse = ","),
    WHO_score = paste(WHO_score, collapse = ","),
    severity = paste(Severity, collapse = ",")
  ) %>%
  (function(dtfm) {
    save_to <- str_glue("{out_dir}/mhh50_sample_info_summary.csv")
    if (!file.exists(save_to) || force_save) {
      fwrite(dtfm, save_to)
    }

    return(dtfm)
  })
