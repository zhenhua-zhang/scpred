#!/usr/bin/env Rscript
# Author: Zhenhua Zhang
# E-mail: zhenhua.zhang217@gmail.com
# Created: Mar 02, 2022
# Updated: Mar 08, 2022

options(stringsAsFactors = FALSE, error = traceback)

suppressPackageStartupMessages({
  library(data.table)
  library(tidyverse)
  library(magrittr)
  library(Seurat)
})

proj_dir <- "~/Documents/projects/wp_codex"
out_dir <- file.path(proj_dir, "outputs")
in_dir <- file.path(proj_dir, "inputs")

min_nfeature <- 200
min_ncount <- 500
min_mt_pct <- 25

#
## Bohn cohort
#
if (TRUE) {
  bohn_sobj_file <- file.path(
    in_dir, "scRNA-seq", "seurat_COVID19_PBMC_jonas_FG_2020-07-23.rds"
  )
  bohn_sobj <- base::readRDS(bohn_sobj_file) %>%
    subset(subset = min_nfeature <= nFeature_RNA
           & min_ncount <= nCount_RNA
           & percent.mito < min_mt_pct)

  bohn_sobj@meta.data %>%
    colnames() %>%
    paste0(collapse = "\n") %>%
    cat("\n")

  # Check the sampling days and WHO score per sample
  bohn_kept_mt_cols <- c(
    "donor", "who_per_sample", "date_of_sampling",
    "days_after_onset", "sex", "sampleID"
  )
  bohn_sample_info <- bohn_sobj@meta.data %>%
    select(one_of(bohn_kept_mt_cols)) %>%
    unique() %>%
    group_by(donor, sex) %>%
    filter(n() > 1) %>% # Removing donors who only have one sample.
    mutate(
      date_of_sampling = as.Date(
        date_of_sampling,
        format = "%d.%m.%Y", origin = "%d-%m-%Y"
      )
    ) %>%
    arrange(date_of_sampling) %>% # Sort the records by sampling date.
    summarise(
      WHO_Classification = paste0(who_per_sample, collapse = ","),
      sampling_day = paste0(date_of_sampling, collapse = ","),
      sample_id = paste0(sampleID, collapse = ","),
      # Using the maximum WHO score as the response variable.
      response_var = max(who_per_sample),
      # Using the earliest sample as the predictor statement.
      predictor_var_id = sampleID[1]
    ) %>%
    (function(d) {
      fwrite(d, str_glue("{out_dir}/bohn_sample_info_summary.csv"))
      return(d)
    })


  # Donors should be included
  # From the table above, we can figure out which samples should be removed.
  #   1. BN-22, no WHO score available.
  #   2. BN-01, the participant is "recovering".
  #   3. BN-10, wired, his situation worsen repidly, 4 on 03-31, but 7 on 04-01

  # Samples should be include
  # We include samples with the lowest WHO score
  bohn_kept_samples <- bohn_sample_info %>%
    filter(!donor %in% c("BN-22", "BN-01", "BN-10")) %$% predictor_var_id


  # Cells should be included
  bohn_kept_cells <- bohn_sobj@meta.data %>%
    filter(sampleID %in% bohn_kept_samples) %>%
    rownames()

  # A summary for all samples.
  bohn_sample_summary <- bohn_sobj[, bohn_kept_cells]@meta.data %>%
    group_by(sampleID, donor, sex) %>%
    summarise(n_cells = n(), max_WHO_score = max(who_per_sample)) %>%
    mutate(condition = if_else(as.integer(max_WHO_score) >= 5, "S", "M"))

  # Here, we only select genes expressed in classical monocytes,
  # but removing MT- and AS (antisense) genes.
  bohn_kept_genes <- AverageExpression(bohn_sobj[, bohn_kept_cells],
    assays = "RNA", group.by = "cluster_labels_res.0.4"
  ) %>%
    as.data.frame() %>%
    filter(RNA.Classical.Monocytes > 0) %>%
    rownames() %>%
    discard(~ str_detect(.x, "^MT-|-AS[0-9]$|^RP[LS]"))

  # Expression matrix will be used for training.
  bohn_sub_sobj <- bohn_sobj[bohn_kept_genes, bohn_kept_cells]

  # Remove the object to free some memories, if RAM is the limit.
  rm(bohn_sobj)
  gc()

  # Condition per donor
  bohn_cmap_per_donor <- bohn_sample_summary %>%
    ungroup() %>%
    select(donor, condition) %>%
    deframe()

  # Worst condition per cell
  bohn_cmap_per_cell <- bohn_sub_sobj@meta.data %>%
    as.data.frame() %>%
    mutate(
      cellbarcodes = rownames(.), worst_condition = bohn_cmap_per_donor[donor]
    ) %>%
    filter(sampleID %in% bohn_kept_samples) %>%
    select(cellbarcodes, worst_condition) %>%
    deframe()

  # Meta data for all kept cells
  bohn_kept_meta <- c(
    "age", "sex", "sampleID", "donor", "cluster_labels_res.0.4", "cellbarcodes"
  )
  bohn_sub_meta <- bohn_sub_sobj@meta.data %>%
    mutate(cellbarcodes = rownames(.)) %>%
    select(one_of(bohn_kept_meta)) %>%
    rename(
      c(
        "Age" = "age", "Gender" = "sex", "SampleID" = "sampleID",
        "PatientID" = "donor", "celltypeL0" = "cluster_labels_res.0.4"
      )
    )

  # Expression matrix for training.
  bohn_sub_sobj@assays$RNA@data %>%
    as.data.frame() %>%
    t() %>%
    as.data.frame() %>%
    mutate(
      cellbarcodes = rownames(.),
      label = bohn_cmap_per_cell[cellbarcodes]
    ) %>%
    full_join(bohn_sub_meta, by = "cellbarcodes") %>%
    relocate(label, .after = last_col()) %>%
    fwrite(file.path(out_dir, "bohn_covid19_scrna-seq.csv"))
}



#
## Berlin cohort
#
if (TRUE) {
  berlin_sobj_file <- file.path(in_dir, "scRNA-seq", "cohort1.annote.rds")
  berlin_sobj <- base::readRDS(berlin_sobj_file) %>%
    subset(subset = min_nfeature <= nFeature_RNA
           & min_ncount <= nCount_RNA
           & percent.mt <= min_mt_pct)

  berlin_sobj@meta.data %>%
    colnames() %>%
    paste0(collapse = "\n") %>%
    cat("\n")

  # Meta-information from
  berlin_meta <- tribble(
    ~patien_id, ~who_per_sample, ~sampling_days_after_symptome,
    "C19-CB-01", 3, "7,11,16",
    "C19-CB-02", 3, "8,13",
    "C19-CB-03", 3, "13,18",
    "C19-CB-05", 3, "15,20",
    "C19-CB-08", 7, "13,20",
    "C19-CB-09", 7, "9,16",
    "C19-CB-012", 7, "9,16",
    "C19-CB-013", 7, "8,15",
  )

  berlin_kept_mt_cols <- c(
    "patien_id", "sample_id", "Sampling_date", "Gender", "Age"
  )
  sample_info <- berlin_sobj@meta.data %>%
    select(one_of(berlin_kept_mt_cols)) %>%
    unique() %>%
    filter(!is.na(Sampling_date)) %>%
    mutate(
      Sampling_date = as.Date(
        Sampling_date,
        format = "%d.%m.%Y", origin = "%d-%m-%Y"
      )
    ) %>%
    group_by(patien_id, Gender) %>%
    filter(n() > 1) %>% # Removing donors who only have one sample.
    arrange(Sampling_date) %>% # Sort the records by sampling date.
    left_join(berlin_meta, by = "patien_id") %>%
    summarise(
      WHO_Classification = who_per_sample[1],
      sampling_day = paste0(Sampling_date, collapse = ","),
      sample_id = paste0(sample_id, collapse = ","),
      # Using the maximum WHO score as the response variable.
      response_var = max(who_per_sample),
      # Using the earliest sample as the predictor statement.
      predictor_var_id = patien_id[1],
    ) %>%
    (function(d) {
      fwrite(d, str_glue("{out_dir}/berlin_sample_info_summary.csv"))
      return(d)
    })


  # Samples should be include
  # We include samples with the lowest WHO score
  berlin_kept_samples <- sample_info %$% predictor_var_id


  # Cells should be included
  kept_cells <- berlin_sobj@meta.data %>%
    filter(patien_id %in% berlin_kept_samples) %>%
    rownames()

  # A summary for all samples.
  berlin_sample_summary <- berlin_sobj[, kept_cells]@meta.data %>%
    group_by(patien_id, sample_id, Gender) %>%
    left_join(berlin_meta, by = "patien_id") %>%
    summarise(n_cells = n(), max_WHO_score = max(who_per_sample)) %>%
    mutate(condition = if_else(as.integer(max_WHO_score) >= 5, "S", "M"))


  # Here, we only select genes expressed in classical monocytes,
  # but removing MT-, AS (antisense) and ribosomal (RP[LS].*) genes.
  berlin_kept_genes <- AverageExpression(berlin_sobj[, kept_cells],
    assays = "RNA", group.by = "celltype"
  ) %>%
    as.data.frame() %>%
    filter(RNA.Classical.Monocytes > 0) %>%
    rownames() %>%
    discard(~ str_detect(.x, "^MT-|-AS[0-9]$|^RP[LS]"))


  # Expression matrix will be used for training.
  berlin_sub_sobj <- berlin_sobj[berlin_kept_genes, kept_cells]

  # Condition map
  berlin_cmap_per_donor <- berlin_sample_summary %>%
    ungroup() %>%
    select(patien_id, condition) %>%
    deframe()


  # Worst condition per cell
  berlin_cmap_per_cell <- berlin_sub_sobj@meta.data %>%
    as.data.frame() %>%
    mutate(
      cellbarcodes = rownames(.),
      worst_condition = berlin_cmap_per_donor[patien_id]
    ) %>%
    filter(patien_id %in% berlin_kept_samples) %>%
    select(cellbarcodes, worst_condition) %>%
    deframe()


  # Meta data for all kept cells
  berlin_kept_meta <- c(
    "Age", "Gender", "sample_id", "patien_id", "cellbarcodes", "celltypeL0"
  )
  berlin_sub_meta <- berlin_sub_sobj@meta.data %>%
    mutate(cellbarcodes = rownames(.)) %>%
    select(one_of(berlin_kept_meta)) %>%
    rename(c("SampleID" = "sample_id", "PatientID" = "patien_id"))

  # Expression matrix for training.
  berlin_sub_sobj@assays$RNA@data %>%
    as.data.frame() %>%
    t() %>%
    as.data.frame() %>%
    mutate(
      cellbarcodes = rownames(.),
      label = berlin_cmap_per_cell[cellbarcodes]
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, digits = 4))) %>%
    full_join(berlin_sub_meta, by = "cellbarcodes") %>%
    relocate(label, .after = last_col()) %>%
    fwrite(file.path(out_dir, "berlin_covid19_scrna-seq.csv"))
}
