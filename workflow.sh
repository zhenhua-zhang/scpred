#!/usr/bin/env bash
# Author: Zhenhua Zhang
# E-mail: zhenhua.zhang217@gmail.com
# Created: Apr 07, 2022
# Updated: Apr 11, 2022

wk_dir=~/Documents/projects/wp_codex
source $wk_dir/scripts/.env/bin/activate


declare -A CellTypeBohn=(
  ["B"]="B cells"
  ["NK"]="NK cells"
  ["CD4T"]="CD4+ T cells"
  ["CD8T"]="CD8+ T cells"
  ["Neut"]="Neutrophils"
  ["Mega"]="Megakaryocytes"
  ["cMono"]="Classical Monocytes"
  ["ncMono"]="Non-classical Monocytes"
  ["mDC"]="mDC"
  ["pDC"]="pDC"
  ["PlaBla"]="Plasmablasts"
# ["ImmNeut"]="Immature Neutrophils"
)
# Activated T cells
# HLA-DRhi CD83hi Monocytes
# HLA-DRlo CD163hi Monocytes
# HLA-DRlo S100Ahi Monocytes
# Progenitors
# Prol. cells


declare -A CellTypeBerlin=(
  ["B"]="B"
  ["NK"]="NK"
  ["CD4T"]="CD4+ T"
  ["CD8T"]="CD8+ T"
  ["Neut"]="Neutrophils"
  ["Mega"]="Megakaryocytes"
  ["cMono"]="CD14+ Monocytes"
  ["ncMono"]="CD16+ Monocytes"
  ["mDC"]="mDCs"
  ["pDC"]="pDCs"
  ["PlaBla"]="Plasmablasts"
)
# CellTypeBerlin["ImmNeut"]="Immature Neutrophils"
# Prol. T


tar_ct=cMono
train_on=bohn
test_on=berlin
test_ratio=0.5
n_rscv_iters=50
save_to=$wk_dir/outputs/$train_on/$tar_ct

if [[ $train_on == bohn ]]; then
  if [[ $tar_ct == cMono ]]; then
    # Train a model using Bohn samples
    python $wk_dir/scripts/scpred.py -P $save_to train \
      -i $wk_dir/outputs/${train_on}_covid19_scrna-seq_exPatientId.csv \
      -c $wk_dir/scripts/config.json \
      -p $test_ratio \
      -I $n_rscv_iters \
      -t "${CellTypeBohn["$tar_ct"]}" "HLA-DRhi CD83hi Monocytes" "HLA-DRlo CD163hi Monocytes"

    python $wk_dir/scripts/scpred.py -P $save_to explain \
      -i $wk_dir/outputs/${train_on}_covid19_scrna-seq_exPatientId.csv \
      -t "${CellTypeBohn["$tar_ct"]}" "HLA-DRhi CD83hi Monocytes" "HLA-DRlo CD163hi Monocytes"
  else
    python $wk_dir/scripts/scpred.py -P $save_to train \
      -i $wk_dir/outputs/${train_on}_covid19_scrna-seq_exPatientId.csv \
      -c $wk_dir/scripts/config.json \
      -p $test_ratio \
      -I $n_rscv_iters \
      -t "${CellTypeBohn["$tar_ct"]}"

    python $wk_dir/scripts/scpred.py -P $save_to explain \
      -i $wk_dir/outputs/${train_on}_covid19_scrna-seq_exPatientId.csv \
      -t "${CellTypeBohn["$tar_ct"]}"
  fi


  # Predict unseen samples, i.e., Berlin cohorts
  # Make sure the way to encode category variables. Such as age group vs age
  python $wk_dir/scripts/scpred.py -P $save_to predict \
    -i $wk_dir/outputs/${test_on}_covid19_scrna-seq_exPatientId.csv \
    -t "${CellTypeBerlin["$tar_ct"]}" \
    -o $save_to/Predict/$test_on
else
  python $wk_dir/scripts/scpred.py -P $save_to train \
    -i $wk_dir/outputs/${train_on}_covid19_scrna-seq_exPatientId.csv \
    -c $wk_dir/scripts/config.json \
    -p $test_ratio \
    -t "${CellTypeBerlin["$tar_ct"]}"

  python $wk_dir/scripts/scpred.py -P $save_to explain \
    -i $wk_dir/outputs/${train_on}_covid19_scrna-seq_exPatientId.csv \
    -t "${CellTypeBerlin["$tar_ct"]}"

  if [[ $tar_ct == cMono ]]; then
    python $wk_dir/scripts/scpred.py -P $save_to predict \
      -i $wk_dir/outputs/${test_on}_covid19_scrna-seq_exPatientId.csv \
      -t "${CellTypeBohn["$tar_ct"]}" "HLA-DRhi CD83hi Monocytes" "HLA-DRlo CD163hi Monocytes" \
      -o $save_to/Predict/$test_on
  else
    python $wk_dir/scripts/scpred.py -P $save_to predict \
      -i $wk_dir/outputs/${test_on}_covid19_scrna-seq_exPatientId.csv \
      -t "${CellTypeBohn["$tar_ct"]}" \
      -o $save_to/Predict/$test_on
  fi

fi
