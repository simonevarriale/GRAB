"""Download all GRAB datasets from HuggingFace and save to disk.

Run from the project root (DATA_DIR must be set in .env):
    python download_datasets.py

Datasets handled here
---------------------
Automatic (HF load_dataset + save_to_disk):
  wtq            wikitablequestions
  wikisql        wikisql
  hitab          kasnerz/hitab
  hctqa          qcri-ai/HCTQA
  atis           vaishali/atis-tableQA
  geoquery       vaishali/geoQuery-tableQA
  spider         vaishali/spider-tableQA   (used by spider_qa + spider_text2sql)

Manual (raw files — see comments below):
  tabmwp         JSON files from the TabMWP GitHub repo
  multihiertt    JSON files from the MultiHierTT GitHub repo
  scitat         JSON files from https://github.com/zhxlia/SciTaT/tree/main/dataset
  tqa_bench      SQLite + domain folders from the TableBench release
  tabfact        JSON + CSV files from the TabFact GitHub repo
  mmqa           https://drive.google.com/drive/folders/1XQ9djKSK4yjxLWAmHMzsyPAKMqdCIpXo?usp=drive_link
  structProbe    https://github.com/liyaooi/TAMO/tree/main/dataset/structProbe/structProbe
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets
from src.global_path import data_dir

print(f"DATA_DIR = {data_dir}\n")


# ---------------------------------------------------------------------------
# wtq  →  {data_dir}/wtq/wikitablequestions
# ---------------------------------------------------------------------------
print("Downloading wikitablequestions (wtq)...")
wtq = datasets.load_dataset('wikitablequestions', revision='refs/convert/parquet')
wtq.save_to_disk(f'{data_dir}/wtq/wikitablequestions')
print("  saved.\n")


# ---------------------------------------------------------------------------
# wikisql  →  {data_dir}/wikisql/wikisql
# NOTE: answers.json is NOT included in the HF dataset — it must come from
#       data.zip or be generated separately.
# ---------------------------------------------------------------------------
print("Downloading wikisql... ATTENTION: extract from data.zip the answers.json to add in the folder (./data/wikisql/answers.json)")
wikisql = datasets.load_dataset('wikisql', revision='refs/convert/parquet')
wikisql.save_to_disk(f'{data_dir}/wikisql/wikisql')
print("  saved.\n")


# ---------------------------------------------------------------------------
# hitab  →  {data_dir}/hitab/hitab
# ---------------------------------------------------------------------------
print("Downloading hitab...")
hitab = datasets.load_dataset('kasnerz/hitab')
hitab.save_to_disk(f'{data_dir}/hitab/hitab')
print("  saved.\n")


# ---------------------------------------------------------------------------
# hctqa  →  {data_dir}/hctqa/hctqa
# NOTE: the stress_test split (used by hctqa_stress_test.py) is NOT in the
#       main HF repo — include it from data.zip.
# ---------------------------------------------------------------------------
print("Downloading hctqa (qcri-ai/HCTQA)...")
hctqa = datasets.load_dataset('qcri-ai/HCTQA')
hctqa.save_to_disk(f'{data_dir}/hctqa/hctqa')
print("  saved.\n")


# ---------------------------------------------------------------------------
# atis  →  {data_dir}/atis/atis
# ---------------------------------------------------------------------------
print("Downloading atis (vaishali/atis-tableQA)...")
atis = datasets.load_dataset('vaishali/atis-tableQA')
atis.save_to_disk(f'{data_dir}/atis/atis')
print("  saved.\n")


# ---------------------------------------------------------------------------
# geoquery  →  {data_dir}/geoquery/geoquery
# ---------------------------------------------------------------------------
print("Downloading geoquery (vaishali/geoQuery-tableQA)...")
geoquery = datasets.load_dataset('vaishali/geoQuery-tableQA')
geoquery.save_to_disk(f'{data_dir}/geoquery/geoquery')
print("  saved.\n")


# ---------------------------------------------------------------------------
# spider  →  {data_dir}/spider/spider
# Used by both spider_qa and spider_text2sql dataset loaders.
# ---------------------------------------------------------------------------
print("Downloading spider (vaishali/spider-tableQA)...")
spider = datasets.load_dataset('vaishali/spider-tableQA')
spider.save_to_disk(f'{data_dir}/spider/spider')
print("  saved.\n")


# ---------------------------------------------------------------------------
# Manual downloads — print instructions
# ---------------------------------------------------------------------------
print("=" * 72)
print("The following datasets require manual download:\n")

print("tabmwp  →  {data_dir}/tabmwp/tabmwp/")
print("  Files needed: problems_train.json, problems_dev.json, problems_test.json")
print("  Source: https://github.com/lupantech/PromptPG (data/ folder)\n")

print("multihiertt  →  {data_dir}/multihiertt/multihiertt/")
print("  Files needed: train.json, dev.json")
print("  Source: https://github.com/psunlpgroup/MultiHiertt (data/ folder)\n")

print("scitat  →  {data_dir}/scitat/scitat/")
print("  Files needed: scitat_train.json, scitat_dev.json, scitat_test.json")
print("  Source: https://github.com/zhxlia/SciTaT/tree/main/dataset\n")

print("tqa_bench  →  {data_dir}/tqa_bench/tqa_bench/")
print("  Files needed: dataset.sqlite + domain subdirs (8k/, …)")
print("  Source: https://github.com/Relaxed-System-Lab/TQA-Bench \n")

print("structProbe  →  {data_dir}/structProbe/structProbe/")
print("  Source: https://github.com/liyaooi/TAMO/tree/main/dataset/structProbe/structProbe\n")

print("tabfact  →  {data_dir}/tabfact/tabfact/")
print("  Files needed: train_examples.json, validation_examples.json,")
print("                test_examples.json, all_csv/")
print("  Source: https://github.com/wenhuchen/Table-Fact-Checking\n")

print("mmqa  →  {data_dir}/mmqa/mmqa/")
print("  Source: https://drive.google.com/drive/folders/1XQ9djKSK4yjxLWAmHMzsyPAKMqdCIpXo?usp=drive_link\n")

print("hctqa stress_test split  →  {data_dir}/hctqa/hctqa/stress_test/")
print("  Included in the data/ folder of this repository\n")

print("wikisql answers.json  →  {data_dir}/wikisql/answers.json")
print("  Included in the data/ folder of this repository\n")

print("All manual datasets done.")
