# Data Mining Assignment 1: Frequent Itemset Mining

This repository contains the implementation for Lab Assignment #1 of the **Data Mining (Fall 2024)** course at **NYCU**.

## 📁 Files
- `Dataset/` — Folder containing the test datasets used in experiments
- `apriori.py` — Implementation of Apriori algorithm for:
  - Frequent Itemsets
  - Closed Frequent Itemsets
- `fpgrowth.py` — FP-Growth implementation using `mlxtend`
- `ItemsetVerifier.py` — Script to verify mining results


## 🚀 How to Run

### Setup
Install required package:
```
pip install mlxtend
```

### Step 2: Apriori
Run Apriori to mine frequent itemsets and closed frequent itemsets:
```
python apriori.py <input_file> <min_support> <task>
```
- `<task>`: `1` for frequent itemsets, `2` for closed frequent itemsets
  
For example: Running Apriori on datasetA.txt with min_support = 0.02 to find frequent itemsets
```
python apriori.py Dataset/datasetA.txt 0.02 1
```

### Step 3: FP-Growth
Run FP-Growth using mlxtend:
```
python fpgrowth.py <input_file> <min_support>
```
For example: Running fpgrowth on datasetA.txt with min_support = 0.02 to find frequent itemsets
```
python fpgrowth.py Dataset/datasetA.txt 0.02
```


## 🔍 Summary of Modifications

### Apriori Enhancements
- **Input handling**: Stripped input using `line.split(" ")[3:]`
- **Stat tracking**: Counted pre/post-pruning stats
- **Closed itemsets**: Added `runApriori_closed()` with closed-check logic
- **Output formatting**: Sorted by descending support, removed single quotes

### FP-Growth Highlights
- Used `mlxtend.frequent_patterns.fpgrowth` for efficient frequent itemset mining
- Applied `TransactionEncoder` and transformed data into pandas DataFrame
- Results sorted by descending support

## 📈 Observations
- Lower `min_support` leads to exponentially higher runtime
- Closed itemsets are fewer in number but cost more time to compute
- FP-Growth significantly outperforms Apriori, especially on large datasets
