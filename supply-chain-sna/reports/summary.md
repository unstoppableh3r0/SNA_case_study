# Supply Chain Social Network Analysis — Research Summary

**Generated**: 2026-09-23 10:00 UTC  
**Seed**: 42  
**Dataset Version**: v1  

---

## 1. Dataset Statistics

- **Nodes**: 1000
- **Edges**: 7517
- **Density**: 0.007525
- **Weakly Connected Components**: 1
- **Largest WCC Fraction**: 1.0000
- **Average Total Degree**: 15.03
- **Avg Clustering (undirected)**: 0.0186
- **Global Efficiency**: 0.37617047046835733

---

## 2. Centrality Findings

### Top 3 by Total Degree
- MFG-00131: 74.000000
- MFG-00028: 63.000000
- MFG-00141: 62.000000
### Top 3 by Betweenness
- MFG-00131: 0.019171
- WHS-00012: 0.018229
- WHS-00068: 0.012974
### Top 3 by Pagerank
- RET-00032: 0.003657
- RET-00029: 0.003577
- WHS-00002: 0.003550

### Rank Correlations (Spearman)
- total_degree_vs_betweenness: r = 0.7753
- total_degree_vs_pagerank: r = 0.7461
- betweenness_vs_pagerank: r = 0.5775
- eigenvector_vs_pagerank: r = 0.9657
- betweenness_vs_closeness: r = 0.4823

---

## 3. Community Detection

- **Algorithm**: louvain
- **Communities Detected**: 16
- **Modularity**: 0.25243513995817374
- **Inter-community Edge Fraction**: 0.7501

---

## 4. Dependency Analysis

- **Nodes with >80% upstream concentration**: 90
- **Mean supplier dependency ratio**: 0.2944

---

## 5. Temporal Analysis

- **Months analyzed**: 24
- **Node count range**: 1000–1000
- **Edge count range**: 4312–5000
- **Density trend**: 0.005005 → 0.004316

---

## 6. Resilience Findings

- **Random removal at 10%**: LCC = 0.9998
- **Degree removal at 10%**: LCC = 0.9989
- **Betweenness removal at 10%**: LCC = 0.9978
- **Pagerank removal at 10%**: LCC = 1.0000

---

## 7. Ground-Truth Evaluation

- **Bridge recovery rate**: 0.0
- **Community ARI**: None
- **Community NMI**: None
- **Efficiency change after removing critical nodes**: -0.003845701083428188

---

## 8. Limitations

- Dataset is synthetic and not equivalent to real enterprise supply-chain data
- Synthetic generation rules influence network structure and SNA findings
- SNA identifies structural patterns but does not establish causation
- Centrality does not automatically imply business importance
- Community detection results depend on algorithm choice and graph representation
- Large-graph metrics (path length, efficiency) may use sampling approximations
- Blockchain/decentralized layer is abstracted, not a live implementation