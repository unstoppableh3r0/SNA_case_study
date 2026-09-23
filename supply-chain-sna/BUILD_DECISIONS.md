# BUILD_DECISIONS.md
# Architectural and Methodological Decisions

## Graph Representation
- **Decision**: Use `networkx.DiGraph` as primary graph representation
- **Rationale**: Supply chains have directional flow (Supplier → Manufacturer → Distributor → Retailer). Direction is meaningful and must be preserved.
- **Weight semantics**: Multiple weight modes supported:
  - `frequency` = transaction count between node pairs
  - `quantity` = total quantity exchanged
  - `transaction_value` = total monetary value

## Temporal Snapshots
- **Decision**: Monthly snapshots, accumulated (not sliding window)
- **Rationale**: Each snapshot includes all transactions up to and including that month. This creates cumulative relationship graphs that reflect the network state at each point.
- **Alternative considered**: Sliding 3-month windows — rejected for simplicity without a specific use-case requirement.

## Community Detection Algorithm
- **Decision**: Louvain as primary (on undirected projection of DiGraph)
- **Rationale**: Louvain is widely used, fast, and produces good modularity. Greedy modularity provided as secondary option.
- **Note**: Community detection applied on undirected version of graph since Louvain requires undirected input. Documented in methodology.

## Closeness Centrality on Disconnected Graphs
- **Decision**: Use NetworkX `closeness_centrality` with default Wasserman-Faust normalization
- **Rationale**: This handles disconnected graphs by computing within connected components. Explicitly documented.

## Eigenvector Centrality Convergence
- **Decision**: Use `max_iter=1000`, `tol=1e-6`, fallback to PageRank if convergence fails
- **Rationale**: Directed graphs may have convergence issues. PageRank is a valid alternative for measuring importance in directed networks.

## Resilience Random Removal
- **Decision**: Run 10 seeds, report mean ± std
- **Rationale**: Single random run is statistically misleading. Multiple seeds provide reliable estimate.

## Synthetic Data Realism
- **Decision**: Use hierarchical generation with power-law degree heterogeneity
- **Rationale**: Real supply chains have hub organizations with many connections and peripheral organizations with few. Uniform random generation would not produce this.

## Decentralized Layer
- **Decision**: Implement `SyntheticTransactionStore` + abstract `TransactionStore` interface. No real blockchain required.
- **Rationale**: Real blockchain (Hyperledger/Ethereum) would significantly increase build complexity without improving SNA research objective. The abstraction allows future integration.

## Betweenness Centrality
- **Decision**: Use normalized betweenness on directed graph
- **Rationale**: Normalization allows comparison across different network sizes. Directed betweenness correctly identifies intermediaries in directional flow.

## K-Core Analysis
- **Decision**: Run k-core on undirected projection
- **Rationale**: NetworkX k-core decomposition requires undirected graphs. The undirected projection preserves connectivity information needed for core analysis.
