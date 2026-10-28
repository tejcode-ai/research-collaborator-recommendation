# Dataset package
Source: https://snap.stanford.edu/data/ca-GrQc.html
Download: https://snap.stanford.edu/data/ca-GrQc.txt.gz
Citation: Leskovec, Kleinberg and Faloutsos. Graph Evolution: Densification and Shrinking Diameters. ACM TKDD 1(1), 2007.
Original gzip preserved byte-for-byte. SHA256: a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230
Integer IDs only. Raw rows: 28980; removed self-loop rows: 12; duplicate/reverse rows: 14484.
cleaned_edges.csv: source,target, one row per unordered observed edge (source < target).
nodes.csv: id,label,degree. GraphML preserves all observed nodes. Load nodes.csv as well as the edges to retain the one node isolated after loop removal.
train_edges.csv and test_edges.csv: primary seed 42, floor(0.8*M) train edges. Training retains all nodes.
negative_pairs.csv: distinct unordered pairs absent from the full graph, equal in count to test edges.
Seeds: 42 primary; 7,21,84 sensitivity. Python random.Random samples sorted IDs and shuffled sorted edges.
visualization_edges.csv has edge_type observed/predicted and only the explained visualization subset.
For Cytoscape: import observed_network.graphml, or nodes.csv then cleaned_edges.csv as source/target edges. For the bounded view import nodes.csv and visualization_edges.csv, map edge_type to gray solid / orange dashed. No Cytoscape render was produced because Cytoscape was unavailable on PATH.
