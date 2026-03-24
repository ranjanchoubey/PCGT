#!/usr/bin/env python3
"""Quick check: which datasets are present after download_data.sh?"""
import os

data = "data"
checks = [
    ("cora", f"{data}/Planetoid/cora/raw/ind.cora.x"),
    ("citeseer", f"{data}/Planetoid/citeseer/raw/ind.citeseer.x"),
    ("pubmed", f"{data}/Planetoid/pubmed/raw/ind.pubmed.x"),
    ("film", f"{data}/geom-gcn/film/out1_graph_edges.txt"),
    ("deezer", f"{data}/deezer/deezer-europe.mat"),
    ("squirrel", f"{data}/wiki_new/squirrel/squirrel_filtered.npz"),
    ("chameleon", f"{data}/wiki_new/chameleon/chameleon_filtered.npz"),
    ("ogbn-arxiv", f"{data}/ogb/ogbn_arxiv/processed/data_processed"),
    ("ogbn-proteins", f"{data}/ogb/ogbn_proteins/processed/data_processed"),
    ("ogbn-products", f"{data}/ogb/ogbn_products/processed/data_processed"),
    ("pokec", f"{data}/pokec/pokec.mat"),
]

ok = fail = 0
for name, path in checks:
    if os.path.exists(path):
        sz = os.path.getsize(path)
        print(f"  [OK]      {name:20s}  {sz:>12,} bytes")
        ok += 1
    else:
        print(f"  [MISSING] {name}")
        fail += 1

print(f"\nTotal: {ok} OK, {fail} MISSING")
