# Deliverables and verification

## Completed

- [x] Real SNAP ca-GrQc archive downloaded and preserved, with SHA256 and citation.
- [x] Clean full graph: 5,242 nodes, 14,484 nonloop unique edges, including one isolated node.
- [x] Node/edge CSVs, original archive, GraphML, primary train/test edges and distinct negative pairs.
- [x] Primary seed 42 and sensitivity seeds 7, 21 and 84, with connectivity statistics.
- [x] Four custom score implementations checked against NetworkX on every evaluation pair.
- [x] Training-only evaluation using identical balanced candidate pairs, deterministic ties, AUC and P/R/F1 at 10 and 20.
- [x] Exact global top 10 and top 20 for each method from all 13,722,177 full-graph nonedges.
- [x] Structural explanations using actual shared collaborator IDs.
- [x] Executed end-to-end notebook with saved outputs.
- [x] PNG/SVG degree, network overview, recommended links, comparison and ROC figures.
- [x] Self-contained interactive HTML and Cytoscape import files/instructions.
- [x] Ten-page formatted evaluation/recommendation PDF, rendered and visually inspected page by page.
- [x] Twelve-slide presentation PDF, rendered and visually inspected slide by slide.
- [x] Streamlit source with six sections, caching, bounded networks, researcher selectors and CSV downloads.
- [x] Server launched successfully at http://127.0.0.1:8501 and health endpoint returned `ok`.
- [x] Streamlit AppTest checked all six sections, each algorithm, top 10/20, researcher changes, download controls and CSV round-trip payloads.
- [x] Reusable Python sources, tested pinned dependencies, README, startup script and application archive.
- [x] Presentation script, live demo sequence and 25 concise viva answers.
- [x] Automated data/reproducibility/artifact checks saved in verification.json.

## Remaining limitations

- [ ] **PPTX export is unavailable.** The prescribed presentation skill requires the bundled `@oai/artifact-tool` runtime. Its dependency loader is absent and importing the package reports `Module not found`. The skill prohibits installing substitute presentation dependencies. No `.pptx` file was created. `presentation.pdf` and `slides_content.json` preserve the 12-slide presentation content.
- [ ] **Native browser interaction checks remain manual.** The computer-use inventory returned no browsers. Creating an in-app browser returned `Browser is not available: iab`. AppTest verifies application logic and widgets but does not verify JavaScript zoom/pan, visual browser rendering or native download dialogs. Open the app locally and follow the demo sequence to check these.
- [ ] **No Cytoscape render.** Cytoscape was not available on PATH. Python PNG/SVG figures and interactive HTML were produced. Import instructions accompany the CSV/GraphML exports.

## Interpretation

Adamic-Adar has the highest primary AUC, 0.8889447251, with only a small lead over the other neighborhood methods. All four methods have primary Precision@10 and Precision@20 of 1.0 on the balanced sampled set. This does not establish real-world all-pair precision or future prediction. The primary test set also selects the winner, so there is no independent validation of that selection.

The initial pandas 3.1.0 wheel encountered a Windows Application Control DLL block. Replacing it with pandas 2.2.3 resolved execution. The tested requirements pin pandas 2.2.3. No system security policy was changed.
