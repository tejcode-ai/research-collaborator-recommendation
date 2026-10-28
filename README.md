# Research Collaborator Recommendation

A beginner-friendly Social Network Analysis AAT project, independent of the existing CreatorLens application.

The analysis, executed notebook, app, report, PDF slides and data exports are complete. PPTX export remains blocked by the unavailable prescribed artifact runtime. Native browser interaction and Cytoscape rendering were unavailable. See `outputs/CHECKLIST.md` for exact completion status and `outputs/verification.json` for executed checks.

## Setup (Windows PowerShell)

Install Python 3.12 if needed. From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe analysis.py
.\.venv\Scripts\python.exe build_deliverables.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The original prepared workspace contains a ready `.venv`. A GitHub clone or downloaded ZIP excludes the environment, so first complete the setup commands above. Browse to http://localhost:8501. Stop it with Ctrl+C.

On macOS/Linux replace `.venv\Scripts\python.exe` with `.venv/bin/python` and create the environment with `python3.12 -m venv .venv`.

## Notebook

Open `outputs/Research_Collaborator_Recommendation.ipynb` in VS Code or Jupyter and select this project's `.venv` kernel. Run all cells. The notebook runs the shared pipeline end to end and does not require Streamlit. To install a notebook UI if needed: `python -m pip install jupyterlab`, then `python -m jupyterlab`. For automated execution use `python build_deliverables.py`.

## Folder guide

- `analysis.py`: download, cleaning, graph statistics, training-only evaluation, exact full-network rankings, figures and offline HTML.
- `app.py`: six-section Streamlit explorer using the same functions and CSV results.
- `build_deliverables.py`: executed notebook, formatted report, PDF slides, presentation script and viva.
- `verify.py`: data integrity, reproducibility, app interaction and artifact checks.
- `outputs/dataset/`: preserved original gzip, clean edges, nodes, primary split, negative samples, GraphML and source/schema README.
- `outputs/figures/`: PNG/SVG charts and bounded network visualizations.
- `outputs/evaluation.csv`: primary results, seed 42.
- `outputs/sensitivity.csv` and `sensitivity_summary.csv`: seeds 42, 7, 21, 84.
- `outputs/evaluation_pairs.csv`: exact scored evaluation candidates and labels.
- `outputs/recommendations_top10.csv` and `recommendations_top20.csv`: one ranking per algorithm.
- `outputs/summary.json`: computed statistics, seeds, cleaning counts and archive checksum.
- `outputs/network.html`: self-contained offline interactive network.
- `outputs/evaluation_report.pdf`, `presentation.pdf`, and `presentation_script_and_viva.md`: submission and presentation materials.
- `outputs/CHECKLIST.md` and `verification.json`: completion status, checks and limitations.

## Reproducibility and interpretation

Primary seed 42. Additional fixed seeds 7, 21 and 84. Training uses floor(0.8 times unique edges), with all original nodes retained. One sampled full-graph nonedge per held-out edge. All methods use identical evaluation pairs and training-only scores. Ties: descending score then ascending canonical IDs. For personal recommendations, the selected ID remains the first column, while tie comparison uses canonical IDs.

The strongest method is selected by primary ROC-AUC, with alphabetical method names resolving ties. This is descriptive selection on the same test results, not independent validation. Recall@K and F1@K are measured at K. Sampled nonedges are assumed negatives. Balanced sampled precision differs from all-missing-link precision. A random split does not establish future collaboration prediction.

Global recommendations separately score every unordered nonedge on the full observed graph. Four fixed heaps retain 20 candidates per method. No candidate filtering is used. Exact ranking may take a minute or more on a slow machine. Researcher suggestions scan all possible partners for the selected ID. Scores are structural similarities, not probabilities or compatibility assessments.

## Sources

Dataset: https://snap.stanford.edu/data/ca-GrQc.html

J. Leskovec, J. Kleinberg and C. Faloutsos (2007). Graph Evolution: Densification and Shrinking Diameters. ACM TKDD 1(1).

D. Liben-Nowell and J. Kleinberg. The Link Prediction Problem for Social Networks. Author manuscript (2004): https://www.cs.cornell.edu/home/kleinber/link-pred.pdf. Journal version: JASIST 58(7), 2007.

NetworkX method definitions: https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html

See `outputs/dataset/README.md` for preprocessing and Cytoscape import instructions. No researcher names, affiliations or research interests are supplied or invented.
