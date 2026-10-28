"""Build notebook, report, PDF slides, and student notes from executed results."""
from pathlib import Path
import json, sys, os
import nbformat as nbf
from nbclient import NotebookClient
import pandas as pd
import networkx as nx
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from xml.sax.saxutils import escape
from analysis import ROOT, OUT, DATA, FIG, METHODS, EXPLANATIONS, FORMULAS, LIMITATIONS, load_graph

def notebook():
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.jupyter_runtime')
    os.environ['IPYTHONDIR']=str(ROOT/'.ipython')
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    cells=[md('# Research Collaborator Recommendation\nSocial Network Analysis AAT. This notebook executes the complete shared pipeline independently of Streamlit. Use Python 3.12 with requirements.txt.\n\nSource: https://snap.stanford.edu/data/ca-GrQc.html. Nodes are dataset researcher IDs, edges are observed co-authorships. No names or interests are supplied.'),
    code("from pathlib import Path\nimport sys\n# Support launching from the project folder or outputs folder.\nproject = Path.cwd() if (Path.cwd() / 'analysis.py').exists() else Path.cwd().parent\nsys.path.insert(0, str(project))\nfrom analysis import *\nfrom IPython.display import display, Image, HTML\nsummary = run()"),
    md('## Data preparation and graph model\nThe original gzip is preserved. Remove self-loops and duplicate unordered edges. Retain every node in the full simple undirected unweighted graph. Graph statistics below are computed, not copied from the source page.'),
    code("g=load_graph()\ndisplay(pd.DataFrame([summary['full']]))\nprint('Raw rows:', summary['raw_rows'], 'Self-loop rows:',summary['self_loop_rows'], 'Duplicate/reverse rows:',summary['duplicate_or_reverse_rows'])\nprint('Original SHA256:',summary['sha256'])"),
    md('## Reproducible evaluation\nPrimary seed 42. Sort unique edges, shuffle with Python random.Random, retain floor(0.8 M) edges for training and hold out the remainder. Add all original nodes before adding training edges. Negatives are distinct full-graph nonedges with no self-pairs, sampled 1:1 with positives. All methods use exactly the same evaluation pairs. Test edges enter labels only, never scoring or tuning.'),
    code("display(pd.DataFrame(summary['training']).T)\ntrain=pd.read_csv(DATA/'train_edges.csv')\ntest=pd.read_csv(DATA/'test_edges.csv')\nneg=pd.read_csv(DATA/'negative_pairs.csv')\nassert set(map(tuple,train.values)).isdisjoint(map(tuple,test.values))\nassert all(not g.has_edge(int(u),int(v)) for u,v in neg.values)\nprint('Primary train/test/negative counts:',len(train),len(test),len(neg))"),
    md('## Four structural scores\nLet $N(u)$ be the neighbors of researcher $u$, and $d(u)=|N(u)|$. Scores are similarities, not probabilities.\n\n'+'\n\n'.join('### '+m+'\n$$'+f+'$$\n'+e for m,f,e in zip(METHODS,FORMULAS,EXPLANATIONS))),
    code("# Every evaluation pair is checked against NetworkX during run().\nt=nx.Graph(); t.add_nodes_from(g); t.add_edges_from(train.values.tolist())\nu,v=map(int,test.iloc[0])\nprint('Example held-out pair:',u,v,'training-only scores:',dict(zip(METHODS,scores(t,u,v))))"),
    md('## Metrics and tie handling\nROC-AUC compares positive/negative ordering, with half credit for tied scores. Precision@K = true positives in the first K / K. Recall@K = true positives in the first K / all held-out positives. F1@K = 2PR/(P+R), or zero when P+R is zero. Recall and F1 are at K. Sort by descending score, then ascending unordered researcher IDs. Precision on this balanced sampled set differs from precision over all missing links.'),
    code("metrics=pd.read_csv(OUT/'evaluation.csv'); display(metrics)\nprint('Strongest by primary ROC-AUC:', summary['best_method'])\ndisplay(Image(filename=str(FIG/'model_comparison.png')))\ndisplay(Image(filename=str(FIG/'roc_curves.png')))"),
    md('## Split sensitivity\nSeeds 7, 21 and 84 repeat the complete split and negative sampling. The primary split remains seed 42. This describes sensitivity rather than an independent validation of the selected method.'),
    code("sensitivity=pd.read_csv(OUT/'sensitivity.csv'); display(sensitivity)\ndisplay(sensitivity.groupby('method')['ROC_AUC'].agg(['mean','std','min','max']))"),
    md('## Full-network recommendations\nAfter evaluation, score the complete cleaned observed network. Examine every unordered nonedge exactly, streaming candidates through four heaps of size 20. No filtering is used. Exclude self-pairs and existing links. Each method has its own deterministic ranking. These are structural possibilities.'),
    code("recommendations=pd.read_csv(OUT/'recommendations_top20.csv')\nprint('Exact candidate nonedges:',summary['candidate_non_edges'])\nfor method in METHODS:\n    print(method,'top 20 (first 10 are the top 10)')\n    display(recommendations[recommendations.method==method])\nassert all(not g.has_edge(int(u),int(v)) and u!=v for u,v in recommendations[['researcher_1','researcher_2']].values)"),
    code("for row in recommendations[recommendations.method==summary['best_method']].head(3).itertuples():\n    common=sorted(nx.common_neighbors(g,row.researcher_1,row.researcher_2))\n    print(f'{row.researcher_1} + {row.researcher_2}: score={row.score:.6g}; degrees={g.degree(row.researcher_1)}, {g.degree(row.researcher_2)}; shared IDs={common}')"),
    md('## Visualizations\nThe degree chart uses the full network. The overview uses a deterministic breadth-first subset of at most 35 nodes rooted at the lowest researcher ID with degree 3 in the largest component, with degree then ID ordering. The recommendation view shows three top pairs from the strongest primary-AUC method and up to 10 shared collaborators per pair, with induced observed edges. Gray lines are observed, orange dashed lines are predictions. Full-network analysis remains unchanged.'),
    code("for name in ['degree_distribution','network_overview','recommended_links']:\n    display(Image(filename=str(FIG/(name+'.png'))))\ndisplay(HTML((OUT/'network.html').read_text(encoding='utf-8')))"),
    md('## Application\nFrom the project directory: `python -m streamlit run app.py`. The app imports this same analysis module and reads the generated CSVs. It caches graph loading and researcher results. Explorer views are bounded. See README.md for setup and the demo sequence.'),
    md('## Limitations and conclusion\n'+LIMITATIONS+'\n\nCytoscape was unavailable on PATH. Python figures and a self-contained PyVis HTML supply equivalent views. No Cytoscape render was produced. Dataset import instructions are in dataset/README.md.\n\nReferences: Leskovec, Kleinberg and Faloutsos (2007), Graph Evolution: Densification and Shrinking Diameters, ACM TKDD 1(1); https://snap.stanford.edu/data/ca-GrQc.html; https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html; Liben-Nowell and Kleinberg (2007), The link-prediction problem for social networks, JASIST 58(7).')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    path=OUT/'Research_Collaborator_Recommendation.ipynb'; nbf.write(nb,path)
    NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,path)

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='SmallBody',fontSize=9,leading=13,spaceAfter=8))
def p(text,style='BodyText'): return Paragraph(escape(str(text)),styles[style])
def table(data,widths=None):
    t=Table([[p(x,'SmallBody') for x in row] for row in data],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#254c78')),('BOTTOMPADDING',(0,0),(-1,-1),5)])); return t
def footer(c,d):
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#65758b')); c.drawString(40,22,'Research Collaborator Recommendation | SNA AAT'); c.drawRightString(d.pagesize[0]-40,22,str(d.page))

def report(s,metrics,recs,g):
    story=[]
    def heading(t): story.extend([Spacer(1,12),p(t,'Heading1')])
    def body(t): story.append(KeepTogether([p(t),Spacer(1,7)]))
    story.extend([p('Research Collaborator Recommendation','Title'),p('Social Network Analysis AAT | Executed experiment on Stanford SNAP ca-GrQc','Heading2')])
    heading('Abstract'); body(f"This project compares four structural link prediction scores on a real research collaboration network. The cleaned graph has {len(g):,} researchers and {g.number_of_edges():,} observed co-authorship links. A reproducible random edge split evaluates missing-link recovery. {s['best_method']} has the highest primary ROC-AUC in this experiment. Separate exact rankings on the full graph suggest potential collaborator pairs. A notebook and local Streamlit app expose the same calculations.")
    heading('Problem statement and objectives'); body('Given only the observed co-authorship structure, rank unconnected researcher pairs. Objectives: construct a clean graph, compare Common Neighbors, Jaccard, Adamic-Adar and Preferential Attachment, evaluate reproducibly, and explain top 10 and top 20 structural recommendations. Scores do not measure topic compatibility.')
    heading('Dataset and preparation'); body(f"Source: Stanford SNAP ca-GrQc. The archive covers arXiv GR-QC papers from January 1993 to April 2003. Nodes are integer researcher IDs. Edges represent co-authorship, with no weights or directions. The preserved original contains {s['raw_rows']:,} rows. Cleaning removes {s['self_loop_rows']} self-loop rows and {s['duplicate_or_reverse_rows']:,} duplicate/reverse rows. All nodes remain available. SHA256 is recorded in the dataset README.")
    body('SNAP lists 14,496 edges. The downloaded file contains 12 self-loop rows, so the required simple loop-free graph has 14,484 edges after cleaning. The report consistently uses the cleaned count.')
    story.append(table([['Statistic','Full cleaned graph']]+[[k.replace('_',' '),f'{v:.6f}' if isinstance(v,float) else v] for k,v in s['full'].items()],[230,270]))
    heading('Methodology and formulas')
    for m,e,f in zip(METHODS,EXPLANATIONS,['CN(u,v) = |N(u) intersection N(v)|','J(u,v) = |N(u) intersection N(v)| / |N(u) union N(v)|','AA(u,v) = sum over shared w of 1 / ln(d(w))','PA(u,v) = d(u) * d(v)']):
        body(m+': '+e+' '+f)
    body('N(u) is the neighbor set and d(u) its size. The Jaccard score is zero for an empty union. A shared neighbor of distinct nodes has degree at least two, so Adamic-Adar is defined. Natural logarithms are used. Custom scores are checked against NetworkX on every evaluation pair, with Common Neighbors checked against nx.common_neighbors.')
    heading('Experimental setup'); body('Primary seed: 42. Sort unique unordered edges, shuffle with random.Random, take floor(80%) for training, and hold out the rest. Add all original nodes to the training graph before edges. Negatives are sampled without replacement from pairs absent from the complete graph, with no self-pairs. Use one negative per held-out positive. Scoring uses only the training graph. The full graph is consulted solely to reject false negatives. Every method sees the same evaluation pairs. No parameters are tuned.')
    t=s['training']['42']; body(f"Training has {t['edges']:,} edges and {t['nodes']:,} nodes. Holding out edges changes components from {s['full']['components']} to {t['components']}, largest component size from {s['full']['largest_component']} to {t['largest_component']}, and isolates from {s['full']['isolates']} to {t['isolates']}. Training disconnection is allowed, not repaired.")
    body('Deterministic ranking sorts by descending score then ascending smaller ID and larger ID. ROC-AUC compares positive/negative ordering and gives tied scores half credit. Precision@K = hits/K. Recall@K = hits/number of held-out positives. F1@K = 2PR/(P+R), with zero if both are zero. Recall and F1 are explicitly at K. Balanced sampled-set precision is not precision over every missing link.')
    heading('Actual primary results')
    for group in [['ROC_AUC','Precision@10','Precision@20'],['Recall@10','Recall@20','F1@10','F1@20']]:
        story.append(table([['Method']+group]+[[r['method']]+[f'{r[c]:.5f}' for c in group] for _,r in metrics.iterrows()],[140]+[360/len(group)]*len(group))); story.append(Spacer(1,12))
    body(f"Selection criterion: highest primary ROC-AUC, with alphabetical method name breaking any tie. The strongest method by this criterion is {s['best_method']}. The three neighborhood methods have nearly equal AUCs on this split. All four methods achieve Precision@10 and Precision@20 of 1.0 on the sampled set. This is a descriptive choice on the same test results, not independent validation. High Precision@K can coexist with low Recall@K because only 10 or 20 predictions cover thousands of positives. Preferential Attachment favors popular researchers; the neighborhood scores reward shared structure.")
    for name in ['model_comparison','roc_curves']:
        story.append(Image(str(FIG/f'{name}.png'),width=490,height=285,kind='proportional'))
    heading('Split sensitivity'); sensitivity=pd.read_csv(OUT/'sensitivity.csv'); agg=sensitivity.groupby('method').ROC_AUC.agg(['mean','std','min','max'])
    story.append(table([['Method','Mean AUC','SD','Min','Max']]+[[m]+[f'{x:.5f}' for x in r] for m,r in agg.iterrows()],[140,90,90,90,90])); body('Summary includes seeds 42, 7, 21 and 84. Each seed changes both the split and sampled negative set. Four runs provide a small sensitivity check, not a confidence interval or independent final test.')
    heading('Final collaborator recommendations'); body(f"After evaluation, the shared pipeline scores all {s['candidate_non_edges']:,} unordered nonedges of the full cleaned graph. It streams all pairs with four heaps of 20 records, avoiding a large all-pairs table. There is no candidate filtering or approximation. Existing links, self-links and duplicate unordered pairs are excluded. The top-10 CSV is the first ten rows per method of the top-20 CSV.")
    for m in METHODS:
        frame=recs[recs.method==m]
        story.append(KeepTogether([p(m,'Heading2'),table([['Rank','ID 1','ID 2','Score','Shared']]+[[int(r['rank']),int(r.researcher_1),int(r.researcher_2),f'{r.score:.5f}',int(r.shared_collaborators)] for _,r in frame.iterrows()],[50,80,80,180,110])]))
    heading('Structural explanations')
    for r in recs[recs.method==s['best_method']].head(3).itertuples():
        shared=sorted(nx.common_neighbors(g,r.researcher_1,r.researcher_2))
        body(f"IDs {r.researcher_1} and {r.researcher_2}: score {r.score:.6g}; degrees {g.degree(r.researcher_1)} and {g.degree(r.researcher_2)}; {len(shared)} shared collaborators. Actual shared IDs: {', '.join(map(str,shared)) or 'none'}. These nodes have no observed co-authorship edge. The score describes graph structure only.")
    heading('Visualizations')
    body('The degree distribution uses all nodes. The overview is a deterministic BFS subset of at most 35 researchers rooted at the lowest researcher ID with degree 3 in the largest component, with degree then ID ordering. The recommendation visualization contains three top pairs for the primary-AUC winner and up to ten shared collaborators per pair. Gray edges are observed and orange dashed edges are predicted. Full analysis always uses the entire graph.')
    for name in ['degree_distribution','network_overview','recommended_links']: story.append(Image(str(FIG/f'{name}.png'),width=490,height=330,kind='proportional'))
    body('network.html embeds its JavaScript and supports offline interaction. GraphML and CSV exports support Cytoscape. Cytoscape was unavailable on PATH, so no Cytoscape render was produced. See dataset/README.md for import and edge styling instructions.')
    heading('Streamlit application'); body('The local app provides Overview, Network Explorer, Algorithm Comparison, Collaborator Recommendations, Researcher Profile, and About and Limitations. Graph loading and per-researcher scores are cached. Global rankings come from precomputed CSVs. Researcher IDs come from validated selectors. Neighborhood views are bounded and clearly labeled. CSV downloads expose global and personal recommendations. Launch with python -m streamlit run app.py from the project folder.')
    heading('Limitations'); body(LIMITATIONS); body('The data is historical and may contain incomplete author coverage or identity ambiguity. Uniform random negatives can be easier than plausible nearby nonedges. Edge splitting can remove bridges and isolate nodes. The experiment has no temporal validation, topic features, or human assessment. Similarity scores are not calibrated probabilities.')
    heading('Conclusion'); body(f"The project supplies a reproducible comparison and exact structural recommendation rankings. {s['best_method']} leads on primary ROC-AUC under the stated protocol. A future extension could use timestamped papers and harder negative sets, with a separate validation/test design and topic evidence.")
    heading('References')
    for ref in ['Stanford SNAP. General Relativity and Quantum Cosmology collaboration network. https://snap.stanford.edu/data/ca-GrQc.html','J. Leskovec, J. Kleinberg and C. Faloutsos (2007). Graph Evolution: Densification and Shrinking Diameters. ACM TKDD 1(1).','D. Liben-Nowell and J. Kleinberg (2007). The link-prediction problem for social networks. JASIST 58(7), 1019-1031. https://www.cs.cornell.edu/home/kleinber/link-pred.pdf','NetworkX documentation: Jaccard coefficient, Adamic-Adar index, Preferential Attachment, and common_neighbors. https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html']:
        body(ref)
    SimpleDocTemplate(str(OUT/'evaluation_report.pdf'),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)

def slides(s,metrics,recs):
    winner=metrics[metrics.method==s['best_method']].iloc[0]; top=recs[recs.method==s['best_method']].iloc[0]
    slides=[
    ('Research Collaborator Recommendation',['Social Network Analysis AAT','A structural link prediction experiment on Stanford SNAP ca-GrQc']),
    ('Problem and objectives',['Rank currently unconnected researchers using co-authorship structure.','Compare four transparent scores and evaluate missing-link recovery.','Export exact global top 10 and top 20 rankings.']),
    ('Dataset and graph model',[f"{s['full']['nodes']:,} researcher IDs and {s['full']['edges']:,} cleaned co-authorship edges.",'Simple, undirected and unweighted graph.','arXiv GR-QC papers: January 1993 to April 2003.','Source: snap.stanford.edu/data/ca-GrQc.html']),
    ('Four prediction methods',['Common Neighbors: number of shared collaborators.','Jaccard: shared / combined collaborators.','Adamic-Adar: sum of 1 / ln(shared neighbor degree).','Preferential Attachment: degree(u) x degree(v).']),
    ('Reproducible evaluation',['Primary seed 42. Random 80/20 split of unique edges.','Keep every node in training, including isolates.','Use one full-graph nonedge per held-out positive.','Score only the training graph. Ties use ascending IDs.']),
    ('Evaluation metrics',['ROC-AUC measures positive/negative ranking quality.','Precision@K = hits / K. Recall@K = hits / test positives.','F1@K is the harmonic mean of precision and recall at K.','Balanced sampled precision differs from all-nonedge precision.']),
    ('Primary results',[f"Highest primary ROC-AUC: {s['best_method']} ({winner.ROC_AUC:.5f}).",f"Winner Precision@10: {winner['Precision@10']:.3f}; Precision@20: {winner['Precision@20']:.3f}.",'Same test results select the winner, without independent validation.']),
    ('Split sensitivity',['Repeat the protocol with seeds 7, 21 and 84.','The primary split remains seed 42.','The exported sensitivity tables report all metrics and mean/SD.']),
    ('Exact collaborator recommendations',[f"All {s['candidate_non_edges']:,} unordered nonedges are scored.",'Four heaps retain only 20 records each, with no candidate filter.',f"Example ({s['best_method']}): IDs {int(top.researcher_1)} and {int(top.researcher_2)}, {int(top.shared_collaborators)} shared collaborators.",'Final recommendations use the full observed graph.']),
    ('Network visualizations',['Bounded subset. Gray = observed; orange dashed = predicted.','Full-network charts, offline HTML and Cytoscape exports accompany this view.']),
    ('Application demonstration',['Open Overview, then select an ID in Network Explorer.','Compare metrics and ROC curves in Algorithm Comparison.','Switch method and top 10/20, explain a pair, download CSV.','Open Researcher Profile for local structure and personal suggestions.']),
    ('Limitations and conclusion',['IDs supply no names, interests or compatibility evidence.','Sampled nonedges are assumed negatives.','Random edge recovery does not prove future prediction.','Structural scores provide explainable possibilities for further study.'])]
    (OUT/'slides_content.json').write_text(json.dumps(slides,indent=2))
    slide_style=ParagraphStyle('SlideTitle',fontName='Helvetica-Bold',fontSize=30,leading=36,textColor=colors.HexColor('#123451'),spaceAfter=25)
    body_style=ParagraphStyle('SlideBody',fontName='Helvetica',fontSize=20,leading=29,spaceAfter=18)
    cell_style=ParagraphStyle('SlideCell',fontName='Helvetica',fontSize=17,leading=22)
    def slide_table(rows,widths):
        t=Table([[Paragraph(escape(str(x)),cell_style) for x in row] for row in rows],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef8')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        return t
    story=[]
    for i,(title,lines) in enumerate(slides):
        if i: story.append(PageBreak())
        story.append(Paragraph(escape(title),slide_style))
        for line in lines: story.append(Paragraph(escape(line),body_style))
        if i==6:
            story.append(slide_table([['Method','AUC','P@10','P@20']]+[[r['method'],f'{r.ROC_AUC:.4f}',f"{r['Precision@10']:.2f}",f"{r['Precision@20']:.2f}"] for _,r in metrics.iterrows()],[300,160,160,160]))
        if i==7:
            sensitivity=pd.read_csv(OUT/'sensitivity.csv').groupby('method').ROC_AUC.agg(['mean','std'])
            story.append(slide_table([['Method','Mean AUC','SD (4 seeds)']]+[[m,f'{r["mean"]:.5f}',f'{r["std"]:.5f}'] for m,r in sensitivity.iterrows()],[360,210,210]))
        if i==9:
            story.append(Image(str(FIG/'recommended_links.png'),width=520,height=280,kind='proportional'))
    SimpleDocTemplate(str(OUT/'presentation.pdf'),pagesize=(960,540),leftMargin=55,rightMargin=55,topMargin=45,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    return slides

def notes(s,slides):
    qas=[('What is a node?','An anonymized researcher represented by the supplied integer ID.'),('What is an edge?','An observed co-authorship relation between two researchers.'),('Why undirected?','Co-authorship is mutual.'),('Why remove self-loops?','Collaborating with oneself is not a distinct collaborator relation.'),('Why remove duplicates?','The model records existence of collaboration, not its frequency.'),('What does degree mean?','The number of distinct observed collaborators.'),('What is clustering?','The fraction of possible links among a node\'s neighbors that exist, averaged over all nodes, with zero for degree below two.'),('How does Common Neighbors work?','It counts researchers connected to both endpoints.'),('Why use Jaccard?','It normalizes common neighbors by the union of neighbor sets.'),('Why Adamic-Adar?','It downweights very popular shared collaborators using inverse log degree.'),('What does Preferential Attachment favor?','Pairs whose endpoint degrees have a large product.'),('How is leakage prevented?','All evaluation scores use training edges only; held-out edges supply labels.'),('Why sample negatives from the full graph complement?','To avoid labeling held-out true edges as negatives.'),('Are negatives certain?','No. Missing observed edges are assumed negatives for evaluation.'),('What does AUC mean?','The probability a positive ranks above a negative, with half credit for ties.'),('What is Recall@20?','Held-out positives found among the first 20 ranked evaluation pairs divided by all held-out positives.'),('Why is recall small?','Only K slots can retrieve thousands of test positives.'),('What is F1@K?','The harmonic mean of precision and recall at that K.'),('How are ties resolved?','Descending score, then ascending canonical endpoint IDs.'),('Does the best method have independent validation?','No. It is chosen descriptively using the same primary test results.'),('Do these results prove future prediction?','No. A random edge split evaluates missing-link recovery, not chronological forecasting.'),('Are global rankings approximate?','No. Every unordered full-graph nonedge is scored, and small heaps retain the exact top 20.'),('Why keep all training nodes?','Removing isolated nodes would change the candidate population and exclude difficult examples.'),('Why use separate full-network rankings?','After evaluation, all observed structure is available for current recommendations.'),('How do the app and notebook agree?','They import the same score and ranking functions and read the same exported results.')]
    text='# Presentation script\n\n'
    for i,(title,lines) in enumerate(slides,1): text+=f'## Slide {i}: {title}\n'+ ' '.join(lines)+'\n\n'
    text+='## Live app sequence\n1. Run `python -m streamlit run app.py`.\n2. Read the graph counts on Overview.\n3. In Network Explorer select a researcher and show predicted links. Explain the subset label.\n4. Open Algorithm Comparison and distinguish training-only scoring from full-network recommendations.\n5. In Collaborator Recommendations switch all four algorithms and top 10/20, explain a selected rank and download CSV.\n6. Open Researcher Profile and inspect shared collaborator IDs.\n7. Finish with About and Limitations.\n\n# Viva questions\n\n'
    for i,(q,a) in enumerate(qas,1): text+=f'{i}. **{q}** {a}\n\n'
    (OUT/'presentation_script_and_viva.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    s=json.loads((OUT/'summary.json').read_text()); metrics=pd.read_csv(OUT/'evaluation.csv'); recs=pd.read_csv(OUT/'recommendations_top20.csv'); g=load_graph()
    report(s,metrics,recs,g); deck=slides(s,metrics,recs); notes(s,deck); notebook()
