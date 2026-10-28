"""Local demonstration: streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
import networkx as nx
from analysis import OUT, FIG, METHODS, FORMULAS, EXPLANATIONS, LIMITATIONS, load_graph, ranked_for, bounded, interactive

st.set_page_config(page_title='Research Collaborator Recommendation',page_icon='🔗',layout='wide')
st.title('Research Collaborator Recommendation')
st.caption('Social Network Analysis AAT • Stanford SNAP ca-GrQc • Dataset IDs only')
if not (OUT/'summary.json').exists():
    st.error('Run python analysis.py first to generate the project results.'); st.stop()

@st.cache_resource
def graph(): return load_graph()
@st.cache_data
def results():
    return json.loads((OUT/'summary.json').read_text()),pd.read_csv(OUT/'evaluation.csv'),pd.read_csv(OUT/'recommendations_top20.csv')
@st.cache_data
def personal(u,m,k): return ranked_for(graph(),u,m,k)

g=graph(); summary,metrics,recs=results()
page=st.sidebar.radio('Section',['Overview','Network Explorer','Algorithm Comparison','Collaborator Recommendations','Researcher Profile','About and Limitations'])

def researcher(label='Researcher ID'):
    return st.selectbox(label,sorted(g),index=0)
def view(u, predicted=()):
    h=bounded(g,u,60)
    # Include suggested endpoints without exceeding 80 displayed nodes.
    nodes=set(h)
    for a,b in predicted: nodes.update([a,b])
    h=g.subgraph(nodes).copy()
    st.caption(f'Bounded subset: {len(h)} of {len(g)} researchers. Gray = observed co-authorship. Orange dashed = predicted possibility. Hover for IDs; drag, zoom and pan.')
    st.iframe(interactive(h,predicted),height=540)
def download(df,label,key): st.download_button(label,df.to_csv(index=False).encode(),file_name=key+'.csv',mime='text/csv',key=key)
def recommendations(u,m,k):
    frame=personal(u,m,k)
    if frame.empty: st.info('No unconnected researchers are available.'); return
    st.dataframe(frame,hide_index=True,width='stretch')
    download(frame,'Download researcher recommendations','personal_'+page)
    row=frame.iloc[0]; v=int(row.researcher_2); shared=sorted(nx.common_neighbors(g,u,v))
    st.write(f'Top suggestion: researcher {u} and researcher {v} have {len(shared)} shared collaborators. Degrees: {g.degree(u)} and {g.degree(v)}. Score: {row.score:.6g}.')
    st.write('Shared collaborator IDs:',', '.join(map(str,shared)) or 'None')
    return [(u,v)]

if page=='Overview':
    st.write('Can the structure of a co-authorship network suggest potential collaborators? We compare four transparent ways to score missing links.')
    st.markdown('[Dataset source: Stanford SNAP](https://snap.stanford.edu/data/ca-GrQc.html). Nodes are researcher IDs. An undirected edge means two researchers co-authored a paper. The graph is simple and unweighted.')
    cols=st.columns(4)
    for c,key in zip(cols,['nodes','edges','components','average_clustering']): c.metric(key.replace('_',' ').title(),f"{summary['full'][key]:,.4f}" if isinstance(summary['full'][key],float) else f"{summary['full'][key]:,}")
    st.dataframe(pd.DataFrame([summary['full']]),hide_index=True)
    st.image(str(FIG/'degree_distribution.png'))
    st.image(str(FIG/'network_overview.png'))
elif page=='Network Explorer':
    st.subheader('Explore observed and suggested links')
    u=researcher(); m=st.selectbox('Algorithm',METHODS); frame=personal(u,m,3)
    show=st.checkbox('Show up to 3 predicted links',value=True)
    view(u,[(u,int(v)) for v in frame.researcher_2] if show else [])
elif page=='Algorithm Comparison':
    st.subheader('Held-out evaluation: primary seed 42')
    st.write('Shuffle unique unordered edges, train on floor(80%), and hold out the remainder. Keep every node, including training isolates. Score only the training graph. Sample one distinct full-graph nonedge per test positive. All methods use identical pairs. Break score ties by ascending unordered researcher IDs.')
    for m,f,e in zip(METHODS,FORMULAS,EXPLANATIONS):
        st.markdown('**'+m+'**'); st.latex(f); st.write(e)
    st.caption('N(u) = collaborators of u; d(u) = degree; ln = natural logarithm.')
    st.dataframe(metrics,hide_index=True,width='stretch')
    st.image(str(FIG/'model_comparison.png')); st.image(str(FIG/'roc_curves.png'))
    st.write('ROC-AUC: probability a randomly chosen positive outranks a negative, with half credit for ties. Precision@K = hits/K. Recall@K = hits/all held-out positives. F1@K = harmonic mean of Precision@K and Recall@K. Recall and F1 are measured at K, not over an arbitrary classification threshold.')
    st.write(f"Strongest by highest primary ROC-AUC: {summary['best_method']}. This is a descriptive selection on the same test set, without independent validation. Popularity and neighborhood similarity can favor different pairs.")
    st.subheader('Split sensitivity'); st.dataframe(pd.read_csv(OUT/'sensitivity.csv'),hide_index=True)
    st.warning(LIMITATIONS)
elif page=='Collaborator Recommendations':
    st.subheader('Full observed graph recommendations')
    st.caption('These scores use the full cleaned graph after evaluation. Global rankings examine every unordered nonedge exactly and keep only the highest 20 per method.')
    m=st.selectbox('Algorithm',METHODS); k=st.selectbox('Number of recommendations',[10,20])
    frame=recs[(recs.method==m)&(recs['rank']<=k)]
    st.markdown('**Global pair ranking**'); st.dataframe(frame,hide_index=True,width='stretch'); download(frame,'Download global ranking','global')
    rank=st.selectbox('Explain global rank',frame['rank'].tolist()); row=frame[frame['rank']==rank].iloc[0]
    u,v=int(row.researcher_1),int(row.researcher_2); common=sorted(nx.common_neighbors(g,u,v))
    st.write(f'IDs {u} and {v}: degrees {g.degree(u)}, {g.degree(v)}; {len(common)} shared collaborators; score {row.score:.6g}. No observed edge connects this pair.')
    st.write('Shared IDs:',', '.join(map(str,common)) or 'None')
    st.markdown('**Recommendations for one researcher**'); u=researcher(); recommendations(u,m,k)
elif page=='Researcher Profile':
    u=researcher(); m=st.selectbox('Algorithm',METHODS)
    st.metric('Observed degree',g.degree(u)); st.write('Existing collaborator IDs:',', '.join(map(str,sorted(g[u]))) or 'None')
    pred=recommendations(u,m,10); view(u,pred or [])
else:
    st.write(LIMITATIONS)
    st.write('Data covers arXiv General Relativity and Quantum Cosmology papers from January 1993 to April 2003. Cleaning removes self-loops and duplicate unordered edges. Seeds 7, 21 and 84 supplement the primary seed 42. No test edges enter scoring or tuning. Global recommendations use the full observed graph separately.')
    st.markdown('Dataset citation: J. Leskovec, J. Kleinberg and C. Faloutsos. *Graph Evolution: Densification and Shrinking Diameters*. ACM TKDD 1(1), 2007. [SNAP](https://snap.stanford.edu/data/ca-GrQc.html). Methods: [NetworkX link prediction documentation](https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html).')
    st.write('Cytoscape was unavailable on PATH, so this submission uses Python-rendered figures. Import instructions accompany the dataset. No Cytoscape render was produced.')
