"""Reproducible collaboration analysis. Run: python analysis.py"""
from pathlib import Path
import gzip, hashlib, heapq, itertools, json, math, random, urllib.request
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from pyvis.network import Network

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
DATA = OUT / 'dataset'
FIG = OUT / 'figures'
METHODS = ['Common Neighbors', 'Jaccard', 'Adamic-Adar', 'Preferential Attachment']
FORMULAS = [r'|N(u) \cap N(v)|', r'\frac{|N(u)\cap N(v)|}{|N(u)\cup N(v)|}', r'\sum_{w\in N(u)\cap N(v)}\frac{1}{\ln d(w)}', r'd(u)d(v)']
EXPLANATIONS = ['Counts shared collaborators.', 'Divides shared collaborators by the size of the combined collaborator set.', 'Gives more weight to shared collaborators who have fewer connections.', 'Multiplies the two researchers\' degrees, favoring well-connected researchers.']
LIMITATIONS = ('IDs have no supplied names, affiliations or interests. Recommendations are structural possibilities, not confirmed collaborations or assessments of research compatibility. Sampled nonedges are assumed negatives. A random edge split tests recovery of missing observed links, not future collaboration prediction. Balanced sampled precision differs from precision across all missing links. The best method is selected on these same test results, without independent validation.')

def load_graph():
    g=nx.read_edgelist(DATA/'cleaned_edges.csv', delimiter=',', comments='source', nodetype=int)
    g.add_nodes_from(pd.read_csv(DATA/'nodes.csv')['id'].astype(int))
    return g

def scores(g, u, v):
    a, b = set(g[u]), set(g[v]); common = a & b
    return [len(common), len(common)/len(a|b) if a|b else 0., sum(1/math.log(g.degree(w)) for w in sorted(common)), len(a)*len(b)]

def ranked_for(g, u, method, k=20):
    index = METHODS.index(method)
    rows = [(u,v,scores(g,u,v)) for v in sorted(g) if v != u and not g.has_edge(u,v)]
    rows.sort(key=lambda r:(-r[2][index],min(r[0],r[1]),max(r[0],r[1])))
    return pd.DataFrame([dict(researcher_1=a,researcher_2=b,method=method,rank=i+1,score=s[index],shared_collaborators=s[0]) for i,(a,b,s) in enumerate(rows[:k])],columns=['researcher_1','researcher_2','method','rank','score','shared_collaborators'])

def stats(g):
    d = np.array([x for _,x in g.degree()])
    return dict(nodes=len(g),edges=g.number_of_edges(),components=nx.number_connected_components(g),largest_component=len(max(nx.connected_components(g),key=len)),isolates=nx.number_of_isolates(g),density=nx.density(g),degree_min=int(d.min()),degree_max=int(d.max()),degree_mean=float(d.mean()),degree_median=float(np.median(d)),average_clustering=nx.average_clustering(g))

def evaluate(g, seed, save=False):
    edges=sorted(tuple(sorted(e)) for e in g.edges()); rng=random.Random(seed); rng.shuffle(edges)
    cut=int(.8*len(edges)); train,test=edges[:cut],sorted(edges[cut:])
    t=nx.Graph(); t.add_nodes_from(sorted(g)); t.add_edges_from(train)
    nodes=sorted(g); negatives=set()
    while len(negatives)<len(test):
        u,v=sorted(rng.sample(nodes,2))
        if not g.has_edge(u,v): negatives.add((u,v))
    negatives=sorted(negatives); pairs=test+negatives; y=np.array([1]*len(test)+[0]*len(negatives))
    assert set(map(frozenset,train)).isdisjoint(map(frozenset,test))
    assert all(not g.has_edge(*e) for e in negatives)
    assert set(t)==set(g)
    values=np.array([scores(t,*e) for e in pairs])
    for j,fn in [(1,nx.jaccard_coefficient),(2,nx.adamic_adar_index),(3,nx.preferential_attachment)]:
        expected=[s for _,_,s in fn(t,pairs)]
        assert np.allclose(values[:,j],expected)
    assert all(values[i,0]==len(list(nx.common_neighbors(t,*p))) for i,p in enumerate(pairs))
    result=[]; rocs={}
    for j,m in enumerate(METHODS):
        order=sorted(range(len(pairs)),key=lambda i:(-values[i,j],*pairs[i]))
        row=dict(seed=seed,method=m,ROC_AUC=roc_auc_score(y,values[:,j]))
        for k in (10,20):
            hits=int(y[order[:k]].sum()); p=hits/k; r=hits/len(test)
            row.update({f'Precision@{k}':p,f'Recall@{k}':r,f'F1@{k}':2*p*r/(p+r) if p+r else 0})
        result.append(row)
        fpr,tpr,_=roc_curve(y,values[:,j]); rocs[m]=dict(fpr=fpr.tolist(),tpr=tpr.tolist())
    if save:
        for name,rows in [('train_edges',sorted(train)),('test_edges',test),('negative_pairs',negatives)]:
            pd.DataFrame(rows,columns=['source','target']).to_csv(DATA/f'{name}.csv',index=False)
        table=pd.DataFrame(pairs,columns=['source','target']); table['label']=y
        for j,m in enumerate(METHODS): table[m]=values[:,j]
        table.to_csv(OUT/'evaluation_pairs.csv',index=False)
        (OUT/'roc.json').write_text(json.dumps(rocs))
    return result,stats(t)

def global_rankings(g):
    # Stream all unordered nonedges: exact ranking with only four heaps of 20.
    adj={u:set(g[u]) for u in g}; deg=dict(g.degree()); inv={u:1/math.log(d) if d>1 else 0 for u,d in deg.items()}
    heaps=[[] for _ in METHODS]; count=0
    for u,v in itertools.combinations(sorted(g),2):
        if v in adj[u]: continue
        count+=1; common=adj[u]&adj[v]; cn=len(common)
        vals=[cn,cn/(deg[u]+deg[v]-cn) if deg[u]+deg[v]-cn else 0,sum(inv[w] for w in sorted(common)),deg[u]*deg[v]]
        for j,s in enumerate(vals):
            item=(s,-u,-v,cn)
            if len(heaps[j])<20: heapq.heappush(heaps[j],item)
            elif item>heaps[j][0]: heapq.heapreplace(heaps[j],item)
    rows=[]
    for m,h in zip(METHODS,heaps):
        for rank,(s,nu,nv,cn) in enumerate(sorted(h,reverse=True),1):
            rows.append(dict(researcher_1=-nu,researcher_2=-nv,method=m,rank=rank,score=s,shared_collaborators=cn))
    frame=pd.DataFrame(rows)
    assert all(u!=v and not g.has_edge(u,v) for u,v in frame[['researcher_1','researcher_2']].itertuples(index=False,name=None))
    frame.to_csv(OUT/'recommendations_top20.csv',index=False)
    frame[frame['rank']<=10].to_csv(OUT/'recommendations_top10.csv',index=False)
    return frame,count

def bounded(g, center=None, limit=70):
    if center is None: center=sorted(g,key=lambda u:(-g.degree(u),u))[0]
    chosen=[center]; seen={center}
    for u in chosen:
        for v in sorted(g[u],key=lambda v:(-g.degree(v),v)):
            if v not in seen and len(chosen)<limit: chosen.append(v); seen.add(v)
        if len(chosen)>=limit: break
    return g.subgraph(chosen).copy()

def interactive(g, predicted=()):
    net=Network(height='520px',width='100%',cdn_resources='in_line',bgcolor='#f7fafc')
    for u in sorted(g): net.add_node(u,label=str(u),title=f'Researcher {u}; displayed degree {g.degree(u)}',color='#2563eb',size=9)
    for u,v in g.edges(): net.add_edge(u,v,color='#aab5c5',title='Observed co-authorship')
    for u,v in predicted:
        if u in g and v in g: net.add_edge(u,v,color='#e76f20',dashes=True,width=3,title='Predicted structural possibility')
    net.set_options('{"physics":{"stabilization":{"iterations":100}},"interaction":{"hover":true,"navigationButtons":true}}')
    # PyVis embeds vis-network with in_line, but its template also includes
    # optional external Bootstrap assets. No Bootstrap controls are used here.
    import re
    html=net.generate_html()
    html=re.sub(r'<link[^>]+href="https?://[^>]+>', '', html)
    html=re.sub(r'<script[^>]+src="https?://[^>]+>\s*</script>', '', html)
    return html

def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/f'{name}.png',dpi=170); plt.savefig(FIG/f'{name}.svg'); plt.close()

def run():
    for p in (OUT,DATA,FIG): p.mkdir(parents=True,exist_ok=True)
    raw=DATA/'ca-GrQc.txt.gz'
    if not raw.exists(): urllib.request.urlretrieve('https://snap.stanford.edu/data/ca-GrQc.txt.gz',raw)
    g=nx.Graph(); raw_rows=loops=0
    with gzip.open(raw,'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            u,v=map(int,line.split()); raw_rows+=1; g.add_nodes_from([u,v])
            if u==v: loops+=1
            else: g.add_edge(u,v)
    pd.DataFrame(sorted(tuple(sorted(e)) for e in g.edges()),columns=['source','target']).to_csv(DATA/'cleaned_edges.csv',index=False)
    pd.DataFrame([{'id':u,'label':str(u),'degree':g.degree(u)} for u in sorted(g)]).to_csv(DATA/'nodes.csv',index=False)
    nx.write_graphml(g,DATA/'observed_network.graphml')
    rows=[]; split={}
    for seed in [42,7,21,84]:
        res,st=evaluate(g,seed,seed==42); rows.extend(res); split[str(seed)]=st
    all_metrics=pd.DataFrame(rows); all_metrics.to_csv(OUT/'sensitivity.csv',index=False)
    primary=all_metrics[all_metrics.seed==42]; primary.to_csv(OUT/'evaluation.csv',index=False)
    all_metrics.groupby('method').agg({c:['mean','std'] for c in primary.columns if c not in ['seed','method']}).to_csv(OUT/'sensitivity_summary.csv')
    rec,count=global_rankings(g)
    best=primary.sort_values(['ROC_AUC','method'],ascending=[False,True]).iloc[0]['method']
    summary=dict(full=stats(g),training=split,primary_seed=42,seeds=[42,7,21,84],raw_rows=raw_rows,self_loop_rows=loops,duplicate_or_reverse_rows=raw_rows-loops-g.number_of_edges(),candidate_non_edges=count,best_method=best,selection_criterion='Highest primary ROC-AUC; alphabetical method name breaks ties',sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    giant=max(nx.connected_components(g),key=len)
    overview_center=min(u for u in giant if g.degree(u)==3)
    summary['overview']=dict(center=overview_center,node_limit=35,rule='BFS from lowest ID with degree 3 in the largest component; neighbor order descending degree then ID; all induced edges')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.figure(figsize=(8,4)); degrees=[d for _,d in g.degree()]; plt.hist(degrees,bins=50); plt.yscale('log'); plt.xlabel('Degree'); plt.ylabel('Researchers (log scale)'); plt.title('Full cleaned network degree distribution'); savefig('degree_distribution')
    primary.set_index('method')[['ROC_AUC','Precision@10','Precision@20']].plot.bar(figsize=(9,5),ylim=(0,1.08)); plt.xticks(rotation=12,ha='right'); plt.ylabel('Metric'); savefig('model_comparison')
    rocs=json.loads((OUT/'roc.json').read_text()); plt.figure(figsize=(7,5))
    for m,r in rocs.items(): plt.plot(r['fpr'],r['tpr'],label=m)
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.legend(); plt.title('Primary split ROC curves'); savefig('roc_curves')
    h=bounded(g,overview_center,35); pos=nx.spring_layout(h,seed=42,iterations=150)
    plt.figure(figsize=(10,7)); nx.draw_networkx(h,pos,node_size=120,font_size=9,width=.6,edge_color='#b0b9c6'); plt.title(f'Overview: BFS from ID {overview_center}, {len(h)} of {len(g)} researchers'); savefig('network_overview')
    selected=rec[rec.method==best].head(3); pred=list(selected[['researcher_1','researcher_2']].itertuples(index=False,name=None))
    nodes=set()
    for u,v in pred: nodes.update([u,v]); nodes.update(sorted(nx.common_neighbors(g,u,v))[:10])
    h=g.subgraph(nodes).copy(); pos=nx.spring_layout(h,seed=42)
    plt.figure(figsize=(10,7)); nx.draw_networkx(h,pos,node_size=190,font_size=11,edge_color='#adb5bd',width=.6); nx.draw_networkx_edges(h,pos,edgelist=pred,edge_color='#e76f20',style='dashed',width=2); plt.title('Selected recommendations: orange dashed predicted; gray observed'); savefig('recommended_links')
    (OUT/'network.html').write_text(interactive(h,pred),encoding='utf-8')
    pd.DataFrame([{'source':u,'target':v,'edge_type':'observed'} for u,v in h.edges()]+[{'source':u,'target':v,'edge_type':'predicted'} for u,v in pred]).to_csv(DATA/'visualization_edges.csv',index=False)
    nx.write_graphml(h,DATA/'visualization_observed.graphml')
    (DATA/'README.md').write_text(f'''# Dataset package
Source: https://snap.stanford.edu/data/ca-GrQc.html
Download: https://snap.stanford.edu/data/ca-GrQc.txt.gz
Citation: Leskovec, Kleinberg and Faloutsos. Graph Evolution: Densification and Shrinking Diameters. ACM TKDD 1(1), 2007.
Original gzip preserved byte-for-byte. SHA256: {summary['sha256']}
Integer IDs only. Raw rows: {raw_rows}; removed self-loop rows: {loops}; duplicate/reverse rows: {summary['duplicate_or_reverse_rows']}.
cleaned_edges.csv: source,target, one row per unordered observed edge (source < target).
nodes.csv: id,label,degree. GraphML preserves all observed nodes. Load nodes.csv as well as the edges to retain the one node isolated after loop removal.
train_edges.csv and test_edges.csv: primary seed 42, floor(0.8*M) train edges. Training retains all nodes.
negative_pairs.csv: distinct unordered pairs absent from the full graph, equal in count to test edges.
Seeds: 42 primary; 7,21,84 sensitivity. Python random.Random samples sorted IDs and shuffled sorted edges.
visualization_edges.csv has edge_type observed/predicted and only the explained visualization subset.
For Cytoscape: import observed_network.graphml, or nodes.csv then cleaned_edges.csv as source/target edges. For the bounded view import nodes.csv and visualization_edges.csv, map edge_type to gray solid / orange dashed. No Cytoscape render was produced because Cytoscape was unavailable on PATH.
''')
    print(json.dumps(summary,indent=2)); return summary

if __name__=='__main__': run()
