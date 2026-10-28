"""Check saved data, methods, notebook outputs, PDFs and Streamlit interactions."""
import json, itertools
import networkx as nx
import pandas as pd
import numpy as np
import nbformat
import fitz
from streamlit.testing.v1 import AppTest
from analysis import *

def verify():
    checks={}; g=load_graph(); s=json.loads((OUT/'summary.json').read_text())
    train=pd.read_csv(DATA/'train_edges.csv'); test=pd.read_csv(DATA/'test_edges.csv'); neg=pd.read_csv(DATA/'negative_pairs.csv')
    checks['train_test_disjoint']=set(map(tuple,train.values)).isdisjoint(map(tuple,test.values))
    checks['split_reconstructs_full']=set(map(tuple,train.values))|set(map(tuple,test.values))==set(tuple(sorted(e)) for e in g.edges())
    checks['negative_validity']=len(neg)==len(test) and len(set(map(tuple,neg.values)))==len(neg) and all(u<v and not g.has_edge(u,v) for u,v in neg.values)
    metrics=pd.read_csv(OUT/'evaluation.csv'); rerun,_=evaluate(g,42)
    checks['reproducible_primary']=np.allclose(metrics.select_dtypes('number'),pd.DataFrame(rerun).select_dtypes('number'))
    rec=pd.read_csv(OUT/'recommendations_top20.csv'); checks['recommendations_valid']=all(u<v and not g.has_edge(u,v) for u,v in rec[['researcher_1','researcher_2']].values)
    checks['recommendation_count']=all(len(rec[rec.method==m])==20 for m in METHODS)
    checks['recommendation_scores_match_shared_code']=all(np.isclose(r.score,scores(g,r.researcher_1,r.researcher_2)[METHODS.index(r.method)]) and r.shared_collaborators==len(list(nx.common_neighbors(g,r.researcher_1,r.researcher_2))) for r in rec.itertuples())
    checks['exact_candidate_count']=s['candidate_non_edges']==len(g)*(len(g)-1)//2-g.number_of_edges()
    checks['graphml_roundtrip']=len(nx.read_graphml(DATA/'observed_network.graphml'))==len(g)
    nb=nbformat.read(OUT/'Research_Collaborator_Recommendation.ipynb',as_version=4)
    checks['notebook_executed']=all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
    checks['html_embeds_javascript']='vis-network' in (OUT/'network.html').read_text(encoding='utf-8')
    import re, xml.etree.ElementTree as ET
    from PIL import Image as PILImage
    html=(OUT/'network.html').read_text(encoding='utf-8')
    checks['html_no_external_script_assets']=not bool(re.search(r'<script[^>]+src=["\']https?://',html))
    for f in FIG.glob('*.png'):
        with PILImage.open(f) as image: image.verify()
    for f in FIG.glob('*.svg'): ET.parse(f)
    checks['png_svg_open']=True
    pdfs={}
    qa=ROOT/'qa'; qa.mkdir(exist_ok=True)
    for name in ['evaluation_report.pdf','presentation.pdf']:
        doc=fitz.open(OUT/name); pdfs[name]=len(doc)
        checks[name+'_readable']=all(len(page.get_text().strip())>30 for page in doc)
        for i,page in enumerate(doc): page.get_pixmap(matrix=fitz.Matrix(1,1)).save(qa/f'{Path(name).stem}_{i+1:02d}.png')
    checks['presentation_12_pages']=pdfs['presentation.pdf']==12
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run(); checks['app_initial']=not bool(app.exception)
    pages=['Overview','Network Explorer','Algorithm Comparison','Collaborator Recommendations','Researcher Profile','About and Limitations']
    details=[]
    for page in pages:
        app.sidebar.radio[0].set_value(page).run(); assert not app.exception, str(app.exception)
        details.append(page)
        if page=='Collaborator Recommendations':
            for m in METHODS:
                app.selectbox[0].set_value(m).run(); assert not app.exception
                for k in [10,20]:
                    app.selectbox[1].set_value(k).run(); assert not app.exception
                    assert len(app.dataframe[0].value)==k
            assert len(app.get('download_button'))==2
            checks['app_download_controls']=True
        if page in ['Network Explorer','Researcher Profile']:
            app.selectbox[0].set_value(sorted(g)[-1]).run(); assert not app.exception
    checks['all_app_sections_and_selectors']=True
    # Check CSV payloads directly: AppTest does not simulate native browser saves.
    for m in METHODS:
        frame=ranked_for(g,sorted(g)[0],m); assert len(frame)==20
        assert len(pd.read_csv(__import__('io').StringIO(frame.to_csv(index=False))))==20
    checks['csv_download_payload_roundtrip']=True
    result=dict(checks=checks,all_passed=all(checks.values()),pdf_pages=pdfs,app_sections_checked=details,browser_limitation='AppTest checks Python widget interactions and CSV payloads; browser zoom/pan and native download dialogs require manual review.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    assert result['all_passed']

if __name__=='__main__': verify()
