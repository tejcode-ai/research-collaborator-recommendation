"""Create portable submission and application archives, excluding environments."""
from pathlib import Path
import zipfile
from analysis import ROOT, OUT

def package():
    app=OUT/'streamlit_application.zip'
    with zipfile.ZipFile(app,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['app.py','analysis.py','requirements.txt','README.md','start_app.ps1']:
            z.write(ROOT/name,name)
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix in ['.csv','.json','.png','.svg','.html','.graphml','.gz']:
                z.write(p,p.relative_to(ROOT))
    submission=ROOT/'Research_Collaborator_Submission.zip'
    with zipfile.ZipFile(submission,'w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            rel=p.relative_to(ROOT)
            if p.is_file() and p!=submission and not any(x in rel.parts for x in ['.git','.venv','qa','__pycache__','.ipynb_checkpoints','.jupyter_runtime','.ipython','.mplconfig']):
                z.write(p,rel)
    print(submission)

if __name__=='__main__': package()
