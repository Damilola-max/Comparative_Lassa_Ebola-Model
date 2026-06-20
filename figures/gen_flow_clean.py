"""Clean vertical pipeline flowchart for Figure 1."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(7, 12))
ax.set_xlim(0, 7)
ax.set_ylim(0, 12)
ax.axis('off')

c = {
    'i':'#BBDEFB','p':'#FFE0B2','f':'#C8E6C9','t':'#E1BEE7',
    'v':'#B2EBF2','r':'#FFCDD2','o':'#C5CAE9',
    'bi':'#1565C0','bp':'#EF6C00','bf':'#2E7D32',
    'bt':'#6A1B9A','bv':'#00838F','br':'#C62828','bo':'#283593',
}

def box(cx,cy,w,h,txt,key,fs=9,bold=False):
    b=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.03,rounding_size=0.12",
                     facecolor=c[key],edgecolor=c['b'+key],linewidth=1.8)
    ax.add_patch(b)
    ax.text(cx,cy,txt,ha='center',va='center',fontsize=fs,weight=('bold'if bold else'normal'),
            color='#212121',linespacing=1.2)

def arrow(x1,y1,x2,y2,col='#555'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=col,lw=1.4))

ax.text(3.5,11.6,'ESM-embedR Pipeline',ha='center',va='center',fontsize=15,weight='bold',color='#1565C0')
ax.text(3.5,11.25,'From raw sequence to narrative interpretation',ha='center',va='center',fontsize=9,style='italic',color='#546E7A')

# Stage 1
ax.text(3.5,10.85,'1. INPUT',ha='center',fontsize=11,weight='bold',color=c['bi'])
box(3.5,10.35,3.8,0.6,'Raw Sequence\n(FASTA or plain text)','i',10,True)

# Stage 2
ax.text(3.5,9.75,'2. PREPROCESSING',ha='center',fontsize=11,weight='bold',color=c['bp'])
box(3.5,9.2,4.2,0.6,'clean_sequence()\nuppercase + regex filter','p',9,True)
arrow(3.5,10.05,3.5,9.52)
box(1.3,8.45,1.7,0.6,'Whitelist\n20 AA codes','p',7)
box(3.5,8.45,1.7,0.6,'Regex filter\nnon-AA removal','p',7)
box(5.7,8.45,1.7,0.6,'Min length\n≥10 residues','p',7)
arrow(3.5,8.9,1.3,8.76)
arrow(3.5,8.9,3.5,8.76)
arrow(3.5,8.9,5.7,8.76)
box(3.5,7.7,3.8,0.55,'Cleaned Canonical Sequence','p',10,True)
arrow(1.3,8.15,3.5,7.98)
arrow(3.5,8.15,3.5,7.98)
arrow(5.7,8.15,3.5,7.98)

# Stage 3
ax.text(3.5,7.15,'3. FEATURE EXTRACTION',ha='center',fontsize=11,weight='bold',color=c['bf'])
arrow(3.5,7.42,3.5,7.3)
box(3.5,6.7,4.2,0.6,'amino_acid_frequency_features()\n21-D vector per sequence','f',9,True)
box(2.1,6.0,2.0,0.55,'seq_length\n(1 feature)','f',7)
box(4.9,6.0,2.0,0.55,'aa_freq_A … Y\n(20 features)','f',7)
arrow(3.5,6.4,2.1,6.28)
arrow(3.5,6.4,4.9,6.28)
box(3.5,5.35,4.0,0.5,'Feature Vector: [seq_length, aa_freq_*]','f',9,True)
arrow(2.1,5.72,3.5,5.61)
arrow(4.9,5.72,3.5,5.61)

# Stage 4
ax.text(3.5,4.85,'4. TRAINING',ha='center',fontsize=11,weight='bold',color=c['bt'])
arrow(3.5,5.1,3.5,4.98)
box(2.2,4.35,2.6,0.6,'LogisticRegression\n+ StandardScaler','t',8,True)
box(4.8,4.35,2.6,0.6,'RandomForest\nclass_weight=balanced','t',8,True)
arrow(3.5,4.68,2.2,4.65)
arrow(3.5,4.68,4.8,4.65)
box(3.5,3.65,3.3,0.5,'F1-score selection → best model','t',9,True)
arrow(2.2,4.05,3.5,3.9)
arrow(4.8,4.05,3.5,3.9)
box(3.5,3.05,3.8,0.5,'Serialize: joblib.dump()','t',8)
arrow(3.5,3.4,3.5,3.31)
box(3.5,2.45,3.6,0.5,'_build_risk_calibration()','t',9,True)
arrow(3.5,2.8,3.5,2.71)

# Inference side branch
ax.text(5.95,3.8,'5. INFERENCE',fontsize=10,weight='bold',color=c['bv'],ha='center',rotation=90)
box(5.95,4.35,1.8,0.55,'predict_sequences()\nload model','v',7)
box(5.95,3.65,1.8,0.5,'StandardScaler\n(training params)','v',7)
box(5.95,3.0,1.8,0.5,'predict_proba\n→ EBOV prob','v',7)
box(5.95,2.35,1.8,0.5,'Class assign\nthreshold ≥0.5','v',7)
arrow(5.95,4.07,5.95,3.9)
arrow(5.95,3.4,5.95,3.25)
arrow(5.95,2.75,5.95,2.6)
box(5.95,1.75,1.8,0.5,'_compute_risk_scores()','r',8,True)
arrow(5.95,2.1,5.95,2.01)

# Calibration data arrow
ax.annotate('',xy=(4.95,1.95),xytext=(4.0,2.2),
            arrowprops=dict(arrowstyle='->',color='#7B1FA2',lw=1.4,ls='--'))
ax.text(4.45,2.08,'calibration\ndata',ha='center',fontsize=6.5,color='#7B1FA2',style='italic')

# Output
ax.text(3.5,1.35,'6. OUTPUT',ha='center',fontsize=11,weight='bold',color=c['bo'])
arrow(3.5,2.2,3.5,1.55)
arrow(4.95,1.75,3.5,1.5)
for i,(label,desc)in enumerate([('predicted_virus','Lassa / Ebola'),('confidence','predict_proba'),
    ('risk_score','0 – 100'),('risk_category','5-tier'),('narrative','NL report')]):
    y=0.95-i*0.32
    box(3.5,y,4.5,0.28,f'{label}: {desc}','o',7)
    if i==0:arrow(3.5,1.35,3.5,1.1)

ax.text(3.5,-0.35,'Deployment: app.py (Streamlit ≥1.30.0)',ha='center',fontsize=9,weight='bold',color='#1565C0')
ax.text(3.5,-0.65,'https://mutation-analysis.streamlit.app',ha='center',fontsize=8,color='#546E7A')

plt.tight_layout()
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/assets/refined3_1/media/image1.png',
            dpi=300,bbox_inches='tight',facecolor='white',edgecolor='none')
plt.savefig('/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model/manuscript/assets/refined3_1/media/image1.svg',
            bbox_inches='tight',facecolor='white',edgecolor='none')
print("Done.")
