


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from collections import deque
import heapq

st.set_page_config(
    page_title="AI/ML Virtual Laboratory | BECE309L",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)






st.markdown('<style>\n/* ===== Dr. Suraj Prakash Sahoo ===== */\nhtml, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {\n    background:#ffffff !important;\n    color:#111827 !important;\n}\n\n/* Main content */\n.main, .block-container, [data-testid="stMainBlockContainer"] {\n    background:#ffffff !important;\n    color:#111827 !important;\n}\n\n/* Every normal text element */\np, li, span, label, small, div, article, section {\n    color:#111827 !important;\n}\n\n/* Headings */\nh1, h2, h3, h4, h5, h6,\n.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,\n.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {\n    color:#0b1f3a !important;\n    background:#ffffff !important;\n    opacity:1 !important;\n    font-weight:800 !important;\n}\n\n/* Markdown containers */\n[data-testid="stMarkdownContainer"],\n[data-testid="stMarkdownContainer"] * {\n    color:#111827 !important;\n    background:transparent !important;\n}\n\n/* Sidebar */\nsection[data-testid="stSidebar"],\nsection[data-testid="stSidebar"] > div,\nsection[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {\n    background:#f8fafc !important;\n    color:#111827 !important;\n}\nsection[data-testid="stSidebar"] * {\n    color:#111827 !important;\n}\n\n/* Form/control labels */\n.stSelectbox label,\n.stMultiSelect label,\n.stNumberInput label,\n.stSlider label,\n.stRadio label,\n.stCheckbox label,\n.stTextInput label,\n.stTextArea label,\n.stFileUploader label {\n    color:#111827 !important;\n    background:#ffffff !important;\n    font-weight:700 !important;\n    opacity:1 !important;\n}\n\n/* Select boxes and inputs */\ndiv[data-baseweb="select"] > div,\ninput, textarea {\n    background:#ffffff !important;\n    color:#111827 !important;\n    border-color:#94a3b8 !important;\n}\ndiv[data-baseweb="select"] *,\ndiv[data-baseweb="input"] *,\ninput::placeholder,\ntextarea::placeholder {\n    color:#111827 !important;\n    opacity:1 !important;\n}\n\n/* Dropdown popups */\ndiv[role="listbox"],\ndiv[role="option"],\nul[role="listbox"] {\n    background:#ffffff !important;\n    color:#111827 !important;\n}\ndiv[role="option"] * {\n    color:#111827 !important;\n}\n\n/* Buttons */\n.stButton button,\nbutton[kind] {\n    background:#0f4c81 !important;\n    color:#ffffff !important;\n    border:1px solid #0b355a !important;\n    font-weight:800 !important;\n}\n.stButton button:hover {\n    background:#0b355a !important;\n    color:#ffffff !important;\n}\n\n/* Cards */\n.card {\n    background:#f8fafc !important;\n    color:#111827 !important;\n    border:2px solid #cbd5e1 !important;\n    border-radius:12px !important;\n    padding:18px !important;\n    margin:10px 0 !important;\n}\n.card * {\n    background:transparent !important;\n    color:#111827 !important;\n}\n.card h2, .card h3, .card h4 {\n    color:#0b1f3a !important;\n}\n\n/* Alerts / info / success / warning / error */\n[data-testid="stAlert"],\n[data-testid="stNotification"],\n.stAlert {\n    background:#f8fafc !important;\n    color:#111827 !important;\n    border:1px solid #94a3b8 !important;\n}\n[data-testid="stAlert"] *,\n[data-testid="stNotification"] *,\n.stAlert * {\n    color:#111827 !important;\n}\n\n/* Metrics */\n[data-testid="stMetric"],\n[data-testid="stMetricLabel"],\n[data-testid="stMetricValue"],\n[data-testid="stMetricDelta"] {\n    background:#ffffff !important;\n    color:#111827 !important;\n}\n[data-testid="stMetricLabel"] {\n    font-weight:700 !important;\n}\n[data-testid="stMetricValue"] {\n    color:#0b1f3a !important;\n    font-weight:800 !important;\n}\n\n/* Expanders */\n[data-testid="stExpander"],\n[data-testid="stExpanderDetails"] {\n    background:#ffffff !important;\n    color:#111827 !important;\n    border:1px solid #cbd5e1 !important;\n}\n[data-testid="stExpander"] summary,\n[data-testid="stExpander"] summary * {\n    background:#ffffff !important;\n    color:#111827 !important;\n    font-weight:800 !important;\n}\n\n/* Dataframes / tables */\n[data-testid="stDataFrame"],\n[data-testid="stDataFrame"] > div,\n.stDataFrame {\n    background:#ffffff !important;\n    color:#111827 !important;\n}\n[data-testid="stDataFrame"] * {\n    color:#111827 !important;\n}\n\n/* Code / formulas */\ncode {\n    background:#fff7ed !important;\n    color:#7c2d12 !important;\n    border:1px solid #fed7aa !important;\n}\npre, pre code {\n    background:#f1f5f9 !important;\n    color:#111827 !important;\n    border:1px solid #cbd5e1 !important;\n}\n\n/* Captions */\n[data-testid="stCaptionContainer"],\n.stCaption {\n    color:#374151 !important;\n    background:#ffffff !important;\n    opacity:1 !important;\n}\n\n/* Links */\na {\n    color:#075985 !important;\n    background:#ffffff !important;\n    font-weight:700 !important;\n}\n\n/* Horizontal separators */\nhr {\n    border-color:#cbd5e1 !important;\n}\n\n/* Plotly chart wrapper */\n[data-testid="stPlotlyChart"],\n[data-testid="stVegaLiteChart"] {\n    background:#ffffff !important;\n    border:1px solid #e2e8f0 !important;\n}\n\n/* Radio / checkbox controls */\n[data-testid="stRadio"] *,\n[data-testid="stCheckbox"] * {\n    color:#111827 !important;\n    opacity:1 !important;\n}\n\n/* Tabs */\nbutton[data-baseweb="tab"] {\n    background:#ffffff !important;\n    color:#111827 !important;\n    font-weight:700 !important;\n}\nbutton[data-baseweb="tab"][aria-selected="true"] {\n    color:#0f4c81 !important;\n}\n\n/* Prevent low-opacity disabled-looking instructional text */\n.stMarkdown, .stText, .stCaption, .st-emotion-cache {\n    opacity:1 !important;\n}\n</style>', unsafe_allow_html=True)
st.markdown("""
<style>
.stApp{background:#0b1020;color:#eef2ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#10172e,#1a2447)}
.hero{padding:28px;border-radius:22px;background:linear-gradient(135deg,#172554,#2563eb);margin-bottom:18px}
.card{padding:17px;border-radius:15px;background:#141d35;border:1px solid #33456e;margin:8px 0}
.explain{padding:18px;border-radius:15px;background:#111a31;border-left:5px solid #60a5fa;margin:12px 0}
.what{padding:20px;border-radius:15px;background:#172b20;border-left:5px solid #4ade80;margin:12px 0}
.warn{padding:18px;border-radius:15px;background:#332716;border-left:5px solid #f59e0b;margin:12px 0}
.small{font-size:0.92rem;opacity:.9}
</style>
""", unsafe_allow_html=True)

def explain(title, body):
    st.markdown(f'<div class="explain"><b>📘 {title}</b><br><br>{body}</div>', unsafe_allow_html=True)

def what_now(text):
    st.markdown(f'<div class="what"><b>🔎 What is happening now?</b><br><br>{text}</div>', unsafe_allow_html=True)

def formula(s):
    st.latex(s)

def legend(items):
    cols = st.columns(len(items))
    for c, (symbol, title, desc) in zip(cols, items):
        c.markdown(f'<div class="card"><h3>{symbol} {title}</h3><div class="small">{desc}</div></div>', unsafe_allow_html=True)

@st.cache_data
def make_health():
    rng=np.random.default_rng(42); n=600
    d=pd.DataFrame({
        "Age":rng.integers(20,80,n),
        "BMI":rng.normal(25,4,n),
        "Blood Pressure":rng.normal(125,18,n),
        "Glucose":rng.normal(120,35,n),
        "Cholesterol":rng.normal(205,35,n),
        "Exercise":np.clip(rng.normal(4,2,n),0,12)
    })
    score=.03*d.Age+.05*d.BMI+.025*d.Glucose+.015*d["Blood Pressure"]-.12*d.Exercise+rng.normal(0,2,n)
    d["Class"]=np.where(score>9,"Positive","Negative")
    return d

@st.cache_data
def blobs():
    return make_blobs(n_samples=260, centers=[(-2,-2),(2,2)], cluster_std=1.25, random_state=7)

health=make_health()
Xblob,yblob=blobs()


# ---------- HOME / TITLE PAGE STYLING ----------
st.markdown("""
<style>
.home-title-page {
    text-align: center;
    padding: 8px 12px 18px 12px;
}
.home-title-page h1 {
    margin: 4px 0 6px 0 !important;
    font-size: clamp(2.0rem, 4vw, 3.5rem) !important;
    line-height: 1.1 !important;
    font-weight: 900 !important;
    color: #172554 !important;
    letter-spacing: -0.02em;
}
.home-title-page .home-kicker {
    display: inline-block;
    margin: 6px 0 12px 0;
    padding: 7px 16px;
    border-radius: 999px;
    background: #e0ecff !important;
    color: #0f4c81 !important;
    font-size: 0.88rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.home-title-page .home-subtitle {
    max-width: 900px;
    margin: 0 auto;
    color: #475569 !important;
    font-size: 1.08rem;
    line-height: 1.6;
}
.home-hero {
    margin: 8px 0 24px 0;
    padding: 28px 24px;
    border-radius: 22px;
    background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 55%, #eef2ff 100%) !important;
    border: 1px solid #c7d7f2;
    box-shadow: 0 8px 24px rgba(15, 76, 129, 0.08);
}
.home-section-title {
    margin: 20px 0 8px 0;
    color: #0b1f3a !important;
    font-size: 1.35rem;
    font-weight: 900;
}
.module-card-new {
    min-height: 128px;
    padding: 18px 18px 16px 18px;
    margin: 7px 0;
    border-radius: 16px;
    background: #ffffff !important;
    border: 1px solid #d7e0ec;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}
.module-card-new .num {
    color: #2563eb !important;
    font-size: 0.78rem;
    font-weight: 900;
    letter-spacing: 0.08em;
}
.module-card-new .title {
    color: #0b1f3a !important;
    font-size: 1.08rem;
    font-weight: 900;
    margin: 5px 0 6px 0;
}
.module-card-new .desc {
    color: #475569 !important;
    font-size: 0.91rem;
    line-height: 1.45;
}
.home-tip {
    margin-top: 18px;
    padding: 15px 18px;
    border-radius: 14px;
    background: #f0fdf4 !important;
    border-left: 5px solid #22c55e;
    color: #14532d !important;
}
.home-tip * { color: #14532d !important; }
.home-footer {
    text-align: center;
    margin-top: 22px;
    padding: 12px;
    color: #64748b !important;
    font-size: 0.86rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- SEARCH ENGINE FOR THE ANIMATED GRID ----------

st.markdown('<style>\nhtml, body, [class*="css"] { color:#111827 !important; }\n.stApp { background:#ffffff !important; }\n.main .block-container { max-width:1400px; padding-top:2rem; }\nh1,h2,h3,h4,h5,h6,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4,.stMarkdown h5,.stMarkdown h6 {\n  color:#0b1f3a !important; opacity:1 !important; font-weight:800 !important;\n}\n.stMarkdown,.stMarkdown p,.stMarkdown li,.stMarkdown span,.stMarkdown label,p,li,span,label,div {\n  color:#111827;\n}\n[data-testid="stCaptionContainer"], .stCaption { color:#374151 !important; opacity:1 !important; }\nsection[data-testid="stSidebar"] { background:#f1f5f9 !important; }\nsection[data-testid="stSidebar"] * { color:#111827 !important; opacity:1 !important; }\nsection[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3 { color:#0b1f3a !important; }\n.stSelectbox label,.stMultiSelect label,.stNumberInput label,.stSlider label,.stRadio label,.stCheckbox label,.stTextInput label,.stTextArea label {\n  color:#111827 !important; font-weight:700 !important; opacity:1 !important;\n}\ndiv[data-baseweb="select"] > div { background:#ffffff !important; border:1px solid #94a3b8 !important; }\ndiv[data-baseweb="select"] * { color:#111827 !important; }\ndiv[role="listbox"],div[role="option"] { background:#ffffff !important; color:#111827 !important; }\ndiv[role="option"] * { color:#111827 !important; }\n.stButton button { background:#0f4c81 !important; color:#ffffff !important; border:1px solid #0b355a !important; font-weight:800 !important; min-height:44px; }\n.stButton button:hover { background:#0b355a !important; color:#ffffff !important; }\n.card { background:#f8fafc !important; color:#111827 !important; border:2px solid #cbd5e1 !important; border-radius:12px !important; padding:18px !important; margin:10px 0 !important; }\n.card h2,.card h3,.card h4,.card p,.card li,.card code { color:#111827 !important; }\ncode { color:#7c2d12 !important; background:#fff7ed !important; border:1px solid #fed7aa !important; padding:2px 5px !important; }\npre { background:#0f172a !important; color:#f8fafc !important; border-radius:8px !important; }\npre code { color:#f8fafc !important; background:transparent !important; border:0 !important; }\n[data-testid="stDataFrame"],.stDataFrame { color:#111827 !important; }\n[data-testid="stDataFrame"] * { color:#111827 !important; }\n[data-testid="stMetricLabel"] { color:#374151 !important; font-weight:700 !important; }\n[data-testid="stMetricValue"] { color:#0b1f3a !important; font-weight:800 !important; }\n[data-testid="stExpander"] { border:1px solid #cbd5e1 !important; }\n[data-testid="stExpander"] summary,[data-testid="stExpander"] summary * { color:#111827 !important; font-weight:800 !important; }\n.stRadio [role="radiogroup"] label,.stCheckbox label { color:#111827 !important; }\nhr { border-color:#cbd5e1 !important; }\na { color:#075985 !important; font-weight:700; }\n</style>', unsafe_allow_html=True)
def search_steps(n, obstacles, start, goal, algorithm):
    def heuristic(p):
        return abs(p[0]-goal[0])+abs(p[1]-goal[1])

    def neighbors(p):
        r,c=p
        # Fixed order makes the demonstration repeatable.
        for z in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
            if 0<=z[0]<n and 0<=z[1]<n and z not in obstacles:
                yield z

    steps=[]
    parent={start:None}
    gscore={start:0}
    visited=set()
    frontier=deque([start]) if algorithm=="BFS" else [start]
    discovered={start}
    seq=0

    def snapshot(current=None, action="", note=""):
        steps.append({
            "current": current, "frontier": list(frontier) if algorithm in ["BFS","DFS"] else list(frontier),
            "visited": set(visited), "parent": dict(parent), "g": dict(gscore),
            "action": action, "note": note
        })

    snapshot(start,"START","The algorithm begins at the start node.")

    while frontier:
        if algorithm=="BFS":
            current=frontier.popleft()
        elif algorithm=="DFS":
            current=frontier.pop()
        elif algorithm=="Greedy Best-First":
            frontier.sort(key=lambda p:(heuristic(p),p[0],p[1]))
            current=frontier.pop(0)
        else: # A*
            frontier.sort(key=lambda p:(gscore.get(p,999)+heuristic(p),heuristic(p),p[0],p[1]))
            current=frontier.pop(0)

        if current in visited:
            continue
        visited.add(current)
        if current!=start:
            snapshot(current,"EXPLORE",
                     f"Exploring {current}. It is selected according to {algorithm}'s rule.")
        if current==goal:
            snapshot(current,"GOAL","The goal has been reached. Now reconstruct the final path.")
            break

        for nb in neighbors(current):
            if nb in visited or nb in discovered:
                continue
            discovered.add(nb)
            parent[nb]=current
            gscore[nb]=gscore[current]+1
            if algorithm in ["BFS","DFS"]:
                frontier.append(nb)
            else:
                frontier.append(nb)
            if algorithm=="BFS":
                rule=f"BFS adds {nb} to the queue (first-in, first-out)."
            elif algorithm=="DFS":
                rule=f"DFS adds {nb} to the stack (last-in, first-out)."
            elif algorithm=="Greedy Best-First":
                rule=f"Greedy considers {nb} using h(n)={heuristic(nb)}."
            else:
                rule=f"A* considers {nb}: g(n)={gscore[nb]}, h(n)={heuristic(nb)}, f(n)={gscore[nb]+heuristic(nb)}."
            snapshot(current,"ADD",rule)

    path=[]
    if goal in parent and goal in visited:
        z=goal
        while z is not None:
            path.append(z)
            z=parent[z]
        path.reverse()
    snapshot(goal if goal in visited else None,"DONE",
             "Final path highlighted in yellow." if path else "No path was found.")
    return steps,path

def draw_search_grid(n, obstacles, start, goal, step, path, algorithm):
    current=step["current"]
    visited=step["visited"]
    frontier=set(step["frontier"])
    grid=np.zeros((n,n))
    # 0 empty, 1 visited, 2 obstacle, 3 frontier, 4 path, 5 start, 6 goal, 7 current
    for p in visited: grid[p]=1
    for p in frontier: grid[p]=3
    for p in obstacles: grid[p]=2
    for p in path: grid[p]=4
    grid[start]=5
    grid[goal]=6
    if current is not None and current not in [start,goal]:
        grid[current]=7

    colorscale=[
        [0.00,"#f8fafc"],[0.142,"#93c5fd"],[0.285,"#1f2937"],
        [0.428,"#60a5fa"],[0.571,"#facc15"],[0.714,"#22c55e"],[0.857,"#ef4444"],[1.0,"#a855f7"]
    ]
    fig=go.Figure(go.Heatmap(
        z=grid, x=list(range(n)), y=list(range(n)),
        zmin=0,zmax=7,colorscale=colorscale,
        showscale=False, xgap=3, ygap=3,
        hovertemplate="Row %{y}, Column %{x}<extra></extra>"
    ))
    # Labels
    labels={}
    for r in range(n):
        for c in range(n):
            labels[(r,c)]=""
    labels[start]="START"
    labels[goal]="GOAL"
    if current not in [None,start,goal]:
        labels[current]="NOW"
    for p in path:
        if p not in [start,goal]:
            labels[p]="PATH"
    for p in obstacles:
        labels[p]="WALL"
    for p in frontier:
        if p not in obstacles and p not in [start,goal] and p not in path and p!=current:
            labels[p]="OPEN"
    for (r,c),txt in labels.items():
        if txt:
            fig.add_annotation(x=c,y=r,text=txt,showarrow=False,font=dict(size=10,color="#111827"))
    fig.update_layout(
        height=560, margin=dict(l=30,r=20,t=20,b=30),
        xaxis=dict(title="Column",dtick=1),
        yaxis=dict(title="Row",dtick=1,autorange="reversed")
    )
    return fig

# ---------- SIDEBAR ----------
st.sidebar.markdown("""
<div style="background:#ffffdd;border:2px solid #94a3b8;border-radius:10px;padding:12px;margin-bottom:12px;">
<b style="color:#0b1f3a;font-size:16px;">Dr. Suraj Prakash Sahoo</b><br>
<span style="color:#111827;">School of Electronics Engineering, VIT Vellore.</span>
</div>
""", unsafe_allow_html=True)

modules = [
    "01 • Intelligent Agents","02 • Search Algorithms","03 • Logic",
    "04 • Bayesian Network & HMM","05 • Data Preparation",
    "06 • Machine Learning","07 • Deep Learning","🎯 Practice & Viva"
]
page=st.sidebar.radio("🧠 AI/ML Virtual Laboratory",["🏠 Home"]+modules)

# ---------- HOME ----------
if page == "🏠 Home":
    st.markdown('<div class="home-hero">', unsafe_allow_html=True)

    # VIT logo above the title.
    logo_left, logo_mid, logo_right = st.columns([1, 2, 1])
    with logo_mid:
        st.image("assets/logo.png", width=470)

    st.markdown("""
    <div class="home-title-page">
        <div class="home-kicker">BECE309L • Artificial Intelligence & Machine Learning</div>
        <h1>AI/ML Virtual Laboratory</h1>
        <div class="home-subtitle">
            An interactive learning space to explore AI and ML algorithms through
            visual experiments, step-by-step calculations, and live results.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="home-section-title">🧪 Laboratory Modules</div>', unsafe_allow_html=True)

    cards = [
        ("01", "Intelligent Agents", "PEAS, agent architectures & environments"),
        ("02", "Search Algorithms", "BFS, DFS, Greedy Best-First & A*"),
        ("03", "Logic", "Propositional logic, FOL & representations"),
        ("04", "Uncertainty", "Bayesian reasoning & Hidden Markov Models"),
        ("05", "Data Preparation", "Missing values, normalization & PCA"),
        ("06", "Machine Learning", "Classification, regression, clustering & SVM"),
        ("07", "Deep Learning", "Neural networks, backpropagation & CNN"),
    ]

    cols = st.columns(3)
    for idx, (no, title, desc) in enumerate(cards):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="module-card-new">
                    <div class="num">MODULE {no}</div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="home-tip">
            <b>🚀 Start exploring:</b> Select a module from the sidebar, choose the
            experiment inputs, run the steps, and observe how the algorithm reaches
            its result.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-footer">
            Developed as an interactive classroom laboratory • VIT Vellore
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- MODULE 1 ----------
elif page.startswith("01"):
    st.title("Module 1 • Intelligent Agents")
    topic=st.sidebar.radio("Module 1 topics",["PEAS","Agent Architecture","Environment Types"])
    if topic=="PEAS":
        explain("PEAS","PEAS describes an intelligent task environment using Performance measure, Environment, Actuators and Sensors.")
        task=st.selectbox("Choose an agent",["Self-driving car","Medical diagnosis system","Vacuum-cleaning robot","Smart farming system"])
        data={
        "Self-driving car":[("P","Safety, travel time, comfort"),("E","Roads, traffic, weather"),("A","Steering, brakes, accelerator"),("S","Camera, LiDAR, GPS, speed sensors")],
        "Medical diagnosis system":[("P","Diagnostic accuracy, low error"),("E","Clinical/patient information"),("A","Diagnosis/report"),("S","Symptoms, tests, history")],
        "Vacuum-cleaning robot":[("P","Clean area, low time/energy"),("E","Rooms, furniture, dirt"),("A","Wheels, vacuum motor"),("S","Bump, dirt, distance sensors")],
        "Smart farming system":[("P","Yield and water efficiency"),("E","Farm field and weather"),("A","Pump and valves"),("S","Soil moisture, temperature, humidity")]
        }[task]
        cols=st.columns(4)
        for c,(h,v) in zip(cols,data):
            c.markdown(f'<div class="card"><h2>{h}</h2>{v}</div>',unsafe_allow_html=True)
        what_now(f"For the selected {task}, the four PEAS components describe what success means, where the agent operates, how it acts, and what it senses.")
    elif topic=="Agent Architecture":
        arch=st.selectbox("Choose architecture",["Simple Reflex","Model-Based Reflex","Goal-Based","Utility-Based","Learning Agent"])
        desc={
        "Simple Reflex":"Uses the current percept and condition-action rules.",
        "Model-Based Reflex":"Maintains an internal state to handle information not directly visible now.",
        "Goal-Based":"Selects actions that help achieve a defined goal.",
        "Utility-Based":"Chooses among alternatives using a utility or preference measure.",
        "Learning Agent":"Improves its behaviour using experience and feedback."
        }[arch]
        explain(arch,desc)
        st.markdown("### Architecture flow")
        flow={
        "Simple Reflex":"Percept → Rule → Action",
        "Model-Based Reflex":"Percept + Internal State → Rule → Action → Update State",
        "Goal-Based":"Percept → State → Goal/Planning → Action",
        "Utility-Based":"Percept → State → Utility comparison → Best Action",
        "Learning Agent":"Experience → Learning → Improved decision → Action"
        }[arch]
        st.markdown(f'<div class="card"><h2>{flow}</h2></div>',unsafe_allow_html=True)
        what_now(f"You selected {arch}. The important distinction is what information the agent uses to choose its next action.")
    else:
        explain("Five environment properties","For this laboratory classify an environment using observable, deterministic, episodic, static and discrete properties.")
        env=st.selectbox("Choose an example",["Chess","Image classification","Self-driving car","Vacuum robot","Medical diagnosis"])
        table={
        "Chess":["Fully observable","Deterministic","Sequential","Static","Discrete"],
        "Image classification":["Fully observable","Deterministic","Episodic","Static","Discrete"],
        "Self-driving car":["Partially observable","Stochastic","Sequential","Dynamic","Continuous"],
        "Vacuum robot":["Partially observable","Mostly deterministic","Sequential","Dynamic","Discrete"],
        "Medical diagnosis":["Partially observable","Stochastic","Sequential","Dynamic","Mostly discrete"]
        }[env]
        st.table(pd.DataFrame({"Property":["Observable","Deterministic","Episodic","Static","Discrete"],"Meaning":["Can the agent see the relevant state?","Is the result of an action predictable?","Are episodes independent?","Does the world remain unchanged while deciding?","Are states/actions separate values?"],"Classification":table}))
        what_now(f"{env} is classified using the five requested environment dimensions.")

# ---------- MODULE 2 ----------
elif page.startswith("02"):
    st.title("Module 2 • Search Algorithms — Tree View")
    explain("Why a tree?","For beginners, a search tree makes the concept explicit: each circle is a state, each line is a possible transition, and the algorithm chooses which frontier node to expand next.")
    algorithm=st.selectbox("Choose algorithm",["BFS","DFS","Greedy Best-First","A*"])
    goal=st.selectbox("Choose goal node",["L","G","O","R"])
    edges={"S":["A","B","C"],"A":["D","E"],"B":["F","G"],"C":["H","I"],"D":["J","K"],"E":["L"],"F":["M"],"G":["N","O"],"H":["P"],"I":["Q","R"],"J":[],"K":[],"L":[],"M":[],"N":[],"O":[],"P":[],"Q":[],"R":[]}
    coords={"S":(0,0),"A":(-3,-1),"B":(0,-1),"C":(3,-1),"D":(-4.5,-2),"E":(-1.5,-2),"F":(-1,-2),"G":(1,-2),"H":(1.5,-2),"I":(4.5,-2),"J":(-5.2,-3),"K":(-3.8,-3),"L":(-1.5,-3),"M":(-1.7,-3),"N":(.4,-3),"O":(1.7,-3),"P":(1,-3),"Q":(4,-3),"R":(5.2,-3)}
    h={n:abs(coords[n][0]-coords[goal][0])+.5*abs(coords[n][1]-coords[goal][1]) for n in edges}
    frontier=["S"]; discovered={"S"}; visited=set(); parent={"S":None}; g={"S":0}; steps=[]
    def snap(cur=None,action="",note=""):
        steps.append({"current":cur,"frontier":list(frontier),"visited":set(visited),"parent":dict(parent),"g":dict(g),"h":dict(h),"action":action,"note":note})
    snap("S","START","Start at root S. The frontier contains S.")
    while frontier:
        if algorithm=="BFS": cur=frontier.pop(0)
        elif algorithm=="DFS": cur=frontier.pop()
        elif algorithm=="Greedy Best-First": frontier.sort(key=lambda n:(h[n],n)); cur=frontier.pop(0)
        else: frontier.sort(key=lambda n:(g[n]+h[n],h[n],n)); cur=frontier.pop(0)
        if cur in visited: continue
        visited.add(cur)
        if cur!="S":
            rule={"BFS":f"BFS removes {cur} from the front of the queue (FIFO).","DFS":f"DFS removes {cur} from the top of the stack (LIFO).","Greedy Best-First":f"Greedy selects {cur} because it has the smallest h(n).","A*":f"A* selects {cur} because it has the smallest f(n)=g(n)+h(n)."}[algorithm]
            snap(cur,"EXPLORE",rule)
        if cur==goal:
            snap(cur,"GOAL","Goal reached. Follow parent links backwards to recover the path."); break
        for nb in edges[cur]:
            if nb in discovered: continue
            discovered.add(nb); parent[nb]=cur; g[nb]=g[cur]+1; frontier.append(nb)
            if algorithm=="BFS": note=f"Discovered {nb}; append it to the queue."
            elif algorithm=="DFS": note=f"Discovered {nb}; push it onto the stack."
            elif algorithm=="Greedy Best-First": note=f"Discovered {nb}: h({nb})={h[nb]:.1f}. Smaller h is preferred."
            else: note=f"Discovered {nb}: g={g[nb]}, h={h[nb]:.1f}, f={g[nb]+h[nb]:.1f}."
            snap(cur,"ADD",note)
    path=[]
    if goal in visited:
        z=goal
        while z is not None: path.append(z); z=parent[z]
        path.reverse()
    snap(goal if goal in visited else None,"DONE","Final solution path is highlighted." if path else "Goal not reached.")
    # The complete expansion order is computed from the exact algorithm run.
    # It is intentionally different from the final parent-link path.
    expansion_sequence=[]
    for s in steps:
        if s["action"] in ("EXPLORE","GOAL") and s["current"] is not None:
            if not expansion_sequence or expansion_sequence[-1] != s["current"]:
                expansion_sequence.append(s["current"])

    key=f"tree_{algorithm}_{goal}"
    if st.session_state.get("tree_key")!=key:
        st.session_state.tree_key=key
        st.session_state.tree_i=0

    i=st.session_state.tree_i
    c1,c2,c3=st.columns(3)
    if c1.button("▶ Next Step",use_container_width=True):
        st.session_state.tree_i=min(i+1,len(steps)-1); st.rerun()
    if c2.button("⏮ Reset",use_container_width=True):
        st.session_state.tree_i=0; st.rerun()
    if c3.button("⏭ Finish",use_container_width=True):
        st.session_state.tree_i=len(steps)-1; st.rerun()

    i=st.session_state.tree_i
    step=steps[i]
    visible_path=path if i==len(steps)-1 else []

    # Draw the COMPLETE search tree from the beginning. Nodes are never hidden;
    # their visual state changes as the algorithm expands them.
    fig=go.Figure()

    # Edges first so that nodes remain visually prominent.
    for a,children in edges.items():
        for b in children:
            fig.add_trace(go.Scatter(
                x=[coords[a][0],coords[b][0]],
                y=[coords[a][1],coords[b][1]],
                mode="lines",
                line=dict(color="#94a3b8",width=2),
                showlegend=False, hoverinfo="skip"
            ))

    frontier_set=set(step["frontier"])
    visited_set=set(step["visited"])
    current_node=step["current"]

    # All nodes are visible before any expansion.
    all_nodes=list(edges.keys())
    base_nodes=[n for n in all_nodes if n not in ["S",goal]]
    fig.add_trace(go.Scatter(
        x=[coords[n][0] for n in base_nodes],
        y=[coords[n][1] for n in base_nodes],
        mode="markers+text",
        text=base_nodes,
        textposition="middle center",
        marker=dict(size=36,color="#e2e8f0",line=dict(color="#475569",width=2)),
        textfont=dict(color="#111827",size=12),
        name="Unvisited"
    ))

    # State overlays.
    state_specs=[
        ("Visited", visited_set, "#60a5fa", 40),
        ("Frontier", frontier_set, "#38bdf8", 44),
        ("Path (parent links)", set(visible_path), "#facc15", 48),
    ]
    for name,nodes,color,size in state_specs:
        nodes=[n for n in nodes if n not in ["S",goal,current_node]]
        if nodes:
            fig.add_trace(go.Scatter(
                x=[coords[n][0] for n in nodes],
                y=[coords[n][1] for n in nodes],
                mode="markers+text",
                text=nodes,textposition="middle center",
                marker=dict(size=size,color=color,line=dict(color="#111827",width=2)),
                textfont=dict(color="#111827",size=12),
                name=name
            ))

    fig.add_trace(go.Scatter(
        x=[coords["S"][0]],y=[coords["S"][1]],
        mode="markers+text",text=["S"],textposition="middle center",
        marker=dict(size=46,color="#22c55e",line=dict(color="#111827",width=2)),
        textfont=dict(color="#111827",size=14),name="START"
    ))
    fig.add_trace(go.Scatter(
        x=[coords[goal][0]],y=[coords[goal][1]],
        mode="markers+text",text=[goal],textposition="middle center",
        marker=dict(size=46,color="#ef4444",line=dict(color="#111827",width=2)),
        textfont=dict(color="#ffffff",size=14),name="GOAL"
    ))
    if current_node and current_node not in ["S",goal]:
        fig.add_trace(go.Scatter(
            x=[coords[current_node][0]],y=[coords[current_node][1]],
            mode="markers",
            marker=dict(size=56,color="#a855f7",opacity=.65,line=dict(color="#111827",width=3)),
            name="CURRENT"
        ))

    fig.update_layout(
        height=600,margin=dict(l=10,r=10,t=30,b=10),
        xaxis_visible=False,yaxis_visible=False,
        yaxis=dict(scaleanchor="x"),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0)
    )
    st.plotly_chart(fig,use_container_width=True)

    legend([
        ("⚪","UNVISITED","All nodes are visible from the start"),
        ("🟦","VISITED","Already expanded"),
        ("🔵","FRONTIER","Discovered and waiting"),
        ("🟪","CURRENT","Node being expanded now"),
        ("🟨","PATH","Final parent-link path only")
    ])

    rule={
        "BFS":"Queue / FIFO: explore level by level.",
        "DFS":"Stack / LIFO: go deep, then backtrack.",
        "Greedy Best-First":"Choose the frontier node with smallest h(n).",
        "A*":"Choose the frontier node with smallest f(n)=g(n)+h(n)."
    }[algorithm]
    explain(f"Rule used by {algorithm}",rule)

    cur=step["current"]
    if cur:
        gv=step["g"].get(cur,"—")
        hv=step["h"].get(cur,"—")
        fv=gv+hv if isinstance(gv,(int,float)) else "—"
        a,b,c,d,e=st.columns(5)
        a.metric("Step",f"{i+1}/{len(steps)}")
        b.metric("Current",cur)
        c.metric("g(n)",str(gv))
        d.metric("h(n)",f"{hv:.1f}" if isinstance(hv,(int,float)) else str(hv))
        e.metric("f(n)",f"{fv:.1f}" if isinstance(fv,float) else str(fv))

    what_now(step["note"])

    # Explicitly report the algorithm's actual expansion sequence.
    st.markdown("### Final expansion sequence")
    if expansion_sequence:
        seq_text=" → ".join(expansion_sequence)
        st.markdown(
            f'<div class="card"><h3>Exact {algorithm} expansion order</h3>'
            f'<p style="font-size:1.12rem;font-weight:800;">{seq_text}</p>'
            f'<p class="small">This is the order in which nodes are actually removed from '
            f'the frontier and expanded. It is <b>not</b> the optimized/shortest solution path.</p></div>',
            unsafe_allow_html=True
        )
        st.dataframe(
            pd.DataFrame({
                "Expansion #":range(1,len(expansion_sequence)+1),
                "Expanded node":expansion_sequence
            }),
            use_container_width=True,hide_index=True
        )
    if path:
        st.markdown("**Final parent-link path:** "+" → ".join(path))
# ---------- MODULE 3 ----------
elif page.startswith("03"):
    st.title("Module 3 • Logic")
    topic=st.sidebar.radio("Module 3 topics",["Propositional Logic","First-Order Logic","English → Logical Representation"])
    if topic=="Propositional Logic":
        explain("Propositional Logic","A proposition is either True or False. Connectives combine propositions.")
        A=st.checkbox("A = Student attended",True); B=st.checkbox("B = Assignment submitted",False)
        expr=st.selectbox("Expression",["A ∧ B","A ∨ B","A → B","¬A"])
        val=(A and B) if expr=="A ∧ B" else (A or B) if expr=="A ∨ B" else ((not A) or B) if expr=="A → B" else not A
        st.metric(expr,"TRUE" if val else "FALSE")
        what_now(f"A={A}, B={B}. The selected expression is {val}.")
    elif topic=="First-Order Logic":
        explain("First-Order Logic","FOL represents objects, properties and relationships using predicates, variables and quantifiers.")
        q=st.selectbox("Quantifier",["∀ (for all)","∃ (there exists)"]); pred=st.selectbox("Predicate",["Student(x)","Learns(x,AI)","Teaches(x,AI)","Uses(x,Sensors)"])
        st.code(f"{q[0]}x {pred}"); what_now("The quantifier determines whether the statement applies to every object or at least one object.")
    else:
        examples=[("Every student studies AI.","∀x (Student(x) → Studies(x,AI))"),("Some student studies AI.","∃x (Student(x) ∧ Studies(x,AI))"),("Every teacher teaches a course.","∀x (Teacher(x) → ∃y (Course(y) ∧ Teaches(x,y)))"),("Ravi is a student.","Student(Ravi)"),("All students submit assignments.","∀x (Student(x) → Submits(x,Assignment))"),("Some student likes every subject.","∃x (Student(x) ∧ ∀y (Subject(y) → Likes(x,y)))")]
        st.markdown("### English statement → logical representation")
        selected=st.selectbox("Choose example",[e[0] for e in examples]); logical=dict(examples)[selected]
        st.markdown(f'<div class="card"><h3>English</h3>{selected}<h3>Logical form</h3><code>{logical}</code></div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(examples,columns=["English statement","Logical representation"]),use_container_width=True,hide_index=True)
        what_now("Identify objects, properties and relationships in the English sentence. Then select predicates and the correct quantifier.")
# ---------- MODULE 4 ----------
elif page.startswith("04"):
    st.title("Module 4 • Bayesian Network & HMM")
    topic=st.sidebar.radio("Module 4 topics",["Bayesian Network","HMM — Bayes/Forward Calculation"])
    if topic=="Bayesian Network":
        explain("Bayes' rule","Bayes' rule updates a prior probability using evidence.")
        rain=st.slider("P(Rain)",0.0,1.0,0.30); ur=st.slider("P(Umbrella | Rain)",0.0,1.0,0.90); un=st.slider("P(Umbrella | No Rain)",0.0,1.0,0.20)
        pu=rain*ur+(1-rain)*un; post=rain*ur/pu if pu else 0
        st.metric("P(Rain | Umbrella)",f"{post:.2%}"); formula(r"P(Rain|Umbrella)=\frac{P(Umbrella|Rain)P(Rain)}{P(Umbrella)}")
        what_now(f"The prior P(Rain)={rain:.2%} becomes the posterior P(Rain|Umbrella)={post:.2%} after observing the evidence.")
    else:
        explain("HMM with Bayes/Forward calculation","The forward algorithm calculates the probability of the observed sequence. Bayes normalization at each time gives the posterior probability of each hidden state given all observations so far.")
        states=["Sunny","Rainy"]; obs_all=["Walk","Shop","Clean"]; pi=np.array([.6,.4]); T=np.array([[.8,.2],[.3,.7]]); E=np.array([[.6,.3,.1],[.1,.4,.5]]); idx={o:i for i,o in enumerate(obs_all)}
        seq=st.selectbox("Observed sequence",["Walk → Shop → Clean","Walk → Walk → Shop","Clean → Shop → Walk"]); obs=seq.split(" → ")
        c1,c2=st.columns(2); c1.dataframe(pd.DataFrame(T,index=states,columns=states),use_container_width=True); c2.dataframe(pd.DataFrame(E,index=states,columns=obs_all),use_container_width=True)
        st.write("Initial probabilities:",dict(zip(states,pi)))
        alpha=[]; a=pi*E[:,idx[obs[0]]]; alpha.append(a)
        for t in range(1,len(obs)): alpha.append((alpha[-1]@T)*E[:,idx[obs[t]]])
        rows=[]
        for t,o in enumerate(obs):
            raw=alpha[t]; prob=raw.sum(); post=raw/prob if prob else raw
            rows.append([t+1,o,raw[0],raw[1],prob,post[0],post[1]])
        st.markdown("### All instances")
        st.dataframe(pd.DataFrame(rows,columns=["t","Observation","α Sunny","α Rainy","P(O₁…Oₜ)","P(Sunny|O₁:t)","P(Rainy|O₁:t)"]),use_container_width=True,hide_index=True)
        final=float(alpha[-1].sum()); st.metric("Final probability of observed sequence",f"{final:.6f}")
        st.markdown("### Step-by-step calculation")
        for t,o in enumerate(obs):
            if t==0: st.latex(r"\alpha_1(s)=\pi_s b_s(o_1)")
            else: st.latex(r"\alpha_t(j)=\left[\sum_i\alpha_{t-1}(i)a_{ij}\right]b_j(o_t)")
            raw=alpha[t]; prob=raw.sum(); post=raw/prob
            st.write(f"t={t+1}, observation={o}: α(Sunny)={raw[0]:.6f}, α(Rainy)={raw[1]:.6f}, sequence probability={prob:.6f}")
            st.write(f"Bayes-normalized posterior: Sunny={post[0]:.2%}, Rainy={post[1]:.2%}")
        formula(r"P(S_t=s\mid O_{1:t})=\frac{\alpha_t(s)}{\sum_j\alpha_t(j)}")
        what_now(f"The final probability of {seq} is {final:.6f}. Every observation produces a new forward probability and a Bayes-normalized hidden-state posterior.")
# ---------- MODULE 5 ----------
elif page.startswith("05"):
    st.title("Module 5 • Data Preparation Laboratory")
    topic=st.sidebar.radio("Module 5 topics",["1. Create Dataset","2. Missing Values","3. Normalization","4. PCA"])
    if topic=="1. Create Dataset":
        explain("Create your dataset first","Choose number of classes, samples per class and feature size. The generated dataset is then used by the following experiments.")
        nc=st.number_input("Number of classes",2,8,3); spc=st.number_input("Samples per class",5,100,20); nf=st.number_input("Feature size",2,15,5)
        if st.button("Generate Dataset"):
            rng=np.random.default_rng(21); X=[]; y=[]
            for c in range(nc): X.append(rng.normal(c*2,0.8,(spc,nf))); y += [f"Class {c+1}"]*spc
            d=pd.DataFrame(np.vstack(X),columns=[f"F{i+1}" for i in range(nf)]); d["Class"]=y; st.session_state.prep_df=d
        if "prep_df" in st.session_state:
            d=st.session_state.prep_df; st.success(f"{len(d)} samples × {len(d.columns)-1} features; {d.Class.nunique()} classes; {len(d)//d.Class.nunique()} samples/class."); st.dataframe(d,use_container_width=True,hide_index=True)
            what_now("This dataset is the common input for missing-value, normalization and PCA demonstrations.")
        else: st.info("Choose the dataset dimensions and click Generate Dataset.")
    elif "prep_df" not in st.session_state:
        st.warning("Create a dataset first using 1. Create Dataset.")
    elif topic=="2. Missing Values":
        d=st.session_state.prep_df.copy(); fcols=[c for c in d.columns if c!="Class"]; rate=st.slider("Missing value percentage",0,50,10); method=st.selectbox("Fill method",["Mean of whole data","Median of whole data","Mean of same class","Median of same class","Global constant"]); constant=st.number_input("Global constant",value=0.0) if method=="Global constant" else 0.0
        rng=np.random.default_rng(9); miss=pd.DataFrame(False,index=d.index,columns=fcols)
        for c in fcols: miss.loc[rng.choice(d.index,max(1,int(len(d)*rate/100)),replace=False),c]=True
        w=d.copy()
        for c in fcols: w.loc[miss[c],c]=np.nan
        st.markdown("### 1. Data with missing cells")
        st.dataframe(w,use_container_width=True,hide_index=True)
        st.caption("Blank/empty cells are the missing values. This is what the student should identify before selecting an imputation method.")
        filled=w.copy()
        if method=="Mean of whole data":
            for c in fcols: filled[c]=filled[c].fillna(w[c].mean())
            explanation="Each blank is replaced by the mean of that feature across the whole dataset."
        elif method=="Median of whole data":
            for c in fcols: filled[c]=filled[c].fillna(w[c].median())
            explanation="Each blank is replaced by the median of that feature across the whole dataset."
        elif method=="Mean of same class":
            for c in fcols: filled[c]=filled.groupby("Class")[c].transform(lambda z:z.fillna(z.mean()))
            explanation="Each blank is replaced by the mean of that feature calculated only from the same class."
        elif method=="Median of same class":
            for c in fcols: filled[c]=filled.groupby("Class")[c].transform(lambda z:z.fillna(z.median()))
            explanation="Each blank is replaced by the median of that feature calculated only from the same class."
        else:
            filled=filled.fillna(constant); explanation=f"Every blank is replaced by the global constant {constant}."
        st.markdown("### 2. Filled data")
        st.dataframe(filled,use_container_width=True,hide_index=True)
        changes=[]
        for c in fcols:
            for ix in d.index[miss[c]]: changes.append([ix,c,d.loc[ix,"Class"],filled.loc[ix,c]])
        st.markdown("### 3. Actual replacement values"); st.dataframe(pd.DataFrame(changes,columns=["Row","Feature","Class","Inserted value"]),use_container_width=True,hide_index=True)
        what_now(explanation)
    elif topic=="3. Normalization":
        d=st.session_state.prep_df
        fcols=[c for c in d.columns if c!="Class"]
        c=st.selectbox("Feature",fcols)
        x=d[c].to_numpy(dtype=float)
        method=st.radio("Normalization method",["Min-Max Normalization","Z-Score Normalization"],horizontal=True)

        mn,mx=x.min(),x.max()
        mean=x.mean()
        std=x.std(ddof=0)

        if method=="Min-Max Normalization":
            norm=(x-mn)/(mx-mn) if mx!=mn else np.zeros_like(x)
            formula(r"x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}")
            explanation=f"{c} is rescaled to the range [0, 1]. The minimum becomes 0 and the maximum becomes 1."
            range_text=f"Range after scaling: [{norm.min():.3f}, {norm.max():.3f}]"
        else:
            norm=(x-mean)/std if std!=0 else np.zeros_like(x)
            formula(r"z=\frac{x-\mu}{\sigma}")
            explanation=f"{c} is centered around 0 and scaled by its standard deviation. The transformed feature has approximately mean 0 and standard deviation 1."
            range_text=f"Mean after scaling: {norm.mean():.3f} | Std. deviation after scaling: {norm.std(ddof=0):.3f}"

        st.dataframe(
            pd.DataFrame({"Original":x,"Normalized":norm}).head(20),
            use_container_width=True,hide_index=True
        )
        a,b,c1=st.columns(3)
        a.metric("Original minimum",f"{mn:.3f}")
        b.metric("Original maximum",f"{mx:.3f}")
        c1.metric("Original mean",f"{mean:.3f}")
        st.info(range_text)
        what_now(explanation)

    else:
        d=st.session_state.prep_df.copy()
        fcols=[c for c in d.columns if c!="Class"]

        if len(fcols)<2:
            st.error("PCA visualization needs at least two numeric features.")
        else:
            st.markdown("### PCA geometric visualization")
            st.write("Select two features for the 2-D PCA demonstration. The same graph is progressively built: raw data → eigenvalues/eigenvectors → principal axes → projections.")

            colx,coly=st.columns(2)
            xcol=colx.selectbox("X-axis feature",fcols,index=0)
            y_default=1 if len(fcols)>1 else 0
            ycol=coly.selectbox("Y-axis feature",fcols,index=y_default)
            standardize=st.checkbox("Standardize X and Y before PCA",value=False)

            if xcol==ycol:
                st.warning("Choose two different features for the X and Y axes.")
            else:
                X2=d[[xcol,ycol]].to_numpy(dtype=float)
                labels=d["Class"].astype(str).to_numpy() if "Class" in d.columns else np.array(["Data"]*len(d))

                # PCA in the selected 2-D feature space.
                mu=X2.mean(axis=0)
                Xc=X2-mu
                scale=np.ones(2)
                if standardize:
                    scale=X2.std(axis=0,ddof=0)
                    scale=np.where(scale==0,1.0,scale)
                    Xwork=Xc/scale
                else:
                    Xwork=Xc

                cov=np.cov(Xwork,rowvar=False,ddof=1)
                eigvals,eigvecs=np.linalg.eigh(cov)
                order=np.argsort(eigvals)[::-1]
                eigvals=eigvals[order]
                eigvecs=eigvecs[:,order]

                # Make eigenvector signs deterministic for a stable classroom plot.
                for j in range(2):
                    if eigvecs[0,j] < 0:
                        eigvecs[:,j]*=-1

                scores=Xwork @ eigvecs
                projected_work=scores @ eigvecs.T
                projected=X2 if standardize else (projected_work+mu)
                if standardize:
                    projected=projected_work*scale+mu

                total_var=eigvals.sum()
                ratios=eigvals/total_var if total_var>0 else np.zeros(2)

                stage=st.radio(
                    "Visualization stage",
                    ["1 • Raw data","2 • Eigenvalues + eigenvectors",
                     "3 • Principal axes","4 • Projections (complete)"],
                    index=3,horizontal=True
                )
                stage_num=int(stage[0])

                # A single consistent coordinate system is used throughout.
                fig=go.Figure()

                # Raw observations.
                classes=pd.unique(labels)
                for cls in classes:
                    mask=labels==cls
                    fig.add_trace(go.Scatter(
                        x=X2[mask,0],y=X2[mask,1],
                        mode="markers",
                        name=str(cls),
                        marker=dict(size=9,line=dict(color="#111827",width=1)),
                        hovertemplate=f"{xcol}: %{{x:.3f}}<br>{ycol}: %{{y:.3f}}<extra>{cls}</extra>"
                    ))

                # Mean.
                fig.add_trace(go.Scatter(
                    x=[mu[0]],y=[mu[1]],mode="markers+text",
                    text=["Mean"],textposition="top center",
                    marker=dict(size=13,symbol="x",color="#111827"),
                    name="Mean"
                ))

                # Plot extent used to draw principal axes and eigenvectors.
                span=np.ptp(X2,axis=0)
                radius=max(float(np.max(span))*0.42,1e-6)
                arrow_scale=radius*0.95

                if stage_num>=2:
                    # Eigenvectors are drawn as arrows from the mean.
                    for j,(name,dash) in enumerate([("Eigenvector 1","solid"),("Eigenvector 2","solid")]):
                        v=eigvecs[:,j]
                        if standardize:
                            # Map a direction in standardized coordinates back to raw X/Y.
                            raw_dir=v*scale
                            raw_dir=raw_dir/np.linalg.norm(raw_dir)
                        else:
                            raw_dir=v
                        end=mu+arrow_scale*raw_dir
                        fig.add_trace(go.Scatter(
                            x=[mu[0],end[0]],y=[mu[1],end[1]],
                            mode="lines+markers",
                            line=dict(width=4,dash=dash),
                            marker=dict(size=[5,10],symbol=["circle","arrow"]),
                            name=name
                        ))
                        end2=mu-arrow_scale*raw_dir
                        fig.add_trace(go.Scatter(
                            x=[mu[0],end2[0]],y=[mu[1],end2[1]],
                            mode="lines",
                            line=dict(width=2,dash="dot"),
                            showlegend=False
                        ))

                if stage_num>=3:
                    # Principal axes are the full lines through the mean in the
                    # eigenvector directions. These are drawn on the SAME graph.
                    line_len=radius*1.35
                    for j,name in enumerate(["PC1 axis","PC2 axis"]):
                        v=eigvecs[:,j]
                        if standardize:
                            raw_dir=v*scale
                            raw_dir=raw_dir/np.linalg.norm(raw_dir)
                        else:
                            raw_dir=v
                        p1=mu-line_len*raw_dir
                        p2=mu+line_len*raw_dir
                        fig.add_trace(go.Scatter(
                            x=[p1[0],p2[0]],y=[p1[1],p2[1]],
                            mode="lines",
                            line=dict(width=2,dash="dash"),
                            name=name
                        ))

                if stage_num>=4:
                    # Projection onto PC1 and PC2, shown by perpendicular
                    # projection segments. Each projected point is the PCA
                    # reconstruction from the selected component(s).
                    for j in range(len(X2)):
                        fig.add_trace(go.Scatter(
                            x=[X2[j,0],projected[j,0]],
                            y=[X2[j,1],projected[j,1]],
                            mode="lines",
                            line=dict(width=1,dash="dot"),
                            showlegend=False,
                            hoverinfo="skip"
                        ))
                    fig.add_trace(go.Scatter(
                        x=projected[:,0],y=projected[:,1],
                        mode="markers",
                        marker=dict(size=8,symbol="diamond"),
                        name="Projected points"
                    ))

                fig.update_layout(
                    height=650,
                    margin=dict(l=40,r=20,t=35,b=45),
                    xaxis_title=xcol,
                    yaxis_title=ycol,
                    title=f"PCA: {xcol} vs {ycol}",
                    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),
                    hovermode="closest"
                )
                fig.update_xaxes(zeroline=True,showgrid=True)
                fig.update_yaxes(zeroline=True,showgrid=True,scaleanchor="x",scaleratio=1)
                st.plotly_chart(fig,use_container_width=True)

                st.markdown("### Eigenvalues and eigenvectors")
                eig_df=pd.DataFrame({
                    "Component":["PC1","PC2"],
                    "Eigenvalue":eigvals,
                    "Explained variance":[f"{r:.2%}" for r in ratios],
                    "Eigenvector X":[eigvecs[0,0],eigvecs[0,1]],
                    "Eigenvector Y":[eigvecs[1,0],eigvecs[1,1]]
                })
                st.dataframe(eig_df,use_container_width=True,hide_index=True)

                a,b=st.columns(2)
                a.metric("PC1 variance",f"{ratios[0]:.2%}")
                b.metric("PC2 variance",f"{ratios[1]:.2%}")

                if stage_num>=4:
                    st.markdown("### PCA projections")
                    proj_df=pd.DataFrame({
                        xcol:X2[:,0], ycol:X2[:,1],
                        "PC1 score":scores[:,0],
                        "PC2 score":scores[:,1],
                        "Projected X":projected[:,0],
                        "Projected Y":projected[:,1],
                        "Class":labels
                    })
                    st.dataframe(proj_df.head(20),use_container_width=True,hide_index=True)

                explain(
                    "How the PCA graph is built",
                    "1) The original observations are plotted using the selected X and Y features. "
                    "2) The covariance matrix is decomposed to obtain eigenvalues and eigenvectors. "
                    "3) Eigenvectors define the principal directions; the principal axes are the same directions drawn as lines through the mean. "
                    "4) Finally, each observation is projected onto the principal-component space. "
                    "The eigenvalues indicate how much variance each principal direction captures."
                )
                what_now(
                    "The graph keeps the original X/Y coordinates throughout. "
                    "Eigenvectors show the directions of maximum and minimum variance, "
                    "principal axes pass through the data mean, and the final stage shows how observations are projected onto those axes."
                )
# ---------- MODULE 6 ----------
elif page.startswith("06"):
    st.title("Module 6 • Machine Learning Laboratory")
    topic=st.sidebar.selectbox("Module 6 topics",["Confusion Matrix","Simple Linear Regression","KNN","Decision Tree","Ensemble Learning","K-Means Clustering","Naïve Bayes","SVM"])
    if topic=="Confusion Matrix":
        explain("Manual confusion matrix","Rows = Actual class; columns = Predicted class. Enter any square matrix. For each class, the diagonal value is TP; the remaining row/column cells determine FN and FP; all remaining cells are TN.")
        n=st.number_input("Number of classes",2,10,3); labels=[f"Class {i+1}" for i in range(n)]
        if st.session_state.get("cm_shape")!=n: st.session_state.cm_shape=n; st.session_state.cm=np.eye(n,dtype=int)*8
        if st.button("Reset matrix"): st.session_state.cm=np.eye(n,dtype=int)*8
        ed=st.data_editor(pd.DataFrame(st.session_state.cm,index=labels,columns=labels),use_container_width=True,num_rows="fixed",key=f"cm_{n}"); st.session_state.cm=ed.to_numpy(dtype=int); cm=st.session_state.cm; total=cm.sum(); correct=np.trace(cm); rows=[]
        for i,l in enumerate(labels):
            tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=total-tp-fn-fp; acc=(tp+tn)/total if total else 0; pre=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; f1=2*pre*rec/(pre+rec) if pre+rec else 0; rows.append([l,tp,tn,fp,fn,acc,pre,rec,f1])
        res=pd.DataFrame(rows,columns=["Class","TP","TN","FP","FN","Class-wise Accuracy","Precision","Recall","F1"]); st.dataframe(res.style.format({"Class-wise Accuracy":"{:.2%}","Precision":"{:.2%}","Recall":"{:.2%}","F1":"{:.2%}"}),use_container_width=True,hide_index=True); st.metric("Overall Accuracy",f"{correct/total:.2%}" if total else "0.00%"); what_now("For each class, TP is the diagonal. FN is the rest of its actual row, FP is the rest of its predicted column, and TN is everything else.")
    elif topic=="Simple Linear Regression":
        x=np.arange(1,11); y=2.5*x+5+np.random.default_rng(5).normal(0,2.5,10); slope=st.slider("Slope β₁",0.,5.,2.5); intercept=st.slider("Intercept β₀",0.,15.,5.); pred=slope*x+intercept; fig=go.Figure(go.Scatter(x=x,y=y,mode="markers",name="Observed")); fig.add_trace(go.Scatter(x=x,y=pred,mode="lines",name="Line")); st.plotly_chart(fig,use_container_width=True); st.metric("MSE",f"{mean_squared_error(y,pred):.2f}"); formula(r"y=\beta_0+\beta_1x")
    elif topic=="KNN":
        k=st.slider("K",1,15,5,step=2); qx=st.slider("Query X",-5.,5.,0.); qy=st.slider("Query Y",-5.,5.,0.); dist=np.linalg.norm(Xblob-np.array([qx,qy]),axis=1); idx=np.argsort(dist)[:k]; pred=int(round(yblob[idx].mean())); fig=px.scatter(x=Xblob[:,0],y=Xblob[:,1],color=yblob.astype(str)); fig.add_trace(go.Scatter(x=Xblob[idx,0],y=Xblob[idx,1],mode="markers",marker=dict(size=17,symbol="circle-open"),name="Nearest")); fig.add_trace(go.Scatter(x=[qx],y=[qy],mode="markers",marker=dict(size=18,symbol="x"),name="Query")); st.plotly_chart(fig,use_container_width=True); what_now(f"The {k} nearest points vote for class {pred}.")
    elif topic=="Decision Tree":
        X=health[["Glucose","BMI"]]; y=(health.Class=="Positive").astype(int); depth=st.slider("Maximum depth",1,5,3); model=DecisionTreeClassifier(max_depth=depth,random_state=42).fit(X,y); fig,ax=plt.subplots(figsize=(12,6)); plot_tree(model,feature_names=X.columns,class_names=["Negative","Positive"],filled=True,ax=ax); st.pyplot(fig); what_now("Each internal node applies a feature threshold; follow the branch to reach a prediction leaf.")
    elif topic=="Ensemble Learning":
        explain("Ensemble methods","Bagging combines independently trained models, boosting trains models sequentially, and stacking learns how to combine different model types.")
        method=st.selectbox("Method",["Bagging","Boosting","Stacking"]); X=health[["Age","BMI","Blood Pressure","Glucose","Cholesterol","Exercise"]]; y=(health.Class=="Positive").astype(int); Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
        if method=="Bagging":
            from sklearn.ensemble import BaggingClassifier; ne=st.slider("Base models",5,100,20,5); model=BaggingClassifier(n_estimators=ne,random_state=42).fit(Xt,yt); note=f"{ne} base models are trained on bootstrap-resampled datasets and their predictions are combined."
        elif method=="Boosting":
            from sklearn.ensemble import AdaBoostClassifier; ne=st.slider("Boosting rounds",5,100,30,5); model=AdaBoostClassifier(n_estimators=ne,random_state=42).fit(Xt,yt); note=f"{ne} weak learners are trained sequentially, with later learners focusing more on difficult samples."
        else:
            from sklearn.ensemble import StackingClassifier; base=[("knn",KNeighborsClassifier(5)),("tree",DecisionTreeClassifier(max_depth=3,random_state=42)),("svm",SVC(probability=True))]; model=StackingClassifier(estimators=base,final_estimator=LogisticRegression(max_iter=1000)).fit(Xt,yt); note="KNN, Decision Tree and SVM predictions are combined by a logistic-regression meta-model."
        st.metric("Validation accuracy",f"{accuracy_score(yv,model.predict(Xv)):.2%}"); what_now(note)
    elif topic=="K-Means Clustering":
        explain("Iteration-wise K-Means","Each iteration has two phases: assign each point to its nearest centroid, then move each centroid to the mean of its assigned points.")
        k=st.slider("K",2,5,2); maxit=st.slider("Maximum iterations",1,10,6); X=Xblob.copy(); centers=X[np.linspace(0,len(X)-1,k).astype(int)].copy(); hist=[]
        for it in range(maxit):
            dist=np.linalg.norm(X[:,None,:]-centers[None,:,:],axis=2); lab=dist.argmin(axis=1); new=np.array([X[lab==j].mean(axis=0) if np.any(lab==j) else centers[j] for j in range(k)]); hist.append((centers.copy(),lab.copy(),new.copy()));
            if np.allclose(new,centers): break
            centers=new
        show=st.slider("Show iteration",1,len(hist),len(hist)); old,lab,new=hist[show-1]; fig=px.scatter(x=X[:,0],y=X[:,1],color=lab.astype(str),title=f"Iteration {show}"); fig.add_trace(go.Scatter(x=old[:,0],y=old[:,1],mode="markers",marker=dict(size=20,symbol="circle-open"),name="Old centroid")); fig.add_trace(go.Scatter(x=new[:,0],y=new[:,1],mode="markers",marker=dict(size=20,symbol="x"),name="New centroid")); st.plotly_chart(fig,use_container_width=True); st.dataframe(pd.DataFrame({"Cluster":[f"C{i+1}" for i in range(k)],"Old X":old[:,0],"Old Y":old[:,1],"New X":new[:,0],"New Y":new[:,1],"Movement":np.linalg.norm(new-old,axis=1)}),use_container_width=True,hide_index=True); what_now(f"Iteration {show}: points were assigned to the nearest centroid and each centroid was updated to the mean of its cluster.")
    elif topic=="Naïve Bayes":
        X=health[["Age","BMI","Blood Pressure","Glucose","Cholesterol","Exercise"]]; y=health.Class; Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); model=GaussianNB().fit(Xt,yt); st.metric("Accuracy",f"{accuracy_score(yv,model.predict(Xv)):.2%}"); probs=model.predict_proba(Xv.iloc[[0]])[0]; st.dataframe(pd.DataFrame({"Class":model.classes_,"Probability":probs}),hide_index=True); formula(r"P(C|X)\propto P(C)\prod_iP(x_i|C)")
    else:
        kernel=st.selectbox("Kernel",["linear","rbf","poly"]); model=SVC(kernel=kernel).fit(Xblob,yblob); xx,yy=np.meshgrid(np.linspace(-6,6,180),np.linspace(-6,6,180)); zz=model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape); fig=go.Figure(go.Contour(x=np.linspace(-6,6,180),y=np.linspace(-6,6,180),z=zz,showscale=False,opacity=.2)); fig.add_trace(go.Scatter(x=Xblob[:,0],y=Xblob[:,1],mode="markers",name="Samples")); st.plotly_chart(fig,use_container_width=True); what_now(f"SVM uses the {kernel} kernel to construct a decision boundary.")
# ---------- MODULE 7 ----------
elif page.startswith("07"):
    st.title("Module 7 • Deep Learning Laboratory")
    topic=st.sidebar.radio("Module 7 topics",["Neural Network — Calculate Outputs","Backpropagation","CNN — Convolution, Activation & Pooling"])
    if topic=="Neural Network — Calculate Outputs":
        explain("Example neural network","Enter inputs, weights and biases. The interface calculates each layer output step-by-step.")
        ni=st.number_input("Number of inputs",1,4,2); nh=st.number_input("Hidden neurons",1,4,2); no=st.number_input("Output neurons",1,3,1); actname=st.selectbox("Activation",["Sigmoid","ReLU","Tanh"])
        xs=[st.number_input(f"x{i+1}",value=float(i+1),key=f"v5x{i}") for i in range(ni)]; x=np.array(xs); W1=np.resize(np.array([[.5,-.4],[.3,.8]]),(ni,nh)); W1=st.data_editor(pd.DataFrame(W1,index=[f"x{i+1}" for i in range(ni)],columns=[f"h{i+1}" for i in range(nh)]),key=f"v5w1{ni}{nh}").to_numpy(dtype=float); b1=np.array([st.number_input(f"Hidden bias b{i+1}",value=.1,key=f"v5b1{i}") for i in range(nh)]); W2=st.data_editor(pd.DataFrame(np.ones((nh,no))*.4,index=[f"h{i+1}" for i in range(nh)],columns=[f"y{i+1}" for i in range(no)]),key=f"v5w2{nh}{no}").to_numpy(dtype=float); b2=np.array([st.number_input(f"Output bias b{i+1}",value=.1,key=f"v5b2{i}") for i in range(no)])
        def fn(z): return 1/(1+np.exp(-z)) if actname=="Sigmoid" else np.maximum(0,z) if actname=="ReLU" else np.tanh(z)
        z1=x@W1+b1; a1=fn(z1); z2=a1@W2+b2; a2=fn(z2); st.markdown(f"**Hidden pre-activation z₁:** `{np.round(z1,4)}`"); st.markdown(f"**Hidden output a₁:** `{np.round(a1,4)}`"); st.markdown(f"**Output pre-activation z₂:** `{np.round(z2,4)}`"); st.markdown(f"# Final output: `{np.round(a2,4)}`"); formula(r"z^{(l)}=a^{(l-1)}W^{(l)}+b^{(l)},\quad a^{(l)}=f(z^{(l)})"); what_now(f"The network first computes z₁=xW₁+b₁, applies {actname}, then computes z₂=a₁W₂+b₂ and applies the same activation. Final output = {np.round(a2,4)}.")
    elif topic=="Backpropagation":
        explain("Backpropagation","Forward pass → loss → backward gradients → weight update."); formula(r"W_{new}=W_{old}-\eta\frac{\partial L}{\partial W}"); lr=st.slider("Learning rate",.001,.2,.03); epochs=st.slider("Epochs",10,200,80); t=np.arange(epochs); st.plotly_chart(px.line(x=t,y=np.exp(-lr*8*t)+.02,title="Illustrative loss curve"),use_container_width=True); what_now("The optimizer uses gradients of the loss to update network weights.")
    else:
        explain("CNN numerical calculation","Choose a small image and kernel. The laboratory shows the actual convolution matrix, activation output and pooling output.")
        choice=st.selectbox("Input image",["Vertical line","Cross","Corner"]); img={"Vertical line":np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],float),"Cross":np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]],float),"Corner":np.array([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],float)}[choice]; kname=st.selectbox("3×3 kernel",["Vertical edge","Horizontal edge","Blur","Identity"]); kernels={"Vertical edge":[[-1,0,1],[-1,0,1],[-1,0,1]],"Horizontal edge":[[-1,-1,-1],[0,0,0],[1,1,1]],"Blur":[[1,1,1],[1,1,1],[1,1,1]],"Identity":[[0,0,0],[0,1,0],[0,0,0]]}; K=np.array(kernels[kname],float); pad=st.selectbox("Padding",["Valid","Same"]); stride=st.slider("Stride",1,2,1); A=np.pad(img,((1,1),(1,1))) if pad=="Same" else img; oh=(A.shape[0]-3)//stride+1; ow=(A.shape[1]-3)//stride+1; conv=np.array([[(A[i*stride:i*stride+3,j*stride:j*stride+3]*K).sum() for j in range(ow)] for i in range(oh)]); aname=st.selectbox("Activation",["None","ReLU","Sigmoid"]); activated=conv if aname=="None" else np.maximum(0,conv) if aname=="ReLU" else 1/(1+np.exp(-conv)); pool=st.selectbox("Pooling",["None","2×2 Max","2×2 Average"]); pooled=activated if pool=="None" else np.array([[(activated[i:i+2,j:j+2].max() if pool=="2×2 Max" else activated[i:i+2,j:j+2].mean()) for j in range(0,activated.shape[1]-1,2)] for i in range(0,activated.shape[0]-1,2)])
        c1,c2=st.columns(2); c1.dataframe(pd.DataFrame(img),hide_index=True); c2.dataframe(pd.DataFrame(K),hide_index=True); st.markdown("### Convolution output"); st.dataframe(pd.DataFrame(conv),hide_index=True); st.markdown("### Activation output"); st.dataframe(pd.DataFrame(activated),hide_index=True); st.markdown("### Pooling output"); st.dataframe(pd.DataFrame(pooled),hide_index=True); formula(r"F(i,j)=\sum_{u,v}I(i+u,j+v)K(u,v)"); what_now(f"The kernel slides across the image. Convolution creates the feature map; {aname} changes the values nonlinearly; {pool} reduces spatial size when selected.")
# ---------- PRACTICE ----------
else:
    st.title("🎯 Practice & Viva")
    qs=[
        ("Module 1","What does PEAS stand for?","Performance measure, Environment, Actuators and Sensors."),
        ("Module 2","What is the difference between Greedy Best-First and A*?","Greedy uses h(n); A* uses g(n)+h(n)."),
        ("Module 3","What extra elements does FOL introduce?","Objects, predicates, variables and quantifiers."),
        ("Module 4","What is hidden in an HMM?","The underlying state sequence."),
        ("Module 5","Why is PCA used?","For dimensionality reduction and visualization while retaining important variance."),
        ("Module 6","What do TP, TN, FP and FN represent?","The four basic outcomes of a binary classifier."),
        ("Module 7","What is backpropagation?","A method for calculating gradients of loss with respect to network parameters.")
    ]
    q=st.selectbox("Choose a question",range(len(qs)),format_func=lambda i:qs[i][0])
    st.markdown(f"### ❓ {qs[q][1]}")
    if st.button("Reveal Answer"):
        st.success(qs[q][2])
    st.info("Classroom tip: ask students to answer first, then use the module's interactive demo to justify the answer.")
