"""
Student Expense Tracker - Hackathon Submission with ML Features
A complete expense tracking solution for college students with predictive analytics
Built with Streamlit, pandas, plotly, and scikit-learn

Requirements met:
✅ Add expense (date, amount, category, payment mode)
✅ View all expenses
✅ See total amount spent
✅ See category with highest spending
✅ Monthly budget with warning (bonus)
✅ 3+ sample expenses pre-loaded
✅ ML: Spending forecast/prediction
✅ ML: Anomaly detection for unusual spending
✅ ML: Spending pattern analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Student Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme matching your style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3a;
    --accent: #00e5ff;
    --accent2: #ff3b6b;
    --accent3: #a0ff6f;
    --text: #e8e8f0;
    --text-dim: #6b6b8a;
    --warn: #ffb547;
}

html, body, .stApp { 
    background: var(--bg) !important; 
    color: var(--text) !important; 
    font-family: 'JetBrains Mono', monospace !important; 
}

h1, h2, h3 { 
    font-family: 'Syne', sans-serif !important; 
    font-weight: 800 !important; 
}

[data-testid="stSidebar"] { 
    background: var(--surface) !important; 
    border-right: 1px solid var(--border) !important; 
}

.stTextInput > div > div > input, 
.stNumberInput > div > div > input,
.stSelectbox > div > div > select,
.stDateInput > div > div > input {
    background: var(--surface2) !important; 
    border: 1px solid var(--border) !important;
    color: var(--text) !important; 
    font-family: 'JetBrains Mono', monospace !important; 
    border-radius: 6px !important;
    padding: 8px 12px !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: var(--accent) !important; 
    box-shadow: 0 0 0 2px rgba(0,229,255,0.2) !important;
}

.stButton > button {
    background: var(--surface2) !important; 
    border: 1px solid var(--border) !important;
    color: var(--text) !important; 
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important; 
    border-radius: 6px !important; 
    transition: all 0.2s !important;
    padding: 8px 20px !important;
}

.stButton > button[kind="primary"] { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; 
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.stButton > button:hover { 
    border-color: var(--accent) !important; 
    box-shadow: 0 0 12px rgba(0,229,255,0.2) !important;
    transform: translateY(-2px);
}

[data-testid="stMetric"] { 
    background: var(--surface2) !important; 
    border: 1px solid var(--border) !important; 
    border-radius: 8px !important; 
    padding: 12px !important;
}

[data-testid="stMetricValue"] { 
    color: var(--accent) !important; 
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stExpander"] { 
    background: var(--surface) !important; 
    border: 1px solid var(--border) !important; 
    border-radius: 8px !important; 
}

.stSuccess { 
    background: rgba(160,255,111,0.08) !important; 
    border: 1px solid rgba(160,255,111,0.3) !important; 
}

.stWarning { 
    background: rgba(255,181,71,0.08) !important; 
    border: 1px solid rgba(255,181,71,0.3) !important; 
}

.stError { 
    background: rgba(255,59,107,0.08) !important; 
    border: 1px solid rgba(255,59,107,0.3) !important; 
}

.stInfo { 
    background: rgba(0,229,255,0.06) !important; 
    border: 1px solid rgba(0,229,255,0.2) !important; 
}

hr { 
    border-color: var(--border) !important; 
}

.stCaption { 
    color: var(--text-dim) !important; 
    font-size: 0.75rem !important; 
}

/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    transition: all 0.2s !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: rgba(0,229,255,0.04) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text) !important;
    background: rgba(0,229,255,0.03) !important;
}

[data-testid="stTabPanel"] {
    background: transparent !important;
    padding-top: 20px !important;
}

/* Dataframe styling */
.dataframe {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    width: 100% !important;
}

.dataframe thead tr th {
    background: var(--surface) !important;
    color: var(--accent) !important;
    font-weight: 700 !important;
    padding: 12px 16px !important;
    border-bottom: 2px solid var(--border) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.dataframe tbody tr td {
    padding: 10px 16px !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-size: 0.85rem !important;
}

.dataframe tbody tr:hover {
    background: rgba(0,229,255,0.03) !important;
}

/* Alert boxes */
.stAlert {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* Budget meter custom */
.budget-meter-container {
    background: var(--surface2);
    border-radius: 8px;
    padding: 16px;
    border: 1px solid var(--border);
    margin: 12px 0;
}

.budget-bar {
    height: 8px;
    border-radius: 4px;
    background: var(--surface);
    overflow: hidden;
    margin: 8px 0;
}

.budget-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Card styles */
.expense-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

.expense-card:hover {
    border-color: var(--accent);
}

.anomaly-card {
    background: rgba(255,59,107,0.05);
    border: 1px solid rgba(255,59,107,0.3);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

.category-tag {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

.tag-food { background: rgba(160,255,111,0.15); color: #a0ff6f; border: 1px solid rgba(160,255,111,0.3); }
.tag-travel { background: rgba(0,229,255,0.12); color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }
.tag-recharge { background: rgba(255,181,71,0.15); color: #ffb547; border: 1px solid rgba(255,181,71,0.3); }
.tag-other { background: rgba(107,107,138,0.15); color: #6b6b8a; border: 1px solid rgba(107,107,138,0.3); }

.stat-row { 
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
    padding: 6px 0; 
    border-bottom: 1px solid var(--border); 
    font-size: 0.8rem; 
}
.stat-label { color: var(--text-dim); }
.stat-value { color: var(--accent); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'expenses' not in st.session_state:
    # Sample expenses as required by hackathon brief
    st.session_state.expenses = [
        {'date': '2026-06-20', 'amount': 350.00, 'category': 'Food', 'payment_mode': 'UPI', 'description': 'Canteen lunch'},
        {'date': '2026-06-21', 'amount': 200.00, 'category': 'Travel', 'payment_mode': 'Cash', 'description': 'Metro ride'},
        {'date': '2026-06-22', 'amount': 299.00, 'category': 'Recharge', 'payment_mode': 'UPI', 'description': 'Mobile recharge'},
        {'date': '2026-06-23', 'amount': 120.00, 'category': 'Food', 'payment_mode': 'Cash', 'description': 'Evening snacks'},
        {'date': '2026-06-24', 'amount': 500.00, 'category': 'Travel', 'payment_mode': 'UPI', 'description': 'Bus pass'},
        {'date': '2026-06-25', 'amount': 75.00, 'category': 'Food', 'payment_mode': 'Cash', 'description': 'Chai & biscuits'},
        {'date': '2026-06-18', 'amount': 1500.00, 'category': 'Other', 'payment_mode': 'Card', 'description': 'Books'},
        {'date': '2026-06-15', 'amount': 50.00, 'category': 'Food', 'payment_mode': 'UPI', 'description': 'Tea & samosa'},
        {'date': '2026-06-12', 'amount': 800.00, 'category': 'Travel', 'payment_mode': 'UPI', 'description': 'Weekend trip'},
        {'date': '2026-06-10', 'amount': 450.00, 'category': 'Recharge', 'payment_mode': 'UPI', 'description': 'Internet bill'},
        {'date': '2026-06-08', 'amount': 2000.00, 'category': 'Other', 'payment_mode': 'Card', 'description': 'Clothing'},
    ]

if 'budget' not in st.session_state:
    st.session_state.budget = 5000.00

if 'month' not in st.session_state:
    st.session_state.month = datetime.now().strftime('%Y-%m')

# Helper functions
def get_category_tag(category):
    mapping = {'Food': 'tag-food', 'Travel': 'tag-travel', 'Recharge': 'tag-recharge', 'Other': 'tag-other'}
    return mapping.get(category, 'tag-other')

def get_month_expenses(month=None):
    if month is None:
        month = st.session_state.month
    month_date = datetime.strptime(month, '%Y-%m')
    return [
        e for e in st.session_state.expenses
        if datetime.strptime(e['date'], '%Y-%m-%d').strftime('%Y-%m') == month
    ]

def get_category_totals(expenses):
    totals = {}
    for e in expenses:
        totals[e['category']] = totals.get(e['category'], 0) + e['amount']
    return totals

def get_total_spent(expenses):
    return sum(e['amount'] for e in expenses)

def get_highest_category(expenses):
    totals = get_category_totals(expenses)
    if not totals:
        return None, 0
    max_cat = max(totals, key=totals.get)
    return max_cat, totals[max_cat]

def get_expenses_df():
    if not st.session_state.expenses:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.expenses)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    return df

def add_expense(date, amount, category, payment_mode, description):
    st.session_state.expenses.append({
        'date': date.strftime('%Y-%m-%d'),
        'amount': float(amount),
        'category': category,
        'payment_mode': payment_mode,
        'description': description
    })

def get_budget_status(expenses):
    total = get_total_spent(expenses)
    budget = st.session_state.budget
    if budget == 0:
        return 0, 'neutral', '#6b6b8a', 'No budget set'
    percentage = (total / budget) * 100
    if percentage < 70:
        return percentage, 'safe', '#a0ff6f', f'✅ On track! Spent ₹{total:,.2f} of ₹{budget:,.2f} budget'
    elif percentage < 90:
        return percentage, 'warning', '#ffb547', f'⚠️ Approaching limit! Spent ₹{total:,.2f} of ₹{budget:,.2f} budget'
    else:
        return percentage, 'danger', '#ff3b6b', f'🔴 Budget exceeded! Spent ₹{total:,.2f} of ₹{budget:,.2f} budget'

# ML Functions
def predict_spending(expenses):
    """Predict future spending using linear regression"""
    if len(expenses) < 3 or not SKLEARN_AVAILABLE:
        return None
    
    df = pd.DataFrame(expenses)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Prepare features (days from start)
    start_date = df['date'].min()
    df['days'] = (df['date'] - start_date).dt.days
    
    # Daily aggregation
    daily = df.groupby('days')['amount'].sum().reset_index()
    
    if len(daily) < 3:
        return None
    
    X = daily[['days']].values
    y = daily['amount'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 7 days
    last_day = daily['days'].max()
    future_days = np.arange(last_day + 1, last_day + 8).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    # Create future dates
    future_dates = [(start_date + timedelta(days=int(d))).strftime('%Y-%m-%d') for d in future_days.flatten()]
    
    return {
        'dates': future_dates,
        'predictions': predictions,
        'avg_daily': y.mean(),
        'trend': model.coef_[0],
        'next_7_days_total': predictions.sum()
    }

def detect_anomalies(expenses):
    """Detect anomalous spending using Isolation Forest"""
    if len(expenses) < 5 or not SKLEARN_AVAILABLE:
        return []
    
    df = pd.DataFrame(expenses)
    df['amount_scaled'] = StandardScaler().fit_transform(df[['amount']])
    
    # Use Isolation Forest
    iso_forest = IsolationForest(contamination=0.2, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(df[['amount_scaled']])
    
    # -1 indicates anomaly
    anomalies = df[df['anomaly'] == -1].copy()
    
    if len(anomalies) > 0:
        # Calculate z-scores for context
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        anomalies['z_score'] = (anomalies['amount'] - mean_amount) / std_amount
        
        return anomalies.to_dict('records')
    return []

def get_spending_patterns(expenses):
    """Analyze spending patterns (day of week, weekly trends)"""
    if len(expenses) < 3:
        return None
    
    df = pd.DataFrame(expenses)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week
    
    # Day of week averages
    dow_avg = df.groupby('day_of_week')['amount'].mean().sort_values(ascending=False)
    
    # Weekly totals
    weekly = df.groupby('week')['amount'].sum().reset_index()
    
    return {
        'day_of_week_avg': dow_avg.to_dict(),
        'week_totals': weekly.to_dict('records'),
        'most_expensive_day': dow_avg.index[0] if len(dow_avg) > 0 else None,
        'cheapest_day': dow_avg.index[-1] if len(dow_avg) > 0 else None,
    }

def get_category_insights(expenses):
    """Get ML-based category insights"""
    if len(expenses) < 5 or not SKLEARN_AVAILABLE:
        return None
    
    df = pd.DataFrame(expenses)
    df['date'] = pd.to_datetime(df['date'])
    
    # Category spending patterns
    category_data = []
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 1:
            # Calculate spending frequency and amount
            category_data.append({
                'category': cat,
                'total': cat_df['amount'].sum(),
                'count': len(cat_df),
                'avg': cat_df['amount'].mean(),
                'max': cat_df['amount'].max(),
                'min': cat_df['amount'].min(),
                'frequency_days': (cat_df['date'].max() - cat_df['date'].min()).days / max(1, len(cat_df) - 1)
            })
    
    return category_data

# Header
st.markdown("""
<div style="padding: 8px 0 24px 0;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b6b8a; letter-spacing:0.15em; margin-bottom:4px;">
        ◈ RETROD TRAVEL TECH · HACKATHON SUBMISSION
    </div>
    <h1 style="font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:#e8e8f0; margin:0; letter-spacing:-0.03em;">
        Student <span style="color:#00e5ff;">Expense</span> Tracker
    </h1>
    <p style="color:#6b6b8a; font-size:0.85rem; margin-top:6px; font-family:'JetBrains Mono',monospace;">
        Track · Analyze · Predict — Smart spending with ML
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1.1rem; color:#e8e8f0; margin-bottom:16px;">⚙ Dashboard Controls</div>', unsafe_allow_html=True)
    
    current_month = datetime.now().strftime('%Y-%m')
    month_options = []
    for i in range(12):
        month_date = datetime.now() - timedelta(days=30*i)
        month_options.append(month_date.strftime('%Y-%m'))
    
    selected_month = st.selectbox(
        "📅 Select Month",
        options=month_options,
        index=0,
        key='month_selector'
    )
    st.session_state.month = selected_month
    
    st.markdown("---")
    
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:0.9rem; color:#6b6b8a; margin-bottom:12px;">💰 Budget Settings</div>', unsafe_allow_html=True)
    
    new_budget = st.number_input(
        "Monthly Budget (₹)",
        min_value=0.0,
        max_value=100000.0,
        value=st.session_state.budget,
        step=500.0,
        format="%.2f"
    )
    if new_budget != st.session_state.budget:
        st.session_state.budget = new_budget
    
    st.markdown("---")
    
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:0.9rem; color:#6b6b8a; margin-bottom:12px;">📊 Quick Stats</div>', unsafe_allow_html=True)
    
    month_expenses = get_month_expenses()
    total = get_total_spent(month_expenses)
    count = len(month_expenses)
    highest_cat, highest_amount = get_highest_category(month_expenses)
    
    if count > 0:
        avg = total / count
        st.markdown(f"""
        <div class="stat-row"><span class="stat-label">Total Spent</span><span class="stat-value" style="color:#00e5ff;">₹{total:,.2f}</span></div>
        <div class="stat-row"><span class="stat-label">Transactions</span><span class="stat-value">{count}</span></div>
        <div class="stat-row"><span class="stat-label">Average</span><span class="stat-value">₹{avg:,.2f}</span></div>
        <div class="stat-row"><span class="stat-label">Top Category</span><span class="stat-value" style="color:#a0ff6f;">{highest_cat if highest_cat else 'N/A'}</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#6b6b8a; font-size:0.8rem; text-align:center; padding:20px 0;">
            No expenses yet this month
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("📥 Export CSV", use_container_width=True):
        df = get_expenses_df()
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"expenses_{selected_month}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv"
            )
        else:
            st.warning("No expenses to export")
    
    st.markdown("---")
    
    if st.button("🗑 Clear All Expenses", use_container_width=True):
        st.session_state.expenses = []
        st.rerun()

# Tabs
tab_add, tab_view, tab_analytics, tab_budget, tab_ml = st.tabs([
    "➕ Add Expense",
    "📋 View All",
    "📊 Analytics",
    "💰 Budget",
    "🤖 ML Insights"
])

# Tab 1: Add Expense
with tab_add:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b6b8a; letter-spacing:0.15em; margin-bottom:4px;">
            ◈ ADD NEW EXPENSE
        </div>
        <h2 style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e8f0; margin:0;">
            Record Your <span style="color:#a0ff6f;">Spending</span>
        </h2>
        <p style="color:#6b6b8a; font-size:0.8rem; margin-top:4px; font-family:'JetBrains Mono',monospace;">
            Track every rupee — UPI, Cash, or Card
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container():
            st.markdown('<div style="background:#12121a; border:1px solid #2a2a3a; border-radius:12px; padding:24px;">', unsafe_allow_html=True)
            
            col_date, col_amount = st.columns(2)
            with col_date:
                expense_date = st.date_input(
                    "📅 Date",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
            with col_amount:
                expense_amount = st.number_input(
                    "💰 Amount (₹)",
                    min_value=0.01,
                    max_value=100000.0,
                    value=100.0,
                    step=10.0,
                    format="%.2f"
                )
            
            col_category, col_payment = st.columns(2)
            with col_category:
                expense_category = st.selectbox(
                    "📂 Category",
                    options=['Food', 'Travel', 'Recharge', 'Other'],
                    index=0
                )
            with col_payment:
                expense_payment = st.selectbox(
                    "💳 Payment Mode",
                    options=['UPI', 'Cash', 'Card'],
                    index=0
                )
            
            expense_description = st.text_area(
                "📝 Description",
                placeholder="What did you spend on?",
                max_chars=100
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                if st.button("➕ Add Expense", type="primary", use_container_width=True):
                    if expense_amount > 0:
                        add_expense(
                            expense_date,
                            expense_amount,
                            expense_category,
                            expense_payment,
                            expense_description or expense_category
                        )
                        st.success(f"✅ Added ₹{expense_amount:,.2f} for {expense_category}")
                        st.rerun()
                    else:
                        st.error("Please enter a valid amount")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background:#12121a; border:1px solid #2a2a3a; border-radius:12px; padding:20px; height:100%;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.9rem; color:#00e5ff; margin-bottom:16px;">💡 Quick Tips</div>
            <div style="font-size:0.78rem; color:#6b6b8a; line-height:2;">
                <span style="color:#a0ff6f;">✓</span> Track <b>every</b> expense — even small ones<br>
                <span style="color:#a0ff6f;">✓</span> Categorize accurately for better insights<br>
                <span style="color:#a0ff6f;">✓</span> Use UPI/Cash/Card to see spending patterns<br>
                <span style="color:#a0ff6f;">✓</span> Set a monthly budget to control spending<br>
                <span style="color:#a0ff6f;">✓</span> Review weekly to stay on track<br>
                <span style="color:#a0ff6f;">✓</span> ML insights will detect anomalies automatically
            </div>
            <div style="margin-top:16px; background:rgba(0,229,255,0.05); border-radius:8px; padding:12px; border:1px solid rgba(0,229,255,0.1);">
                <div style="font-size:0.7rem; color:#6b6b8a;">🔹 11 sample expenses loaded</div>
                <div style="font-size:0.7rem; color:#6b6b8a;">🔹 ML models ready for prediction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: View All Expenses
with tab_view:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b6b8a; letter-spacing:0.15em; margin-bottom:4px;">
            ◈ EXPENSE LEDGER
        </div>
        <h2 style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e8f0; margin:0;">
            All <span style="color:#00e5ff;">Transactions</span>
        </h2>
        <p style="color:#6b6b8a; font-size:0.8rem; margin-top:4px; font-family:'JetBrains Mono',monospace;">
            Complete record of your spending
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df = get_expenses_df()
    
    if df.empty:
        st.info("No expenses recorded yet. Start adding expenses in the 'Add Expense' tab!")
    else:
        total = df['amount'].sum()
        count = len(df)
        avg = total / count if count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spent", f"₹{total:,.2f}", delta=None)
        with col2:
            st.metric("Transactions", count)
        with col3:
            st.metric("Average", f"₹{avg:,.2f}")
        with col4:
            highest_cat, highest_amt = get_highest_category(df.to_dict('records'))
            st.metric("Top Category", highest_cat or 'N/A', delta=f"₹{highest_amt:,.2f}" if highest_cat else None)
        
        st.markdown("---")
        
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1rem; color:#e8e8f0; margin-bottom:16px;">📋 Transaction History</div>', unsafe_allow_html=True)
        
        records = df.to_dict('records')
        for exp in records[:20]:
            category_tag = get_category_tag(exp['category'])
            date_display = pd.to_datetime(exp['date']).strftime('%b %d, %Y')
            
            st.markdown(f"""
            <div class="expense-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div style="display:flex; align-items:center; gap:12px; flex:1;">
                        <span style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.1rem; color:#00e5ff;">₹{exp['amount']:,.2f}</span>
                        <span class="category-tag {category_tag}">{exp['category']}</span>
                        <span style="font-size:0.75rem; color:#6b6b8a;">{date_display}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="font-size:0.75rem; color:#6b6b8a;">💳 {exp['payment_mode']}</span>
                        <span style="font-size:0.8rem; color:#a0a0b8;">{exp['description']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if len(records) > 20:
            st.caption(f"Showing 20 of {len(records)} transactions")
        
        with st.expander("📊 View Full Data Table"):
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "date": "Date",
                    "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                    "category": "Category",
                    "payment_mode": "Payment",
                    "description": "Description"
                }
            )

# Tab 3: Analytics
with tab_analytics:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b6b8a; letter-spacing:0.15em; margin-bottom:4px;">
            ◈ SPENDING ANALYTICS
        </div>
        <h2 style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e8f0; margin:0;">
            Category <span style="color:#ffb547;">Insights</span>
        </h2>
        <p style="color:#6b6b8a; font-size:0.8rem; margin-top:4px; font-family:'JetBrains Mono',monospace;">
            Understand your spending patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    month_expenses = get_month_expenses()
    
    if not month_expenses:
        st.info("No expenses recorded this month. Add some expenses to see analytics!")
    else:
        totals = get_category_totals(month_expenses)
        df_cat = pd.DataFrame([
            {'Category': cat, 'Amount': amt}
            for cat, amt in totals.items()
        ]).sort_values('Amount', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(
                df_cat,
                values='Amount',
                names='Category',
                color='Category',
                color_discrete_map={
                    'Food': '#a0ff6f',
                    'Travel': '#00e5ff',
                    'Recharge': '#ffb547',
                    'Other': '#6b6b8a'
                },
                hole=0.4,
                title='Spending by Category'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='JetBrains Mono, monospace', color='#e8e8f0'),
                title_font=dict(family='Syne, sans-serif', color='#e8e8f0'),
                legend=dict(font=dict(color='#6b6b8a'), bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                df_cat,
                x='Category',
                y='Amount',
                color='Category',
                color_discrete_map={
                    'Food': '#a0ff6f',
                    'Travel': '#00e5ff',
                    'Recharge': '#ffb547',
                    'Other': '#6b6b8a'
                },
                title='Category Spending',
                text_auto='.2f'
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='JetBrains Mono, monospace', color='#e8e8f0'),
                title_font=dict(family='Syne, sans-serif', color='#e8e8f0'),
                showlegend=False,
                yaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a')),
                xaxis=dict(tickfont=dict(color='#6b6b8a'))
            )
            fig2.update_traces(textposition='outside', textfont=dict(color='#a0a0b8'))
            st.plotly_chart(fig2, use_container_width=True)
        
        highest_cat, highest_amt = get_highest_category(month_expenses)
        if highest_cat:
            st.markdown(f"""
            <div style="background:rgba(160,255,111,0.05); border:1px solid rgba(160,255,111,0.2); border-radius:12px; padding:16px; margin-top:8px;">
                <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; color:#a0ff6f;">
                    🏆 Highest Spending Category: {highest_cat}
                </div>
                <div style="font-size:0.9rem; color:#e8e8f0; margin-top:4px;">
                    You spent ₹{highest_amt:,.2f} on {highest_cat.lower()}
                </div>
                <div style="font-size:0.75rem; color:#6b6b8a; margin-top:4px;">
                    {highest_amt/total*100:.1f}% of your total spending
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1rem; color:#e8e8f0; margin-bottom:16px;">📈 Daily Spending Pattern</div>', unsafe_allow_html=True)
        
        df_daily = pd.DataFrame(month_expenses)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        daily_totals = df_daily.groupby('date')['amount'].sum().reset_index()
        
        if len(daily_totals) > 1:
            fig3 = px.line(
                daily_totals,
                x='date',
                y='amount',
                title='Daily Spending Trend',
                labels={'date': 'Date', 'amount': 'Amount (₹)'}
            )
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='JetBrains Mono, monospace', color='#e8e8f0'),
                title_font=dict(family='Syne, sans-serif', color='#e8e8f0'),
                xaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a')),
                yaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a'))
            )
            fig3.update_traces(line=dict(color='#00e5ff', width=2), mode='lines+markers')
            st.plotly_chart(fig3, use_container_width=True)

# Tab 4: Budget Tracker
with tab_budget:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b6b8a; letter-spacing:0.15em; margin-bottom:4px;">
            ◈ BUDGET CONTROL
        </div>
        <h2 style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e8f0; margin:0;">
            Monthly <span style="color:#ff3b6b;">Budget</span> Tracker
        </h2>
        <p style="color:#6b6b8a; font-size:0.8rem; margin-top:4px; font-family:'JetBrains Mono',monospace;">
            Stay within your limits
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    month_expenses = get_month_expenses()
    total_spent = get_total_spent(month_expenses)
    budget = st.session_state.budget
    
    if budget == 0:
        st.warning("Please set a monthly budget in the sidebar to enable tracking.")
    else:
        percentage, status, color, message = get_budget_status(month_expenses)
        remaining = max(0, budget - total_spent)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="budget-meter-container">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="font-size:0.85rem; color:#e8e8f0; font-family:'Syne',sans-serif; font-weight:700;">Budget Progress</span>
                    <span style="font-size:0.85rem; color:{color}; font-family:'Syne',sans-serif; font-weight:700;">{percentage:.1f}%</span>
                </div>
                <div class="budget-bar">
                    <div class="budget-bar-fill" style="width:{min(percentage, 100)}%; background:{color};"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:8px;">
                    <span style="font-size:0.75rem; color:#6b6b8a;">₹0</span>
                    <span style="font-size:0.75rem; color:#6b6b8a;">₹{budget:,.2f}</span>
                </div>
                <div style="margin-top:12px; font-size:0.95rem; color:#e8e8f0;">
                    {message}
                </div>
                <div style="margin-top:4px; font-size:0.8rem; color:#6b6b8a;">
                    Remaining: ₹{remaining:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background:#12121a; border:1px solid #2a2a3a; border-radius:12px; padding:16px; height:100%;">
                <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem; color:#6b6b8a; margin-bottom:12px;">Quick Stats</div>
            """, unsafe_allow_html=True)
            
            days_in_month = datetime.now().day
            daily_avg = total_spent / days_in_month if days_in_month > 0 else 0
            projected = daily_avg * 30
            
            st.markdown(f"""
            <div class="stat-row"><span class="stat-label">Daily Average</span><span class="stat-value">₹{daily_avg:,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Projected Monthly</span><span class="stat-value" style="color:{'#ff3b6b' if projected > budget else '#a0ff6f'};">₹{projected:,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Spent Today</span><span class="stat-value" style="color:#00e5ff;">₹{sum(e['amount'] for e in month_expenses if e['date'] == datetime.now().strftime('%Y-%m-%d')):,.2f}</span></div>
            """, unsafe_allow_html=True)
            
            if projected > budget:
                st.markdown(f"""
                <div style="margin-top:12px; background:rgba(255,59,107,0.1); border:1px solid rgba(255,59,107,0.3); border-radius:8px; padding:10px;">
                    <div style="font-size:0.75rem; color:#ff3b6b; font-weight:600;">⚠️ WARNING</div>
                    <div style="font-size:0.7rem; color:#6b6b8a;">Projected spending exceeds budget by ₹{projected - budget:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Tab 5: ML Insights
with tab_ml:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b6b8a; letter-spacing:0.15em; margin-bottom:4px;">
            ◈ MACHINE LEARNING INSIGHTS
        </div>
        <h2 style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e8f0; margin:0;">
            Smart <span style="color:#00e5ff;">Predictions</span> & Anomalies
        </h2>
        <p style="color:#6b6b8a; font-size:0.8rem; margin-top:4px; font-family:'JetBrains Mono',monospace;">
            AI-powered spending analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not SKLEARN_AVAILABLE:
        st.warning("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")
        st.info("ML features require scikit-learn. Basic tracking still works!")
    
    all_expenses = st.session_state.expenses
    
    if len(all_expenses) < 3:
        st.info("Need at least 3 expenses for ML insights. Add more transactions!")
    else:
        # Anomaly Detection
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1.1rem; color:#ff3b6b; margin-bottom:16px;">🚨 Anomaly Detection</div>', unsafe_allow_html=True)
        
        anomalies = detect_anomalies(all_expenses)
        
        if anomalies and SKLEARN_AVAILABLE:
            st.warning(f"⚠️ {len(anomalies)} unusual spending transactions detected!")
            
            for anomaly in anomalies[:5]:
                date_display = pd.to_datetime(anomaly['date']).strftime('%b %d, %Y')
                st.markdown(f"""
                <div class="anomaly-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.2rem; color:#ff3b6b;">₹{anomaly['amount']:,.2f}</span>
                            <span class="category-tag {get_category_tag(anomaly['category'])}">{anomaly['category']}</span>
                            <span style="font-size:0.75rem; color:#6b6b8a; margin-left:8px;">{date_display}</span>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.7rem; color:#ffb547;">Z-Score: {anomaly.get('z_score', 0):.2f}</div>
                            <div style="font-size:0.65rem; color:#6b6b8a;">{anomaly['description']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption("⚠️ Anomalies are transactions that deviate significantly from your normal spending pattern")
        else:
            st.success("✅ No unusual spending patterns detected! Your transactions look normal.")
        
        st.markdown("---")
        
        # Spending Prediction
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1.1rem; color:#00e5ff; margin-bottom:16px;">📈 Spending Prediction</div>', unsafe_allow_html=True)
        
        prediction = predict_spending(all_expenses)
        
        if prediction and SKLEARN_AVAILABLE:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Daily Spend", f"₹{prediction['avg_daily']:,.2f}")
            with col2:
                trend = "📈 Increasing" if prediction['trend'] > 0 else "📉 Decreasing"
                st.metric("Spending Trend", trend, delta=f"{prediction['trend']:.2f}")
            with col3:
                st.metric("Next 7 Days Forecast", f"₹{prediction['next_7_days_total']:,.2f}")
            
            # Prediction chart
            df_pred = pd.DataFrame({
                'Date': prediction['dates'],
                'Predicted Amount': prediction['predictions']
            })
            
            fig_pred = px.line(
                df_pred,
                x='Date',
                y='Predicted Amount',
                title='7-Day Spending Forecast',
                labels={'Predicted Amount': 'Amount (₹)'}
            )
            fig_pred.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='JetBrains Mono, monospace', color='#e8e8f0'),
                title_font=dict(family='Syne, sans-serif', color='#e8e8f0'),
                xaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a')),
                yaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a'))
            )
            fig_pred.update_traces(line=dict(color='#ffb547', width=3), mode='lines+markers')
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.caption("📊 Based on linear regression of your spending history")
        else:
            st.info("Need more data (at least 3 days of transactions) for accurate predictions.")
        
        st.markdown("---")
        
        # Spending Patterns
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1.1rem; color:#a0ff6f; margin-bottom:16px;">📊 Spending Patterns</div>', unsafe_allow_html=True)
        
        patterns = get_spending_patterns(all_expenses)
        
        if patterns and SKLEARN_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📅 Day of Week Averages**")
                dow_df = pd.DataFrame([
                    {'Day': day, 'Avg Amount': amt}
                    for day, amt in patterns['day_of_week_avg'].items()
                ])
                
                fig_dow = px.bar(
                    dow_df,
                    x='Day',
                    y='Avg Amount',
                    title='Average Spending by Day',
                    color='Day'
                )
                fig_dow.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='JetBrains Mono, monospace', color='#e8e8f0'),
                    title_font=dict(family='Syne, sans-serif', color='#e8e8f0'),
                    showlegend=False,
                    yaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a')),
                    xaxis=dict(tickfont=dict(color='#6b6b8a'))
                )
                st.plotly_chart(fig_dow, use_container_width=True)
            
            with col2:
                if patterns['most_expensive_day']:
                    st.markdown(f"""
                    <div style="background:#12121a; border:1px solid #2a2a3a; border-radius:12px; padding:16px; margin-bottom:12px;">
                        <div style="font-size:0.8rem; color:#6b6b8a;">Most Expensive Day</div>
                        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#ff3b6b;">{patterns['most_expensive_day']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if patterns['cheapest_day']:
                    st.markdown(f"""
                    <div style="background:#12121a; border:1px solid #2a2a3a; border-radius:12px; padding:16px;">
                        <div style="font-size:0.8rem; color:#6b6b8a;">Cheapest Day</div>
                        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#a0ff6f;">{patterns['cheapest_day']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Weekly trend
            if patterns['week_totals']:
                st.markdown("**📊 Weekly Spending Trend**")
                weekly_df = pd.DataFrame(patterns['week_totals'])
                fig_weekly = px.line(
                    weekly_df,
                    x='week',
                    y='amount',
                    title='Weekly Spending Trend',
                    labels={'week': 'Week Number', 'amount': 'Total Amount (₹)'}
                )
                fig_weekly.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='JetBrains Mono, monospace', color='#e8e8f0'),
                    title_font=dict(family='Syne, sans-serif', color='#e8e8f0'),
                    xaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a')),
                    yaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#6b6b8a'))
                )
                fig_weekly.update_traces(line=dict(color='#00e5ff', width=3), mode='lines+markers')
                st.plotly_chart(fig_weekly, use_container_width=True)
        else:
            st.info("Need more data (at least 3 transactions) for pattern analysis.")
        
        st.markdown("---")
        
        # Category Insights
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:1.1rem; color:#ffb547; margin-bottom:16px;">💡 Category Insights</div>', unsafe_allow_html=True)
        
        category_insights = get_category_insights(all_expenses)
        
        if category_insights and SKLEARN_AVAILABLE:
            insight_df = pd.DataFrame(category_insights)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cat_insight = px.scatter(
                    insight_df,
                    x='count',
                    y='avg',
                    size='total',
                    color='category',
                    title='Category: Frequency vs Average Amount',
                    labels={'count': 'Transactions', 'avg': 'Avg Amount (₹)'}
                )
                fig_cat_insight.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='JetBrains Mono, monospace', color='#e8e8f0'),
                    title_font=dict(family='Syne, sans-serif', color='#e8e8f0'),
                    legend=dict(font=dict(color='#6b6b8a'), bgcolor='rgba(0,0,0,0)')
                )
                st.plotly_chart(fig_cat_insight, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Category Statistics**")
                st.dataframe(
                    insight_df[['category', 'total', 'count', 'avg']],
                    use_container_width=True,
                    column_config={
                        'category': 'Category',
                        'total': st.column_config.NumberColumn('Total', format='₹%.2f'),
                        'count': 'Count',
                        'avg': st.column_config.NumberColumn('Average', format='₹%.2f')
                    }
                )
                
                # Find most frequent category
                most_freq = insight_df.loc[insight_df['count'].idxmax()]
                st.markdown(f"""
                <div style="background:#12121a; border:1px solid #2a2a3a; border-radius:8px; padding:12px; margin-top:8px;">
                    <div style="font-size:0.75rem; color:#6b6b8a;">Most Frequent Category</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#00e5ff;">{most_freq['category']}</div>
                    <div style="font-size:0.8rem; color:#6b6b8a;">{most_freq['count']} transactions, avg ₹{most_freq['avg']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Add more transactions for detailed category insights.")

# Footer
st.markdown("---")
st.markdown("""
<div style="font-size:0.7rem; color:#2a2a3a; font-family:'JetBrains Mono',monospace; text-align:center; line-height:2;">
    Retrod Travel Tech · Hackathon Submission · Built with Python · Streamlit · ML Powered
</div>
""", unsafe_allow_html=True)