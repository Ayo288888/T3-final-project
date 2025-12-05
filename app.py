import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec # Fixed Import
from sklearn.linear_model import LinearRegression
from huggingface_hub import InferenceClient
import re

# --- 1. APP CONFIGURATION & UI SETUP ---
st.set_page_config(
    page_title="Academic Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #4B5563;}
    .highlight {color: #1E3A8A; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 2. DATA PROCESSING FUNCTIONS ---
GRADE_POINTS = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 0}

def get_point(grade):
    return GRADE_POINTS.get(str(grade).upper().strip(), 0)

def process_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Normalize column names
        df.columns = [c.strip().title() for c in df.columns] 
        
        # Validate required columns
        required = {'Semester', 'Course', 'Unit', 'Grade'}
        if not required.issubset(df.columns):
            return None, None, f"CSV is missing columns. Required: {required}"

        # Clean & Calculate
        df = df.dropna(subset=['Grade', 'Unit'])
        df['Point'] = df['Grade'].apply(get_point)
        df['Weighted_Point'] = df['Unit'] * df['Point']

        # Aggregation
        semester_stats = df.groupby('Semester').apply(
            lambda x: pd.Series({
                'Total_Units': x['Unit'].sum(),
                'Total_Points': x['Weighted_Point'].sum(),
                'GPA': x['Weighted_Point'].sum() / x['Unit'].sum()
            })
        ).reset_index()

        # Cumulative Stats
        semester_stats['Cumulative_Units'] = semester_stats['Total_Units'].cumsum()
        semester_stats['Cumulative_Points'] = semester_stats['Total_Points'].cumsum()
        semester_stats['CGPA'] = semester_stats['Cumulative_Points'] / semester_stats['Cumulative_Units']

        return df, semester_stats, None
    except Exception as e:
        return None, None, str(e)

# --- 3. VISUALIZATION FUNCTIONS ---
def plot_dashboard(df, stats):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(12, 10))
    # Fixed: Using explicitly imported GridSpec
    grid = GridSpec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.4)

    # A. Trajectory (Top Row)
    ax1 = fig.add_subplot(grid[0, :])
    sns.lineplot(data=stats, x='Semester', y='GPA', marker='o', label='Semester GPA', ax=ax1, color='#3B82F6', linewidth=2)
    sns.lineplot(data=stats, x='Semester', y='CGPA', marker='s', label='CGPA', ax=ax1, color='#10B981', linewidth=3)
    ax1.set_title('Performance Trajectory', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 5.2)
    ax1.legend()

    # B. Grade Distribution (Middle Left)
    ax2 = fig.add_subplot(grid[1, 0])
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F']
    sns.countplot(data=df, x='Grade', order=grade_order, palette='viridis', ax=ax2)
    ax2.set_title('Grade Distribution')
    
    # Simple workaround for bar_label type warning: just wrap in try/except or ignore
    try:
        ax2.bar_label(ax2.containers[0])
    except:
        pass

    # C. Heatmap (Middle Right)
    ax3 = fig.add_subplot(grid[1, 1])
    heatmap_data = pd.crosstab(df['Semester'], df['Grade'])
    for g in grade_order:
        if g not in heatmap_data.columns: heatmap_data[g] = 0
    heatmap_data = heatmap_data[grade_order]
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=False, ax=ax3, fmt='d')
    ax3.set_title('Grade Heatmap')

    # D. Unit Load vs GPA (Bottom Row)
    ax4 = fig.add_subplot(grid[2, :])
    sns.barplot(data=stats, x='Semester', y='Total_Units', ax=ax4, color='#9CA3AF', alpha=0.6)
    ax4_twin = ax4.twinx()
    sns.lineplot(data=stats, x=ax4.get_xticks(), y='GPA', marker='d', color='#EF4444', ax=ax4_twin, linewidth=2, label="GPA Impact")
    ax4.set_title('Workload (Units) vs Performance')
    
    return fig

# --- 4. AI & ML FUNCTIONS ---
def predict_gpa_ml(stats):
    if len(stats) < 2: return None, "Need at least 2 semesters of data for ML prediction."
    
    X = stats['Semester'].values.reshape(-1, 1)
    y = stats['GPA'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_sem = stats['Semester'].max() + 1
    # Fix: Ensure input is 2D array for Pylance happiness
    pred_input = [[next_sem]] 
    pred = model.predict(pred_input)[0]
    return min(5.0, max(0.0, pred)), None

def get_ai_coach_advice(api_key, name, current_cgpa, last_gpa, trend):
    if not api_key: return "⚠️ Please enter a Hugging Face API Token in the sidebar."
    
    try:
        client = InferenceClient(token=api_key)
        
        prompt = f"""
        You are an expert academic coach speaking to a student named {name}.
        
        Student Data:
        - Current CGPA: {current_cgpa:.2f} / 5.0
        - Last Semester GPA: {last_gpa:.2f}
        - Trend: {trend}
        
        Task:
        1. Address {name} directly with a warm, encouraging opening.
        2. Give 3 specific, high-impact study tips based on their trend.
        3. End with a short motivational punchline.
        
        Constraints:
        - Keep total response UNDER 200 words.
        - Do NOT use headers or markdown tags like [Student].
        - Be conversational and human-like.
        """
        
        response = client.chat_completion(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=[
                {"role": "system", "content": "You are a supportive, concise academic mentor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        
        text = response.choices[0].message.content
        # Fix: Ensure text is string for re.sub
        clean_text = re.sub(r'\[.*?\]', '', str(text)).strip()
        return clean_text
        
    except Exception as e:
        return f"AI Connection Error: {str(e)}"

# --- 5. MAIN APPLICATION UI ---

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=80)
    st.title("Settings")
    
    user_name = st.text_input("First Name", "Scholar")
    uploaded_file = st.file_uploader("Upload Transcript (CSV)", type="csv")
    
    st.markdown("---")
    st.subheader("🤖 AI Access")

    # --- NEW LOGIC START ---
    # Check if the key is stored in Streamlit Secrets
    if "HF_API_KEY" in st.secrets:
        api_key = st.secrets["HF_API_KEY"]
        st.success("AI Coach Connected ✅")
    else:
        # Fallback: Ask user if no secret is found (useful for local testing)
        api_key = st.text_input("Hugging Face API Key", type="password")
        st.caption("No system key found. Please enter your own.")
    # --- NEW LOGIC END ---
    
    st.markdown("---")
    st.caption("Don't have a file? Create `results.csv` with columns: Semester, Course, Unit, Grade")


# Main Content
st.markdown('<div class="main-header">🎓 Academic Analytics Suite</div>', unsafe_allow_html=True)
st.markdown(f"Welcome back, <span class='highlight'>{user_name}</span>! Let's optimize your grades.", unsafe_allow_html=True)

if uploaded_file:
    df, stats, error = process_data(uploaded_file)
    
    if error:
        st.error(f"❌ Error processing file: {error}")
    
    # Fix: Explicit check for None to satisfy VS Code linter
    elif df is not None and stats is not None:
        # --- Top Metrics Row ---
        col1, col2, col3, col4 = st.columns(4)
        
        curr_cgpa = stats['CGPA'].iloc[-1]
        last_gpa = stats['GPA'].iloc[-1]
        
        # Safe access for previous GPA
        if len(stats) > 1:
            prev_gpa = stats['GPA'].iloc[-2]
            trend_val = last_gpa - prev_gpa
        else:
            prev_gpa = last_gpa
            trend_val = 0.0
        
        col1.metric("Current CGPA", f"{curr_cgpa:.2f}")
        col2.metric("Last Sem GPA", f"{last_gpa:.2f}", f"{trend_val:.2f}")
        col3.metric("Total Units", f"{int(stats['Cumulative_Units'].iloc[-1])}")
        col4.metric("Semesters Done", f"{stats['Semester'].max()}")
        
        st.markdown("---")

        # --- Tabs ---
        tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "🔮 AI Predictor & Coach", "📝 Raw Transcript"])

        with tab1:
            st.markdown("### Visual Analytics")
            st.pyplot(plot_dashboard(df, stats))

        with tab3:
            st.markdown("### Transcript Data")
            st.dataframe(df, use_container_width=True)
            st.markdown("#### Semester Summaries")
            st.dataframe(stats, use_container_width=True)

        with tab2:
            st.subheader("AI-Powered Insights")
            
            c1, c2 = st.columns(2)
            
            # --- ML Prediction Section ---
            with c1:
                st.info("### 📈 Statistical Forecast")
                pred_gpa, err = predict_gpa_ml(stats)
                
                if err:
                    st.warning(err)
                elif pred_gpa is not None:
                    st.write("Based on your historical performance trend (Linear Regression):")
                    st.metric(label="Predicted Next Semester GPA", value=f"{pred_gpa:.2f}")
                    
                    if pred_gpa > curr_cgpa:
                        st.success("You are statistically trending upwards! 🚀")
                    else:
                        st.warning("Your trend suggests a slight dip. Time to buckle up! 🛡️")

            # --- LLM Coach Section ---
            with c2:
                st.success(f"### 🧠 Coach {user_name}'s Corner")
                if st.button("Get Personalized Advice"):
                    if not api_key:
                        st.error("Please provide an API Key in the sidebar first!")
                    else:
                        trend_desc = "Improving" if trend_val >= 0 else "Declining"
                        
                        with st.spinner("Analyzing your academic profile..."):
                            advice = get_ai_coach_advice(api_key, user_name, curr_cgpa, last_gpa, trend_desc)
                            
                        st.markdown(f"**Coach says:**\n\n{advice}")

else:
    # Empty State
    st.info("👆 Please upload your `csv` file in the sidebar to generate your dashboard.")
    
    st.markdown("### CSV Format Example")
    st.code("""
Semester,Course,Unit,Grade
1,MTH101,3,A
1,PHY101,2,B
2,GST102,2,A
    """, language="csv")