# STUDENT PERFORMANCE DASHBOARD 

## 1. Project Overview
This project is a Streamlit application that analyzes student academic transcripts to visualize performance trends, forecast future GPA using Machine Learning, and provide personalized study advice using Generative AI.

## 2. Links & Deliverables 🔗
* **Live App:** [(https://t3-final-project-fdsnqqtm3ixnyvai9cyglk.streamlit.app/)]
* **Presentation Slides:** [View PDF Slides](./project_slides.pdf)
* **Jupyter Notebook (Code):** [View Source Code](./t3-final-project.ipynb)
* **Notebook (Export):** [View Notebook Export](./t3-final-project.pdf)
## 3. Dataset & Data Sourcing 📊
* **Source:** The data is user-generated. Users upload a CSV transcript or manually input their grades.
* **Cleaning Process:**
    * Normalized column names to Title Case.
    * Mapped letter grades (A-F) to numerical points (5-0).
    * Calculated Weighted Points (Unit * Point).
    * Aggregated data by Semester to calculate GPA and CGPA.

## 4. Methodology 
* **Generative AI:** Uses the Hugging Face Inference API (`HuggingFaceTB/SmolLM3-3B`) to act as an academic coach.
* **Machine Learning:** Uses Scikit-Learn `LinearRegression` to analyze the trend of previous semesters and predict the next semester's GPA.
* **Visualization:** Uses Seaborn and Matplotlib to plot performance trajectories and grade heatmaps.

## 5. How to Run Locally
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
