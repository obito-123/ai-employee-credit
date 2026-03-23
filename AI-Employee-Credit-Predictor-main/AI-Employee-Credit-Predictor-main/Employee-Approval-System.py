from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ---------------------------------------------------------
# 🧠 AI ENGINE: Training the Model
# ---------------------------------------------------------
# 1=High/Good/Yes, 0=Low/Poor/No
data = {
    'income': [1, 1, 1, 0, 0, 0],
    'credit': [1, 1, 0, 1, 0, 0],
    'employed': [1, 0, 1, 1, 1, 0],
    'approved': [1, 1, 0, 1, 0, 0]  # Target
}
df = pd.DataFrame(data)
model = RandomForestClassifier(n_estimators=10)
model.fit(df[['income', 'credit', 'employed']], df['approved'])

# ---------------------------------------------------------
# 🎨 UI DESIGN: Professional Dashboard
# ---------------------------------------------------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Loan Predictor</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: white; display: flex; justify-content: center; padding-top: 50px; }
        .container { background: #1e293b; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); width: 450px; }
        h2 { color: #38bdf8; text-align: center; margin-bottom: 25px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #94a3b8; font-size: 14px; }
        select, button { width: 100%; padding: 12px; border-radius: 8px; border: none; font-size: 16px; }
        select { background: #334155; color: white; }
        button { background: #38bdf8; color: #0f172a; font-weight: bold; cursor: pointer; margin-top: 10px; }
        button:hover { background: #0ea5e9; }
        .result-box { margin-top: 30px; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 18px; display: {{ 'block' if result else 'none' }}; }
        .Approved { background: #065f46; color: #34d399; border: 1px solid #059669; }
        .Rejected { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🤖 AI Credit Predictor</h2>
        <form method="POST">
            <div class="form-group">
                <label>Income Level</label>
                <select name="income">
                    <option value="1">High Income</option>
                    <option value="0">Low Income</option>
                </select>
            </div>
            <div class="form-group">
                <label>Credit Score</label>
                <select name="credit">
                    <option value="1">Good Score</option>
                    <option value="0">Poor Score</option>
                </select>
            </div>
            <div class="form-group">
                <label>Employment Status</label>
                <select name="employed">
                    <option value="1">Currently Employed</option>
                    <option value="0">Unemployed</option>
                </select>
            </div>
            <button type="submit">Predict Eligibility</button>
        </form>

        {% if result %}
        <div class="result-box {{ result }}">
            Result: {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        # Get data from UI
        inc = int(request.form['income'])
        crd = int(request.form['credit'])
        emp = int(request.form['employed'])
        
        # AI Prediction
        pred = model.predict([[inc, crd, emp]])[0]
        result = "Approved" if pred == 1 else "Rejected"
        
    return render_template_string(html_template, result=result)

if __name__ == '__main__':
    app.run(debug=True)