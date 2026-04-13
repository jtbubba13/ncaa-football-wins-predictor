import pandas as pd
import numpy as np
import glob
import os
import re
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats

# ----------------------------------
# Utility Functions (from your code)
# ----------------------------------

def snake_cols(dataframe):
    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
    )

def remove_conferences(dataframe):
    for i, tv in enumerate(dataframe["team"].values):
        tv = re.sub(r"\(.*?\)", "", tv)
        dataframe.at[i, 'team'] = tv

def get_stats(stats_list, stats_arr):
    return {
        "mean": stats_list.mean(),
        "median": stats_list.median(),
        "mode": stats_list.mode()[0],
        "std_dev": stats_list.std(),
        "var": stats_list.var(),
        "percentile_75": np.percentile(stats_arr, 75)
    }

# -----------------------------
# Load + Process Data
# -----------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "data")
csv_files = glob.glob(os.path.join(path, "*.csv"))
cfb_values = ['team', 'win', 'loss', 'off_rank', 'def_rank', 'touchdowns', 'total_points', 'games']

df_all = pd.DataFrame()
year = 2013
i = 0
max_team_count = 0

for f in csv_files:
    filename = f.split("\\")[-1].upper()

    if filename == "CFB22.CSV":
        df22 = pd.read_csv(f)
        snake_cols(df22)
        remove_conferences(df22)
        df22['team'] = df22['team'] + '2022'
        df22[['win', 'loss']] = df22['win_loss'].str.split('-', expand=True).astype(int)
        df22.drop(['win_loss'], axis=1, inplace=True)
        df22 = df22[cfb_values]
        continue

    df = pd.read_csv(f)
    snake_cols(df)
    remove_conferences(df)

    df['team'] = df['team'] + str(year + i)

    if filename == "CFB21.CSV":
        df[['win', 'loss']] = df['win_loss'].str.split('-', expand=True).astype(int)
        df.drop(['win_loss'], axis=1, inplace=True)

    df = df[cfb_values]

    team_count = df['team'].count()
    max_team_count = max(max_team_count, team_count)

    temp_frame = pd.concat([df_all, df])
    df_all = temp_frame
    i += 1

df_all.dropna(axis=1, inplace=True)
df_all.set_index('team', inplace=True, drop=False)

# -----------------------------
# Train Model
# -----------------------------

X = df_all[["off_rank", "def_rank", "touchdowns"]]
y = df_all["win"]

model = linear_model.LinearRegression()
model.fit(X, y)

score = model.score(X, y)

pd.Series(model.coef_, index=X.columns)

# -----------------------------
# UI
# -----------------------------

st.title("🏈 College Football Wins Predictor")

st.markdown("### 🧠 What this model does")
st.write("Predicts college football wins using offensive rank, defensive rank, and total touchdowns using multiple linear regression.")

st.markdown("### 📊 Model Features")
st.write("Offensive Rank, Defensive Rank, Touchdowns")

# -----------------------------
# Prediction FIRST (hook user)
# -----------------------------

st.subheader("🔮 Predict Team Wins")

off_rank = st.number_input("Offensive Rank", min_value=1, max_value=int(max_team_count), value=65)
def_rank = st.number_input("Defensive Rank", min_value=1, max_value=int(max_team_count), value=65)
touchdowns = st.number_input("Total Touchdowns", min_value=0, max_value=100, value=45)
max_games = df_all['games'].max()

if st.button("Predict Wins"):
    prediction = model.predict([[off_rank, def_rank, touchdowns]])
    pred = int(round(prediction[0]))

    if pred > max_games:
        pred = max_games
    elif pred < 0:
        pred = 0

    st.success(f"Predicted Wins: {pred}")

    # Show real teams with similar results (nice touch)
    value = int(pred)
    output = df22.query('win == @value')
    st.write("Teams with similar win totals (2022):")
    st.dataframe(output[['team', 'win', 'off_rank', 'def_rank', 'touchdowns']])

# -----------------------------
# Model performance
# -----------------------------

st.subheader("📈 Model Performance")
st.write(f"R² Score: {score:.4f}")

# -----------------------------
# Scatter Plot (main insight)
# -----------------------------

st.subheader("📊 Variable Relationship to Wins")

option = st.selectbox(
    "Choose variable to analyze",
    ["Touchdowns", "Offensive Rank", "Defensive Rank"]
)

color_map = {
    "Touchdowns": ("steelblue", "darkred"),
    "Offensive Rank": ("green", "black"),
    "Defensive Rank": ("purple", "orange")
}

if option == "Touchdowns":
    x = np.array(df_all['touchdowns'])
elif option == "Offensive Rank":
    x = np.array(df_all['off_rank'])
else:
    x = np.array(df_all['def_rank'])

y = np.array(df_all['win'])

scatter_color, line_color = color_map[option]

fig, ax = plt.subplots()

# Scatter
ax.scatter(x, y, alpha=0.6, color=scatter_color)

# Regression line
slope, intercept, r, p, std_err = stats.linregress(x, y)
sorted_idx = np.argsort(x)
x_sorted = x[sorted_idx]
y_sorted_line = slope * x_sorted + intercept

ax.plot(x_sorted, y_sorted_line, color=line_color)

# Labels
ax.set_xlabel(option)
ax.set_ylabel("Wins")
ax.set_title(f"{option} vs Wins (r = {r:.2f})")

ax.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('white')

st.pyplot(fig)

st.write(f"Correlation (r): {r:.3f}")
st.write(f"P-value: {p:.10e}")

# -----------------------------
# Stats AFTER visualization
# -----------------------------

stats_td = get_stats(df_all['touchdowns'], np.array(df_all['touchdowns']))

st.subheader("📊 Touchdown Summary Statistics")
st.write(f"Min: {df_all['touchdowns'].min():.2f}")
st.write(f"Mean: {stats_td['mean']:.2f}")
st.write(f"Max: {df_all['touchdowns'].max():.2f}")

st.markdown(f"Max teams in dataset: **{max_team_count}**")

# -----------------------------
# Raw Data LAST (optional)
# -----------------------------

if st.checkbox("Show Raw Data (Used to Train Algorithm)"):
    st.dataframe(df_all)

with st.expander("Show Development / Reference Code"):
    st.code("""    
# -----------------------------
# Scatter Plot that shows different correlations
# -----------------------------

option = st.selectbox(
    "Choose variable to analyze",
    ["Touchdowns", "Offensive Rank", "Defensive Rank"]
)

if option == "Touchdowns":
    x = np.array(df_all['touchdowns'])
elif option == "Offensive Rank":
    x = np.array(df_all['off_rank'])
else:
    x = np.array(df_all['def_rank'])

# y = df_all["win"]

st.subheader(option + " vs Wins")

# x = np.array(df_all['touchdowns'])
y = np.array(df_all['win'])

# Scatter
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.6, color='steelblue')

# Sorted regression line
slope, intercept, r, p, std_err = stats.linregress(x, y)
sorted_idx = np.argsort(x)
x_sorted = x[sorted_idx]
y_sorted_line = slope * x_sorted + intercept
ax.plot(x_sorted, y_sorted_line, color='darkred')

# Labels
ax.set_xlabel(option)
ax.set_ylabel("Wins")
ax.set_title(option + f" vs Wins (r = {r:.2f})")

st.write(f"Correlation (r): {r:.3f}")
st.write(f"P-value: {p:.10e}")
ax.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('white')
st.pyplot(fig)

st.subheader("Offensive & Defensive Rank vs Wins")

fig, ax = plt.subplots()

# Defensive Rank
x_def = np.array(df_all["def_rank"])
y = np.array(df_all["win"])
slope_d, intercept_d, r_d, _, _ = stats.linregress(x_def, y)

sorted_idx = np.argsort(x_def)
ax.scatter(x_def, y, alpha=0.5, label="Def Rank")
ax.plot(x_def[sorted_idx], slope_d * x_def[sorted_idx] + intercept_d)

# Offensive Rank
x_off = np.array(df_all["off_rank"])
slope_o, intercept_o, r_o, _, _ = stats.linregress(x_off, y)

sorted_idx = np.argsort(x_off)
ax.scatter(x_off, y, alpha=0.5, label="Off Rank")
ax.plot(x_off[sorted_idx], slope_o * x_off[sorted_idx] + intercept_o)

ax.set_xlabel("Rank")
ax.set_ylabel("Wins")
ax.legend()

st.pyplot(fig)
""")
