# College Football Wins Predictor (Multiple Linear Regression + Streamlit App)

## Overview
This project builds a multiple linear regression model to predict college football team wins based on key performance metrics, including offensive rank, defensive rank, and total touchdowns. The project includes both a Jupyter Notebook for analysis and a Streamlit web application for interactive predictions.

## Features
- Statistical analysis of team performance metrics
- Multiple linear regression model using scikit-learn
- Model evaluation using R² score
- Interactive Streamlit web app for real-time predictions
- Clean and reproducible project structure

## Dataset
The dataset includes the following features:
- Offensive ranking (`off_rank`)
- Defensive ranking (`def_rank`)
- Total touchdowns (`touchdowns`)
- Total wins (`win`)

The data is aggregated across multiple seasons and can be replaced with real-world college football data.

## Model
A multiple linear regression model is trained using:
X = [off_rank, def_rank, touchdowns]
y = win

The model outputs predicted wins and an R² score to evaluate performance.

## Project Structure
ncaa-football-wins-predictor/
├── app.py
├── data/ # NCAA CSV datasets
│   ├── cfb13.csv
│   ├── cfb14.csv
│   ├── cfb15.csv
│   ├── cfb16.csv
│   ├── cfb17.csv
│   ├── cfb18.csv
│   ├── cfb19.csv
│   ├── cfb20.csv
│   ├── cfb21.csv
│   └── cfb22.csv
├── requirements.txt
├── README.md

## How to Run

1. Clone the repository:
git clone <your-repo-url>
cd ncaa-football-wins-predictor

2. Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate (Mac/Linux)
venv\Scripts\activate (Windows)

3. Install dependencies:
pip install -r requirements.txt

4. Ensure CSV files are in the `data/` directory

5. Run the Streamlit app:
streamlit run app.py

6. Open the notebook:
JupyterLab

## Example Output
- Statistical summaries (min, mean, max touchdowns)
- Model R² score
- Predicted wins based on user input

## Future Improvements
- Incorporate additional features (yards, turnovers, strength of schedule)
- Test more advanced models (Random Forest, Gradient Boosting)
- Deploy the Streamlit app to the cloud

## License
MIT License