#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import dask.dataframe as dd
import dask.array as da
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

# In[147]:


df_initial = pd.read_csv('financial_risk_assessment.csv')


# In[148]:


# === Imputation Functions ===
def mice_imputation(df_initial):
    mice_imputer = IterativeImputer(random_state=42, max_iter=10, n_nearest_features=5)
    imputed_data = mice_imputer.fit_transform(df_initial)
    return pd.DataFrame(imputed_data, columns=df_initial.columns)

def knn_imputation(df_initial):
    knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
    imputed_data = knn_imputer.fit_transform(df_initial)
    return pd.DataFrame(imputed_data, columns=df_initial.columns)

def impute_data(df_initial, method='mice'):
    df_copy = df_initial.copy()
    scaler = StandardScaler()
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
    df_copy[numeric_columns] = scaler.fit_transform(df_copy[numeric_columns])

    if method == 'mice':
        imputed_df = mice_imputation(df_copy)
    elif method == 'knn':
        imputed_df = knn_imputation(df_copy)

    imputed_df[numeric_columns] = scaler.inverse_transform(imputed_df[numeric_columns])
    return imputed_df

# Preprocess and Fill Missing Data
df_initial_numeric = df_initial.select_dtypes(include=[np.number])
df_filled = df_initial.copy()
imputed_df_mice = impute_data(df_initial_numeric, method='mice')

for column in df_initial.columns:
    if column in imputed_df_mice.columns:
        nan_mask = df_initial[column].isna()
        df_filled.loc[nan_mask, column] = imputed_df_mice.loc[nan_mask, column]

# === Dash App Setup ===
app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'])

server = app.server

app.layout = html.Div([
    # Page Title
    html.H1("Comprehensive Loan Analysis Dashboard", 
            style={'textAlign': 'center', 'color': '#1E90FF', 'marginBottom': '30px'}),

    # Filters Section 
    html.Div([
        html.Div([
            html.Label("Select Gender:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='gender-filter',
                options=[{'label': gender, 'value': gender} for gender in df_filled['Gender'].unique()],
                multi=True,
                value=df_filled['Gender'].unique(),
                style={'width': '100%'}
            )
        ], className="col-md-4"),
        
        html.Div([
            html.Label("Select Risk Rating:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='risk-filter',
                options=[{'label': rating, 'value': rating} for rating in df_filled['Risk Rating'].unique()],
                multi=True,
                value=df_filled['Risk Rating'].unique(),
                style={'width': '100%'}
            )
        ], className="col-md-4"),
        
        html.Div([
            html.Label("Select Loan Purpose:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='purpose-filter',
                options=[{'label': purpose, 'value': purpose} for purpose in df_filled['Loan Purpose'].unique()],
                multi=True,
                value=df_filled['Loan Purpose'].unique(),
                style={'width': '100%'}
            )
        ], className="col-md-4"),
    ], className="row", style={'marginBottom': '20px'}),
    
    # Key Metrics Section 
    html.Div([
        html.H4("Key Metrics", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div(id='key-metrics', className="d-flex justify-content-center gap-3 flex-wrap")
    ], style={'backgroundColor': '#F0F8FF', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Main Visualizations 
    html.Div([
        html.Div([
            dcc.Graph(id='trend-analysis', style={'height': '400px'}),
            dcc.Graph(id='age-distribution', style={'height': '400px'})
        ], className="row"),
        
        html.Div([
            dcc.Graph(id='income-distribution', style={'height': '400px'}),
            dcc.Graph(id='loan-amount-boxplot', style={'height': '400px'})
        ], className="row"),
        
        html.Div([
            dcc.Graph(id='credit-score-scatter', style={'height': '400px'}),
            dcc.Graph(id='risk-heatmap', style={'height': '400px'})
        ], className="row"),
        
        html.Div([
            dcc.Graph(id='approval-prediction', style={'height': '500px'})
        ], className="row"),
    ], style={'maxWidth': '1200px', 'margin': 'auto'})
])


@app.callback(
    [Output('key-metrics', 'children'),
     Output('trend-analysis', 'figure'),
     Output('age-distribution', 'figure'),
     Output('income-distribution', 'figure'),
     Output('loan-amount-boxplot', 'figure'),
     Output('credit-score-scatter', 'figure'),
     Output('risk-heatmap', 'figure'),
     Output('approval-prediction', 'figure')],
    [Input('gender-filter', 'value'),
     Input('risk-filter', 'value'),
     Input('purpose-filter', 'value')]
)
def update_dashboard(selected_genders, selected_risks, selected_purposes):
    filtered_df = df_filled[
        df_filled['Gender'].isin(selected_genders) & 
        df_filled['Risk Rating'].isin(selected_risks) &
        df_filled['Loan Purpose'].isin(selected_purposes)
    ]

    metrics = [
        html.Div([html.H6("Avg Loan Amount"), html.H4(f"${filtered_df['Loan Amount'].mean():,.2f}")]),
        html.Div([html.H6("Avg Credit Score"), html.H4(f"{filtered_df['Credit Score'].mean():.0f}")]),
        html.Div([html.H6("Total Applications"), html.H4(f"{len(filtered_df):,}")]),
        html.Div([html.H6("Avg DTI Ratio"), html.H4(f"{filtered_df['Debt-to-Income Ratio'].mean():.2%}")])
    ]

    trend_data = (filtered_df.groupby('Years at Current Job')['Loan Amount']
              .mean()
              .reset_index()
              .dropna()
              .sort_values(by='Years at Current Job'))

# 
    trend_data['Years at Current Job'] = pd.to_numeric(trend_data['Years at Current Job'], errors='coerce')
    trend_fig = px.line(
        trend_data, 
        x='Years at Current Job', 
        y='Loan Amount', 
        title='Loan Amount Trend',
        markers=True  # Adds dots for clarity
    )
    
    trend_fig.update_layout(
        xaxis=dict(type='linear')  # Forces numerical axis
    )

    age_fig = px.histogram(filtered_df, x='Age', color='Gender', title='Age Distribution')
    income_fig = px.box(filtered_df, x='Risk Rating', y='Income', color='Gender', title='Income Distribution')
    loan_boxplot = px.box(filtered_df, x='Loan Purpose', y='Loan Amount', color='Risk Rating', title='Loan Amount vs Purpose')
    credit_scatter = px.scatter(filtered_df, x='Credit Score', y='Debt-to-Income Ratio', color='Risk Rating', size='Loan Amount', title='Credit Score vs DTI')
    risk_heatmap = px.density_heatmap(filtered_df, x='Credit Score', y='Income', z='Debt-to-Income Ratio', title='Risk Profile Heatmap')
    approval_fig = px.scatter_3d(filtered_df, x='Credit Score', y='Income', z='Debt-to-Income Ratio', color='Risk Rating', size='Loan Amount', title='Approval Prediction')

    return metrics, trend_fig, age_fig, income_fig, loan_boxplot, credit_scatter, risk_heatmap, approval_fig

# Run Dash App
if __name__ == '__main__':
    app.run_server(debug=True)
