#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import dask.dataframe as dd
import dask.array as da
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')
# get_ipython().run_line_magic('matplotlib', 'inline')
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output


# In[147]:


df_initial = pd.read_csv('financial_risk_assessment.csv')


# In[148]:


# MICE 
def mice_imputation(df_initial):
    mice_imputer = IterativeImputer(
        random_state=42,
        max_iter=10,
        n_nearest_features=5
    )
    
    imputed_data = mice_imputer.fit_transform(df_initial)
    
    imputed_df = pd.DataFrame(imputed_data, columns=df_initial.columns)
    
    return imputed_df

# KNN Imputation
def knn_imputation(df_initial):
    knn_imputer = KNNImputer(
        n_neighbors=5,
        weights='uniform'
    )
    
    imputed_data = knn_imputer.fit_transform(df_initial)
    
    imputed_df = pd.DataFrame(imputed_data, columns=df_initial.columns)
    
    return imputed_df

# Main imputation function
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

# Compare results
def compare_imputation_methods(df_initial):
    mice_results = impute_data(df_initial, method='mice')
    knn_results = impute_data(df_initial, method='knn')
    return mice_results, knn_results

''' print("\nOriginal Data Statistics:")
    print(df_initial.describe())
    
    print("\nMICE Imputed Data Statistics:")
    print(mice_results.describe())
    
    print("\nKNN Imputed Data Statistics:")
    print(knn_results.describe())
'''
    
# Main execution
df_initial_numeric = df_initial.select_dtypes(include=[np.number])

mice_imputed, knn_imputed = compare_imputation_methods(df_initial_numeric)

mice_imputed.to_csv('mice_imputed.csv', index=False)
knn_imputed.to_csv('knn_imputed.csv', index=False)



imputed_df_mice = impute_data(df_initial_numeric, method='mice')
imputed_df_knn = impute_data(df_initial_numeric, method='knn')

df_filled = df_initial.copy()

for column in df_initial.columns:
    if column in imputed_df_mice.columns:
        nan_mask = df_initial[column].isna()

        df_filled.loc[nan_mask, column] = imputed_df_mice.loc[nan_mask, column]


# In[149]:


#df_initial.head(10)


# In[150]:


df = df_filled


# In[154]:


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Comprehensive Loan Analysis Dashboard", 
            style={'textAlign': 'center', 'color': '#1E90FF', 'marginBottom': '30px'}),
    
    # Filters Section
    html.Div([
        html.Div([
            html.Label("Select Gender:"),
            dcc.Dropdown(
                id='gender-filter',
                options=[{'label': gender, 'value': gender} for gender in df['Gender'].unique()],
                multi=True,
                value=df['Gender'].unique()
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Select Risk Rating:"),
            dcc.Dropdown(
                id='risk-filter',
                options=[{'label': rating, 'value': rating} for rating in df['Risk Rating'].unique()],
                multi=True,
                value=df['Risk Rating'].unique()
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Select Loan Purpose:"),
            dcc.Dropdown(
                id='purpose-filter',
                options=[{'label': purpose, 'value': purpose} for purpose in df['Loan Purpose'].unique()],
                multi=True,
                value=df['Loan Purpose'].unique()
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'backgroundColor': '#F0F8FF', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Key Metrics Cards
    html.Div([
        html.Div([
            html.H4("Key Metrics", style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div(id='key-metrics', style={'display': 'flex', 'justifyContent': 'space-around'})
        ], style={'backgroundColor': '#F0F8FF', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'})
    ]),
    
    # Time Series Analysis
    html.Div([
        html.H4("Trend Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='trend-analysis')
    ], style={'backgroundColor': '#F0F8FF', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Main Visualizations
    html.Div([
        # First Row of Graphs
        html.Div([
            dcc.Graph(id='age-distribution'),
            dcc.Graph(id='income-distribution')
        ], style={'display': 'flex'}),
        
        # Second Row of Graphs
        html.Div([
            dcc.Graph(id='loan-amount-boxplot'),
            dcc.Graph(id='credit-score-scatter')
        ], style={'display': 'flex'}),
        
        # Third Row of Graphs (New)
        html.Div([
            dcc.Graph(id='risk-heatmap'),
            dcc.Graph(id='approval-prediction')
        ], style={'display': 'flex'})
    ])
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
    # Filter DataFrame
    filtered_df = df[
        df['Gender'].isin(selected_genders) & 
        df['Risk Rating'].isin(selected_risks) &
        df['Loan Purpose'].isin(selected_purposes)
    ]
    
    # Key Metrics Cards
    metrics = [
        html.Div([
            html.H6("Average Loan Amount"),
            html.H4(f"${filtered_df['Loan Amount'].mean():,.2f}")
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px'}),
        html.Div([
            html.H6("Average Credit Score"),
            html.H4(f"{filtered_df['Credit Score'].mean():.0f}")
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px'}),
        html.Div([
            html.H6("Total Applications"),
            html.H4(f"{len(filtered_df):,}")
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px'}),
        html.Div([
            html.H6("Average DTI Ratio"),
            html.H4(f"{filtered_df['Debt-to-Income Ratio'].mean():.2%}")
        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px'})
    ]
    
    # Trend Analysis
    trend_fig = px.line(
        filtered_df.groupby('Years at Current Job')['Loan Amount'].mean().reset_index(),
        x='Years at Current Job',
        y='Loan Amount',
        title='Average Loan Amount by Years at Job'
    )
    
    # Age Distribution
    age_fig = px.histogram(
        filtered_df, 
        x='Age', 
        color='Gender', 
        marginal='box', 
        title='Age Distribution by Gender'
    )
    
    # Income Distribution
    income_fig = px.box(
        filtered_df, 
        x='Risk Rating', 
        y='Income', 
        color='Gender',
        title='Income Distribution by Risk Rating and Gender'
    )
    
    # Loan Amount Boxplot
    loan_boxplot = px.box(
        filtered_df, 
        x='Loan Purpose', 
        y='Loan Amount', 
        color='Risk Rating',
        title='Loan Amount by Purpose and Risk Rating'
    )
    
    # Credit Score Scatter
    credit_scatter = px.scatter(
        filtered_df, 
        x='Credit Score', 
        y='Debt-to-Income Ratio', 
        color='Risk Rating',
        size='Loan Amount',
        hover_data=['Age', 'Gender', 'Income'],
        title='Credit Score vs Debt-to-Income Ratio'
    )
    
    # Risk Heatmap
    risk_heatmap = px.density_heatmap(
        filtered_df,
        x='Credit Score',
        y='Income',
        z='Debt-to-Income Ratio',
        title='Risk Profile Heatmap',
        labels={'color': 'DTI Ratio'}
    )
    

    # Approval Prediction Analysis
    approval_fig = px.scatter_3d(
        filtered_df,
        x='Credit Score',
        y='Income',
        z='Debt-to-Income Ratio',
        color='Risk Rating',
        size='Loan Amount',
        title='3D Risk Analysis'
    )
    
    return metrics, trend_fig, age_fig, income_fig, loan_boxplot, credit_scatter, risk_heatmap, approval_fig


if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




