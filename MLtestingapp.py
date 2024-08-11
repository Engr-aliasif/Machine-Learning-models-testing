import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px
import numpy as np

# Set title
st.title("Machine Learning Models Testing")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Display file info
if uploaded_file is not None:
    st.sidebar.write("Uploaded file:", uploaded_file.name)
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    # Display table headers and columns
    st.write("Table Headers and Columns:")
    st.write(df.columns.tolist())

    # Label encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # User selects columns
    selected_cols = st.sidebar.multiselect("Select two columns", df.columns.tolist())

    # Choose plot type
    plot_type = st.sidebar.selectbox("Select plot type", ["Bar", "Scatter", "Line"])

    # Regression models
    models = ["None", "Linear Regression", "Decision Tree Regression", "K-Nearest Neighbors Regression"]
    model_selection = st.sidebar.selectbox("Select a model", models)

    if selected_cols and plot_type and model_selection != "None":
        st.write(f"Plotting {plot_type} graph for columns: {selected_cols}")

        # Train model based on selection
        X = df[selected_cols[0]].values.reshape(-1, 1)
        y = df[selected_cols[1]].values

        if model_selection == "Linear Regression":
            model = LinearRegression()
        elif model_selection == "Decision Tree Regression":
            model = DecisionTreeRegressor()
        elif model_selection == "K-Nearest Neighbors Regression":
            model = KNeighborsRegressor()

        model.fit(X, y)

        # Generate predictions
        X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_pred)

        # Create a scatter plot with the model's regression line
        fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1],
                         title=f"Scatter Plot with {model_selection}: {selected_cols[0]} vs {selected_cols[1]}",
                         labels={selected_cols[0]: selected_cols[0], selected_cols[1]: selected_cols[1]})
        fig.add_scatter(x=X_pred.flatten(), y=y_pred, mode='lines', name=model_selection)

    else:
        # Non-regression plots
        if plot_type == "Bar":
            fig = px.bar(df, x=selected_cols[0], y=selected_cols[1],
                         title=f"Bar Plot: {selected_cols[0]} vs {selected_cols[1]}",
                         labels={selected_cols[0]: "Category", selected_cols[1]: "Value"})
            fig.update_xaxes(type='category')

        elif plot_type == "Scatter":
            fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1],
                             title=f"Scatter Plot: {selected_cols[0]} vs {selected_cols[1]}",
                             labels={selected_cols[0]: selected_cols[0], selected_cols[1]: selected_cols[1]})

        else:
            fig = px.line(df, x=selected_cols[0], y=selected_cols[1],
                          title=f"Line Plot: {selected_cols[0]} vs {selected_cols[1]}",
                          labels={selected_cols[0]: selected_cols[0], selected_cols[1]: selected_cols[1]})

    if selected_cols:
        st.plotly_chart(fig)

    else:
        st.sidebar.write("Please select two columns, plot type, and a model.")

else:
    st.sidebar.write("Please upload a file.")
