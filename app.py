# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, confusion_matrix
)

# LIME - Model Interpretation
from lime.lime_tabular import LimeTabularExplainer

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AutoML & Explainability App",
    page_icon="🤖"
)

# --- App Title ---
# MODIFIED: Updated title to include CSV
st.title("Excel/CSV-driven Decision Support App (Charts + ML + LIME)")
st.markdown("Upload your dataset, visualize it, train a model, and understand its predictions.")

# --- Sidebar for Uploads and Settings ---
with st.sidebar:
    st.header("1. Upload & Settings")
    # MODIFIED: Allow .csv files and update the label
    uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"])

    st.markdown("---")
    st.header("2. Modeling Configuration")
    if uploaded_file:
        try:
            # MODIFIED: Check file type and use the correct pandas reader
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            target_col = st.selectbox("Select Target Column", ["-- choose --"] + df.columns.tolist())

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        target_col = None
        df = None
        st.info("Upload a file to configure the model.")

    if df is not None and target_col and target_col != "-- choose --":
        task_type_option = st.radio("Task Type", ["Auto-detect", "Classification", "Regression"], index=0)

        # Auto-detect logic
        if task_type_option == "Auto-detect":
            if df[target_col].dtype.kind in 'biufc' and df[target_col].nunique() < 20:
                task_type = "Classification"
            elif df[target_col].dtype.kind in 'biufc':
                task_type = "Regression"
            else:
                task_type = "Classification"
        else:
            task_type = task_type_option
        st.success(f"Task detected: **{task_type}**")

        # Model selection
        if task_type == "Classification":
            model_choice = st.selectbox("Choose a Classification Model", ["Logistic Regression", "Random Forest Classifier"])
        else:
            model_choice = st.selectbox("Choose a Regression Model", ["Linear Regression", "Random Forest Regressor"])
            
        # Feature selection - MOVED here for better UX
        st.markdown("---")
        st.header("3. Feature Selection")
        # Prepare feature list by excluding the target column
        features_df = df.drop(columns=[target_col], errors='ignore')
        all_features = features_df.columns.tolist()
        selected_features = st.multiselect("Select features to use", all_features, default=all_features)


        # Train-test split settings
        st.markdown("---")
        st.header("4. Training Settings")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5)
        random_state = st.number_input("Random Seed", value=42)

        run_button = st.button("🚀 Train Model & Get Insights", type="primary")
    else:
        run_button = False

# --- Main App Logic ---
if uploaded_file is None:
    st.info("Awaiting file upload to begin...")
    st.subheader("Sample Dataset Format")
    st.write("Your file should have rows as observations and columns as features. The target variable should be one of the columns.")
    st.write(pd.DataFrame({
        "age": [25, 32, 40, 28],
        "salary": [55000, 72000, 85000, 60000],
        "department": ["Sales", "Engineering", "HR", "Engineering"],
        "attrition_target": [0, 1, 0, 0]
    }))
    st.stop()

# --- Initialize Session State ---
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["📊 Data Preview & EDA", "🧠 Model Training & Evaluation", "🔍 Explain Predictions (LIME)"])

# --- Tab 1: Data Preview & EDA ---
with tab1:
    st.header("Data Preview")
    st.dataframe(df.head(200))

    st.header("Exploratory Data Analysis (EDA)")
    
    # EDA columns
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        with st.expander("Dataset Summary Statistics", expanded=True):
            st.write(df.describe(include='all').T)
        with st.expander("Missing Values Count", expanded=True):
            missing_vals = df.isna().sum()
            missing_df = missing_vals[missing_vals > 0].sort_values(ascending=False).to_frame('missing_count')
            if not missing_df.empty:
                st.dataframe(missing_df)
            else:
                st.success("No missing values found!")

    with eda_col2:
        st.subheader("Quick Visualizations")
        plot_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot", "Scatter Plot"])
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if plot_type == "Histogram":
            col = st.selectbox("Select numeric column for histogram", numeric_cols)
            if col and col in df.columns:
                fig = px.histogram(df, x=col, title=f"Histogram of {col}", nbins=30)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box Plot":
            col = st.selectbox("Select numeric column for box plot", numeric_cols)
            if col and col in df.columns:
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)

        else: # Scatter Plot
            if len(df.columns) >= 2:
                x_col = st.selectbox("Select X-axis column", df.columns, index=0)
                y_col = st.selectbox("Select Y-axis column", df.columns, index=1)
                color_col = st.selectbox("Select Color column (optional)", ["None"] + df.columns.tolist())
                
                if x_col and y_col:
                    fig = px.scatter(df, x=x_col, y=y_col, color=None if color_col == "None" else color_col,
                                     title=f"Scatter Plot: {y_col} vs. {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough columns for a scatter plot.")

# --- Preprocessing Pipeline Builder ---
def build_pipeline(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='drop')

    return preprocessor, num_cols, cat_cols

# --- Model Training Logic ---
if run_button:
    if not selected_features:
        st.error("Please select at least one feature in the sidebar to continue.")
        st.stop()
    else:
        with st.spinner("Preparing data and training model... This might take a moment."):
            # 1. Prepare Data using selected features
            X = df[selected_features].copy()
            y = df[target_col].copy()
            
            # Drop rows with missing target
            notna_mask = y.notna()
            X = X[notna_mask]
            y = y[notna_mask]

            # 2. Build Pipeline
            preprocessor, num_cols, cat_cols = build_pipeline(X)
            
            # 3. Choose Model
            if task_type == "Classification":
                model = RandomForestClassifier(random_state=random_state) if "Random Forest" in model_choice else LogisticRegression(max_iter=1000)
            else:
                model = RandomForestRegressor(random_state=random_state) if "Random Forest" in model_choice else LinearRegression()

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # 4. Split and Train
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100.0, random_state=random_state, stratify=(y if task_type=="Classification" and y.nunique() > 1 else None)
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Store results in session state
            st.session_state['model_trained'] = True
            st.session_state['pipeline'] = pipeline
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['task_type'] = task_type
            st.session_state['selected_features'] = selected_features
            st.session_state['num_cols'] = num_cols
            st.session_state['cat_cols'] = cat_cols
            st.session_state['target_col'] = target_col
        st.success("Model training complete! Check the results below.")


# --- Tab 2: Model Training & Evaluation ---
with tab2:
    if not st.session_state['model_trained']:
        st.info("Click 'Train Model' in the sidebar to see results here.")
    else:
        st.header("Model Performance on Test Set")
        
        # Retrieve from session state
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        pipeline = st.session_state['pipeline']
        X_test = st.session_state['X_test']
        task_type = st.session_state['task_type']

        # Layout for metrics and charts
        res_col1, res_col2 = st.columns((1, 2))

        with res_col1:
            st.subheader("Key Metrics")
            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("Precision (weighted)", f"{prec:.4f}")
                st.metric("Recall (weighted)", f"{rec:.4f}")
                st.metric("F1 Score (weighted)", f"{f1:.4f}")

                if hasattr(pipeline.named_steps['model'], "predict_proba") and y_test.nunique() == 2:
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                    st.metric("ROC AUC", f"{auc:.4f}")
            else: # Regression
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.metric("R-squared (R²)", f"{r2:.4f}")
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

        with res_col2:
            st.subheader("Performance Visuals")
            if task_type == "Classification":
                cm = confusion_matrix(y_test, y_pred)
                class_labels = sorted(y_test.unique())
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=[str(c) for c in class_labels], y=[str(c) for c in class_labels],
                                title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else: # Regression
                fig = px.scatter(x=y_test, y=y_pred, 
                                 labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                 title="Actual vs. Predicted Values")
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                         mode='lines', name='Ideal Fit (y=x)', line=dict(dash='dash', color='red')))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # Feature Importances (for tree-based models)
        model_in_pipeline = pipeline.named_steps['model']
        if isinstance(model_in_pipeline, (RandomForestClassifier, RandomForestRegressor)):
            st.subheader("Feature Importances")
            try:
                # Get feature names after one-hot encoding
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                importances = model_in_pipeline.feature_importances_
                
                imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                imp_df = imp_df.sort_values('importance', ascending=False).head(20)

                fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                             title="Top 20 Feature Importances")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate feature importance plot. Error: {e}")

        # Expander for advanced details
        with st.expander("Show Advanced Training Details"):
            st.write("**Model:**", pipeline.named_steps['model'])
            st.write("**Training Set Size:**", len(st.session_state['X_train']))
            st.write("**Test Set Size:**", len(st.session_state['X_test']))
            st.write("**Numeric Features Processed:**", st.session_state['num_cols'])
            st.write("**Categorical Features Processed:**", st.session_state['cat_cols'])

            # Download predictions
            if st.button("Download Test Set Predictions"):
                out_df = st.session_state['X_test'].copy().reset_index(drop=True)
                out_df["actual_target"] = st.session_state['y_test'].reset_index(drop=True)
                out_df["predicted_target"] = st.session_state['y_pred']
                
                towrite = BytesIO()
                out_df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                b64 = base64.b64encode(towrite.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predictions.xlsx">Download predictions.xlsx</a>'
                st.markdown(href, unsafe_allow_html=True)


# --- Tab 3: Explain Predictions (LIME) ---
with tab3:
    if not st.session_state['model_trained']:
        st.info("Train a model first to use the LIME explainer.")
    else:
        st.header("Local Interpretable Model-agnostic Explanations (LIME)")
        st.markdown("Select a single instance from the test set to understand why the model made a specific prediction for it.")

        # Retrieve necessary data from session state
        pipeline = st.session_state['pipeline']
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        task_type = st.session_state['task_type']
        selected_features = st.session_state['selected_features']

        try:
            # LIME needs a prediction function
            def predict_fn(x):
                # Lime creates numpy arrays, so we need to convert back to a DataFrame
                df_tmp = pd.DataFrame(x, columns=selected_features)
                # Ensure dtypes are correct for the pipeline
                for col in df_tmp.columns:
                    df_tmp[col] = df_tmp[col].astype(X_train[col].dtype)

                if task_type == "Classification":
                    return pipeline.predict_proba(df_tmp)
                else:
                    return pipeline.predict(df_tmp)

            explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=selected_features,
                class_names=[str(c) for c in np.unique(y_train)] if task_type == "Classification" else ['prediction'],
                mode=task_type.lower(),
                discretize_continuous=True,
                random_state=42
            )

            # Choose an instance to explain
            instance_idx = st.selectbox(
                "Choose an instance from the test set to explain:",
                X_test.index
            )
            
            if st.button("Explain Instance", type="primary"):
                instance_data = X_test.loc[instance_idx].values
                st.write("Explaining the following instance:")
                st.dataframe(X_test.loc[[instance_idx]])
                
                with st.spinner("Generating LIME explanation..."):
                    explanation = explainer.explain_instance(
                        instance_data,
                        predict_fn,
                        num_features=min(10, len(selected_features))
                    )
                    
                    st.subheader("LIME Explanation Plot")
                    fig = explanation.as_pyplot_figure()
                    st.pyplot(fig)

                    st.subheader("Explanation Details")
                    st.write("The table below shows how each feature contributed to the prediction. Features in green support the prediction, while those in red contradict it.")
                    st.write(explanation.as_list())

        except Exception as e:
            st.error(f"Failed to generate LIME explanation. Error: {e}")
            st.error("This can sometimes happen if the data has low variance or issues with categorical encoding. Please check your dataset.")