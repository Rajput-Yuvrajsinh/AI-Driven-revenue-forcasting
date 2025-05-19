import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from utils import send_telegram_message
from streamlit import config

CSV_FILE = "users.csv"

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["email", "password", "role"])
    df.to_csv(CSV_FILE, index=False)

users_df = pd.read_csv(CSV_FILE)

users_df["email"]= users_df["email"].astype(str)
users_df["password"] = users_df["password"].astype(str)

df = pd.read_csv("Walmart.csv")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "role" not in st.session_state:
    st.session_state.role = None

query_params = st.query_params

if "auth" in query_params and query_params["auth"] == "true":
    st.session_state.authenticated = True
    st.session_state.user_id = query_params.get("user", "")
    st.session_state.role = query_params.get("role", "")

if not st.session_state.authenticated:
    option = st.sidebar.selectbox("Login or Signup", ["Login", "Signup"])

    if option == "Login":
        st.header("Login")
        with st.form(key="login_form"):
            email = st.text_input("Username").strip().lower()
            password = st.text_input("Password", type="password").strip()
            btn = st.form_submit_button(label="Login")

            if btn:
                user_data = users_df[
                    (users_df["email"].str.strip().str.lower() == email) &
                    (users_df["password"].str.strip() == password)]

                if not user_data.empty:
                    st.session_state.authenticated = True
                    st.session_state.user_id = email
                    st.session_state.role = user_data.iloc[0]["role"]

                    st.query_params["auth"] = "true"
                    st.query_params["user"] = email
                    st.query_params["role"] = user_data.iloc[0]["role"]

                    st.success(f"Login successful! Welcome, {st.session_state.role.capitalize()}.")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    elif option == "Signup":
        st.header("Signup")
        with st.form(key="signup_form"):
            email1 = st.text_input("Choose Username").strip().lower()
            password1 = st.text_input("Choose Password", type="password").strip()
            select_role = st.selectbox("Select Role", ["admin", "analyst", "manager"])
            btn = st.form_submit_button(label="Signup")

            if btn:
                if (users_df["email"].str.strip().str.lower() == email1).any():
                    st.error("User already exists!")
                else:
                    new_user = pd.DataFrame([[email1, password1, select_role]], columns=["email", "password", "role"])
                    users_df = pd.concat([users_df, new_user], ignore_index=True)
                    users_df.to_csv(CSV_FILE, index=False)
                    st.success("Sign up successful! Please log in.")

if st.session_state.authenticated:
    st.sidebar.title("📈 AI-Driven Revenue Forcasting")
    st.sidebar.write(f"Welcome: {st.session_state.user_id}")
    st.sidebar.write(f"Role: {st.session_state.role.capitalize()}")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.role = None
        st.query_params.clear()
        st.rerun()

    st.sidebar.markdown("---")

    role = st.session_state.role
    with st.sidebar:
        if role =="admin":
            page = option_menu("Admin",["Project Overview", "Dashboard","Dataset","Prediction","Visualization","Admin Panel"])
        elif role =="manager":
            page = option_menu("Manager",["Project Overview", "Dashboard","Visualization"])
        elif role =="analyst":
            page= option_menu("Analyst",["Dataset","Prediction","Visualization","Growth Analysis"])

    st.title(f"{page} Dashboard")

    if page == "Dashboard":
        if role not in ["manager","admin"]:
            st.warning("Access Denied: Manager or Admins only")
        else:
            st.header("📊Manager Dashboard")
            df["date"] = pd.to_datetime(df["date"])
            st.subheader("📅 filter by data range")

            col1,col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date",value=df["date"].min().date())
            with col2:
                end_date = st.date_input("End Date",value=df["date"].max().date())

            filtered_df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

            st.subheader("📈 Revenue Summary")
            total_profit = filtered_df["profit_margin"].sum()
            avg_profit = filtered_df["profit_margin"].mean()
            st.metric("Total profit margin", f"{total_profit:.2f}")
            st.metric("Average profit margin", f"{avg_profit:.2f}")
            st.markdown(" you can use the date range above to explore revenue pattern over time. perfect fot reports!")

            st.markdown("""
                    - View revenue trends  
                    - Detect anomalies  
                    - Export reports  
                    """)

        if st.button("Reset Dates"):
            st.rerun()

    if page == "Project Overview":
        st.header("📋Project Overview")
        st.write("Details about the revenue forecasting system.")
        st.markdown("""
✅ Project Title:
AI-Driven Revenue Forecasting and Trend Analysis for Business Growth

👨‍💻 Developer:
Rajput Yuvrajsinh Shersinh

🎯 Objective:
To design and develop a smart, interactive system for revenue forecasting using Machine Learning (Random Forest) and Deep Learning (LSTM) models, providing business stakeholders with real-time insights into profit trends, growth patterns, and future projections to aid data-driven decision-making.

🛠️ Key Features:
1. User Authentication & Role-Based Access:
Users can sign up and log in with roles like Admin, Manager, and Analyst.

Role-based navigation and access control:

Admin: Full access to all modules including dataset, visualization, prediction, dashboard, and admin panel.

Manager: View-only access to project overview, dashboard, and visualization.

Analyst: Access to data exploration, profit prediction, and growth analysis.

2. Dataset Used:
Walmart.csv dataset with fields like date, unit_price, quantity, city, category, payment_method, and profit_margin.

Data is cleaned, encoded, and processed for modeling.

3. ML-Based Profit Margin Prediction:
Users input values like unit price, quantity, branch, category, etc.

Random Forest Regressor predicts the profit margin.

Forecasted revenue shown alongside actual trends.

4. Interactive Forecast Visualization:
Uses Plotly to generate:

Actual vs Predicted profit plots.

Forecasted profits for next N days based on user input.

Highlights revenue trends and growth opportunities.

5. Quarterly Forecasting with LSTM:
Uses LSTM Neural Network to forecast profit for the next 90 days (one quarter).

Trained on historical daily profits.

Predicts future values using a sliding window mechanism and visualizes them interactively.

6. Dashboard & Metrics (Role: Manager/Admin):
Filterable date range.

View Total and Average Profit Margins over time.

Designed for reporting and revenue review.

📊 Technologies & Libraries:
Frontend: Streamlit

Data Handling: Pandas, Numpy

Machine Learning: Scikit-learn (Random Forest, LabelEncoder)

Deep Learning: Keras (LSTM)

Visualization: Plotly

Authentication: Custom CSV-based login system with persistent sessions

📈 Benefits:
Helps businesses forecast revenue based on past trends.

Enables role-specific insights, making collaboration smoother.

Uses both traditional ML and LSTM DL models for comprehensive prediction.

Highly interactive dashboard suitable for management, analysis, and reporting.

📁 Future Enhancements (Optional ideas you could add):
Email authentication or OTP-based login.

Saving forecast reports as PDF or Excel.

Monthly revenue breakdown with comparison charts.

Integration with live sales data from a database or API.
        """)

    if page == "Dataset":
        if role not in ["analyst","admin"]:
            st.warning("Access Denied: analysis or admins only")
        else:
            st.header("🗃️ Dataset")
            df = pd.read_csv("Walmart.csv")
            st.dataframe(df)

    if page == "Prediction":
        if role != "analyst":
            st.warning("Access Denied: analysts only.")
        else:
            # Load and preprocess data
            st.header("Profit Margin Prediction")
            df_raw = pd.read_csv("Walmart.csv")

            df = df_raw.copy()
            df["unit_price"] = df["unit_price"].replace('[\$,]', '', regex=True).astype(float)
            df = df.drop(columns=["invoice_id", "date", "time", "rating"])

            cat_cols = ["Branch", "City", "category", "payment_method"]
            df_encoded = df.copy()
            le_dict = {}

            for col in cat_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                le_dict[col] = le

            x = df_encoded.drop("profit_margin", axis=1)
            y = df_encoded["profit_margin"]
            feature_order = x.columns.tolist()

            # Train model
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            rf_classifier = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_classifier.fit(x_train, y_train)

            # Sample prediction view
            with st.expander("📊 Sample Prediction"):
                st.write(rf_classifier.predict(x_test)[:10])

            # Prediction form
            with st.form(key='Prediction'):
                st.subheader("Enter User Details")
                unit_price = st.number_input("Enter unit price")
                quantity = st.number_input("Enter quantity")
                branch = st.selectbox("Select Branch", df["Branch"].unique())
                city = st.selectbox("Select City", df["City"].unique())
                category = st.selectbox("Select Category", df["category"].unique())
                payment_method = st.selectbox("Select Payment Method", df["payment_method"].unique())
                n_days = st.slider("Predict for next N days", min_value=1, max_value=100, value=30)

                if st.form_submit_button("Predict"):
                    # Single user input
                    input_df = pd.DataFrame([{
                        "unit_price": unit_price,
                        "quantity": quantity,
                        "Branch": branch,
                        "City": city,
                        "category": category,
                        "payment_method": payment_method
                    }])

                    for col in cat_cols:
                        le = le_dict[col]
                        input_df[col] = le.transform(input_df[col].astype(str))
                    input_df = input_df[feature_order]

                    prediction = rf_classifier.predict(input_df)
                    st.success(f"Predicted Profit Margin: {prediction[0]:.2f}")

                    # ---------- Visualization with Forecast ----------
                    st.header("📈 Actual vs Predicted Profit with Future Forecast")

                    # Prepare full encoded data again for visualization
                    df_vis = df.copy()
                    df_vis["date"] = pd.to_datetime(df_raw["date"])
                    df_encoded_vis = df_vis.copy()
                    for col in cat_cols:
                        df_encoded_vis[col] = le_dict[col].transform(df_encoded_vis[col].astype(str))

                    x_vis = df_encoded_vis[feature_order]
                    df_vis["predicted_margin"] = rf_classifier.predict(x_vis)

                    # Calculate actual and predicted profit
                    df_vis["actual_profit"] = df_vis["unit_price"] * df_vis["quantity"] * (
                                df_vis["profit_margin"] / 100)
                    df_vis["predicted_profit"] = df_vis["unit_price"] * df_vis["quantity"] * (
                                df_vis["predicted_margin"] / 100)

                    # Group by date
                    profit_over_time = df_vis.groupby("date")[["actual_profit", "predicted_profit"]].sum().reset_index()

                    # Prepare future forecast
                    last_date = profit_over_time["date"].max()
                    future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]

                    # Generate synthetic inputs with slight variation
                    template_input = {
                        "unit_price": df["unit_price"].median(),
                        "quantity": df["quantity"].median()
                    }
                    for col in cat_cols:
                        template_input[col] = df_encoded[col].mode()[0]

                    future_data = pd.DataFrame([template_input] * n_days)

                    # Add trend/noise to avoid flat line
                    future_data["unit_price"] = future_data["unit_price"] * (1 + 0.01 * np.arange(n_days))
                    future_data["quantity"] = future_data["quantity"] * (1 + 0.005 * np.random.randn(n_days))
                    future_data = future_data[feature_order]  # Ensure column order matches training

                    # Predict margins and profit
                    future_margins = rf_classifier.predict(future_data)
                    future_profits = future_data["unit_price"] * future_data["quantity"] * (future_margins / 100)

                    # Create future DataFrame
                    future_df = pd.DataFrame({
                        "date": future_dates,
                        "actual_profit": [None] * n_days,
                        "predicted_profit": future_profits
                    })

                    # Plot actual historical + future predicted only
                    fig = go.Figure()

                    # Actual historical data
                    fig.add_trace(go.Scatter(
                        x=profit_over_time["date"],
                        y=profit_over_time["actual_profit"],
                        mode='lines+markers',
                        name='Actual Profit',
                        line=dict(color='green')
                    ))

                    # Future predictions only
                    fig.add_trace(go.Scatter(
                        x=future_df["date"],
                        y=future_df["predicted_profit"],
                        mode='lines+markers',
                        name=f'Forecasted Profit (Next {n_days} Days)',
                        line=dict(color='orange', dash='dot')
                    ))

                    fig.update_layout(
                        title=f"Actual vs Forecasted Profit (Next {n_days} Days)",
                        xaxis_title="Date",
                        yaxis_title="Profit",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    bot_token = "7486726237:AAGeJ_TpD_JGS0JDQJ3M0mGHnjMyjsQmaPw"
                    chat_id = "1243357040"

                    # telegram_cfg = st.config["telegram"]
                    # bot_token = telegram_cfg["bot_token"]
                    # chat_id = telegram_cfg["chat_id"]

                    profit = prediction[0]*100
                    st.write(profit)

                    threshold = 40

                    if profit < threshold:
                        alert_msg = f"⚠ Alert: Profit has dropped to ₹{profit}, below the threshold of ₹{threshold}!"
                        send_telegram_message(alert_msg, bot_token, chat_id)
                        st.error("🚨 Telegram alert sent due to low profit!")
                    else:
                        st.success(f"✅ Profit is ₹{profit} — all good!")

    if page == "Growth Analysis":
        st.subheader("📊 Next Quarter Profit Forecast")

        # Load data
        df_raw = pd.read_csv("Walmart.csv")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw["unit_price"] = df_raw["unit_price"].replace('[\$,]', '', regex=True).astype(float)

        # Actual profit
        df_raw["profit"] = df_raw["unit_price"] * df_raw["quantity"] * (df_raw["profit_margin"] / 100)

        # Year selector
        df_raw["year"] = df_raw["date"].dt.year
        years = df_raw["year"].unique().tolist()
        selected_years = st.multiselect("Select Years to Compare", sorted(years, reverse=True), default=[max(years)])

        # Filter
        df_selected = df_raw[df_raw["year"].isin(selected_years)]
        df_selected = df_selected.groupby("date")["profit"].sum().reset_index()

        # LSTM-ready dataset
        data = df_selected.set_index("date").resample("D").sum().fillna(0)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)


        # Prepare sequences
        def create_sequences(data, window_size=30):
            x, y = [], []
            for i in range(window_size, len(data)):
                x.append(data[i - window_size:i])
                y.append(data[i])
            return np.array(x), np.array(y)


        window_size = 30
        X, y = create_sequences(scaled_data, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        # Predict next quarter (90 days)
        last_sequence = scaled_data[-window_size:]
        future_preds = []

        for _ in range(90):
            input_seq = last_sequence[-window_size:].reshape(1, window_size, 1)
            pred = model.predict(input_seq, verbose=0)[0][0]
            future_preds.append(pred)
            last_sequence = np.append(last_sequence, [[pred]], axis=0)

        # Reverse scaling
        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

        # Future dates
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i + 1) for i in range(90)]

        # Combine for graph
        df_future = pd.DataFrame({"date": future_dates, "predicted_profit": future_preds})
        df_plot = pd.concat([df_selected, df_future], ignore_index=True)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_selected["date"], y=df_selected["profit"], name="Actual Profit", mode="lines",
                                 line=dict(color="blue")))
        fig.add_trace(
            go.Scatter(x=df_future["date"], y=df_future["predicted_profit"], name="Next Quarter Forecast", mode="lines",
                       line=dict(color="red", dash="dot")))
        fig.update_layout(title="Profit Trends and Next Quarter Forecast", xaxis_title="Date", yaxis_title="Profit",
                          hovermode="x unified")

        st.plotly_chart(fig, use_container_width=True)

    if page == "Visualization":
        st.header("📈 Data Visualization")

        fig = px.bar(df,x="City",y="profit_margin",title="Margin")
        st.plotly_chart(fig)

        fig = px.pie(df,names="category",values="profit_margin",title="Profit margin by category")
        st.plotly_chart(fig)

# ------- monthly profit margin----------------
        df["date"] = pd.to_datetime(df["date"])
        df["Year"] = df["date"].dt.year
        df["Month"] = df["date"].dt.to_period("M")
        monthly_trend = df.groupby("Month")["profit_margin"].sum().reset_index()
        monthly_trend["Month"] = monthly_trend["Month"].astype(str)
        fig = px.bar(monthly_trend, x="Month", y="profit_margin", title="Monthly Revenue Trend")
        st.plotly_chart(fig)

# ----average monthly profit margin-------------
        df["date"] = pd.to_datetime(df["date"])
        df["Month"] = df["date"].dt.to_period("M").astype(str)
        monthly_profit = df.groupby("Month")["profit_margin"].mean().reset_index()
        fig = px.line(monthly_profit,x="Month",y="profit_margin",title="📈 Average Monthly Profit Margin",markers=True)
        fig.update_layout(xaxis_title="Month",yaxis_title="Average Profit Margin",xaxis_tickangle=-45,plot_bgcolor="#f9f9f9")
        st.plotly_chart(fig)

# ------average month and branch

    if page == "Admin Panel":
        if role != "admin":
            st.warning("Access Denied: Admins only.")
        else:
            st.header("User Management")
            st.dataframe(users_df)
