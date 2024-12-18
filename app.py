from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
import plotly.graph_objs as go

app = Flask(__name__)
model = joblib.load('traffic_model_fixed.pkl')

# Initialize Dash and embed it into Flask
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Define the layout for Dash
dash_app.layout = html.Div([
    html.H1("Traffic Prediction Dashboard"),
    html.Div([
        dcc.Graph(id="pie_chart"),
        dcc.Graph(id="box_plot")
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-around'}),
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
])

# Callback for updating charts in Dash
@dash_app.callback(
    [Output("pie_chart", "figure"),
     Output("box_plot", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_charts(n):
    # Sample data for demonstration
    hours = np.arange(0, 24, 1)
    traffic_conditions = [model.predict(np.array([[hour, 1, 2]]))[0] for hour in hours]
    condition_counts = {'low': 5, 'normal': 10, 'high': 6, 'heavy': 3}

    # Create a pie chart
    pie_fig = go.Figure(data=[go.Pie(labels=list(condition_counts.keys()), values=list(condition_counts.values()))])
    pie_fig.update_layout(title="Traffic Condition Distribution")

    # Create a box plot
    box_fig = go.Figure(data=[go.Box(y=traffic_conditions, boxpoints='all', jitter=0.3, pointpos=-1.8)])
    box_fig.update_layout(title="Traffic Conditions Spread by Hour", yaxis_title="Traffic Level")

    return pie_fig, box_fig

def convert_time_to_minutes(time_str):
    time_obj = datetime.strptime(time_str, '%I:%M:%S %p')
    return time_obj.hour * 60 + time_obj.minute

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    time_str = request.form['timeFormatted']  # Nhận thời gian đã chuyển đổi
    day = int(request.form['day'])  # Ngày trong tháng
    car_count = int(request.form['car_count'])
    bus_count = int(request.form['bus_count'])
    bike_count = int(request.form['bike_count'])
    truck_count = int(request.form['truck_count'])
    day_of_week = request.form['day_of_week']  # Thứ trong tuần, ví dụ: 'Monday'

    # Mã hóa 'Day of the week' thành số nếu mô hình yêu cầu
    days_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_of_week_encoded = days_mapping[day_of_week]

    # Chuyển thời gian thành số phút kể từ nửa đêm (nếu mô hình yêu cầu)
    time_in_minutes = convert_time_to_minutes(time_str)  # Hàm này đã có từ trước

    # Tính tổng số xe (nếu mô hình yêu cầu)
    total_traffic = car_count + bus_count + bike_count + truck_count

    missing_feature_1 = 0  
    missing_feature_2 = 0  
    missing_feature_3 = 0  
    missing_feature_4 = 0
    missing_feature_5 = 0

    # Tạo dữ liệu đầu vào cho mô hình
    input_data = np.array([[time_in_minutes, day, car_count, bike_count, bus_count, truck_count, total_traffic, day_of_week_encoded,
                            missing_feature_1,missing_feature_2,missing_feature_3,missing_feature_4,missing_feature_5]])

    # Dự đoán tình trạng giao thông
    prediction = model.predict(input_data)[0]

    return jsonify({'traffic_condition': prediction})


if __name__ == '__main__':
    app.run(debug=True)
