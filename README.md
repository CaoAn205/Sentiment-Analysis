# Sentiment-Analysis: Phân loại dữ liệu và thao tác trên Telegram Bot
### 1. Mục tiêu:
###### - Đây là tài liệu hướng dẫn cài đặt cho mô hình Telegram Bot với đầu vào là văn bản dưới dạng tin nhắn trên Telegram và Bot trả về nhãn phân loại cảm xúc: Postive, Negative hoặc Neutral
### 2. Công nghệ sử dụng: 
###### - Sử dụng các modules có sẵn của Python để thu thập dữ liệu và thao tác trên dataframe như: bs4, pandas,…
###### -	Áp dụng thuật toán Naïve Bayes của Scikit-learn để phân tích ngôn ngữ và từ đó phân loại dữ liệu
###### -	Sử dụng Telegram Bot để đưa ra giao diện thân thiện hơn với người dùng
### 3. Hướng dẫn cài đặt: 
##### -	Download source code và dùng virtual environment (venv) để install các packages thông qua lệnh pip install (ví dụ: pip -m install pandas)
##### -	Download telegram for desktop để tiện thực hiện: https://desktop.telegram.org/
##### - Download ngrok server: https://ngrok.com/download
### 4. Hướng dẫn sử dụng: 
##### -	Ta mở file và chạy ngrok.exe với lệnh: ngrok http 5000
##### -	Sau đó copy link server: https://... dán vào trong phần NGROK_URL ở phần code config.py
##### - Ngoài ra cần xem lại path của nơi lưu file dataset.csv
##### - Sau khi run app.py, ta có thể gửi các văn bản muốn phân loại cảm xúc thông qua tin nhắn

