# BOT TRADER SWING MULTI V6.2.0

Ứng dụng nghiên cứu và mô phỏng Swing Trading cổ phiếu Việt Nam, khung tham chiếu 3–10 phiên. V6.2.0 nâng cấp kiến trúc phân tích dùng chung: một nguồn dữ liệu chuẩn, một lớp tính toán trung tâm và một snapshot phân tích được ba tab sử dụng.

> **Phạm vi:** nhập dữ liệu thủ công; không tự động lấy dữ liệu, không đặt lệnh. Điểm số và nhận diện cấu trúc là heuristic, không phải xác suất thắng hay cam kết lợi nhuận. Đây là công cụ nghiên cứu/mô phỏng, không phải khuyến nghị mua bán cá nhân.

## 1. Chạy ứng dụng

Yêu cầu Python 3.10 trở lên.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 2. Ba tab – một snapshot phân tích

### 1 – TỔNG QUAN
- Trạng thái VN-Index, điểm sàng lọc Swing, pha cấu trúc cổ phiếu.
- Sức mạnh tương đối so với VN-Index và ngành khi có dữ liệu đồng bộ.
- Luận cứ ủng hộ/phản bác, biểu đồ và chất lượng dữ liệu.
- Tóm tắt tích lũy/hấp thụ/phân phối.

### 2 – DẤU CHÂN DÒNG TIỀN
- Proxy cung cầu CLV-volume, OBV proxy và thanh khoản.
- Phân tích tích lũy/hấp thụ/phân phối đa phiên.
- Theo dõi vòng đời nghi vấn Bull Trap, Bear Trap và breakout: chờ xác nhận, xác nhận, vô hiệu hoặc mâu thuẫn.
- OHLCV không xác định danh tính tổ chức, khối ngoại, tự doanh hay bên chủ động mua/bán.

### 3 – CHIẾN LƯỢC
- Kịch bản, vùng theo dõi, trigger, mức vô hiệu/SL, TP1/TP2, R:R.
- Quản trị vị thế mô phỏng, quy mô tham chiếu và chi phí giao dịch.
- Kiểm định holdout ngoài mẫu: 70% lịch sử đầu là giai đoạn khởi tạo/quan sát, 30% cuối là giai đoạn kiểm định. Tín hiệu được tính tuần tự tại từng thời điểm; benchmark chỉ lấy dữ liệu đến ngày tín hiệu. Không tối ưu tham số trên tập ngoài mẫu.

## 3. Quy tắc liên kết dữ liệu và tính toán

- `modules/indicators.py::add_indicators`: nguồn công thức chỉ báo kỹ thuật.
- `modules/indicators.py::add_ohlcv_features`: nguồn công thức CLV, signed-volume proxy, OBV proxy, value proxy, volume z-score và Amihud proxy.
- `REL_VOLUME20` dùng khối lượng phiên hiện tại chia cho trung bình 20 phiên đã hoàn tất trước đó; market behavior, flow/trap và Swing cùng đọc trường này.
- `modules/unified_analytics.py`: tạo snapshot trung tâm, phân tích cấu trúc và gọi một đường suy luận quyết định dùng chung.
- Các tab chỉ đọc kết quả từ snapshot; không tự tính lại chỉ báo hay kết luận.
- Backtest gọi cùng đường suy luận quyết định ở từng mốc lịch sử, với dữ liệu bị giới hạn tại thời điểm đó.
- Snapshot ID và data fingerprint được hiển thị ở tab Tổng quan để truy xuất phiên phân tích.
- Mỗi chỉ tiêu có một công thức chuẩn trong module nguồn; nếu thay đổi công thức phải cập nhật kiểm thử hồi quy và phiên bản engine.

## 4. Dữ liệu đầu vào

### OHLCV – có thể tải nhiều file cùng lúc

Mỗi file cần có các cột:

`date, symbol, open, high, low, close, volume`

- Có thể nhập nhiều mã qua nhiều file; mỗi file trong lô nhiều file phải có cột `symbol`.
- Có thể nhập VN-Index bằng mã `VNINDEX`.
- Tối thiểu 50 phiên cho bộ máy Swing; khuyến nghị 100–200 phiên trở lên.
- Nếu nhiều file có cùng cặp mã/ngày, app từ chối lô nhập để tránh ghi đè mơ hồ.
- Giá cổ phiếu phải chọn đúng đơn vị; VNINDEX không bị quy đổi.

### Dữ liệu ngành – tùy chọn

CSV cần có ít nhất:

`date, sector, close`

Có thể thêm `volume`. Sau khi tải file, chọn ngành tương ứng với mã đang phân tích. Dữ liệu ngành được dùng để tính sức mạnh tương đối 20 phiên; không tự suy ra mã thuộc ngành nào. File ngành được lưu cục bộ trong `data/sector_index.csv`.

Mẫu: `sample_data/OHLCV_TEMPLATE.csv` và `sample_data/SECTOR_INDEX_TEMPLATE.csv`. Dữ liệu demo là dữ liệu tổng hợp, không phải dữ liệu thị trường thật.

## 5. Giới hạn và cách đọc kết quả

- Thiếu VNINDEX: không kết luận chắc chắn trạng thái thị trường chung.
- Thiếu dữ liệu ngành: RS ngành hiển thị chưa đủ dữ liệu, không thay bằng VN-Index.
- Thiếu dữ liệu khối ngoại/tự doanh: không suy diễn từ OHLCV.
- Tích lũy/hấp thụ/phân phối là phân loại hành vi giá/khối lượng, không chứng minh ý đồ hay danh tính người giao dịch.
- Mẫu bẫy mới phát hiện phải chờ diễn biến sau sự kiện; không coi sự kiện chờ xác nhận là đã xác nhận.
- Holdout ít hơn 30 lệnh đóng: thống kê chưa đủ mạnh để kết luận hiệu quả. Kết quả lịch sử không đảm bảo kết quả tương lai.

## 6. Kiểm thử phát hành

- **75/75 kiểm thử tự động đạt** trong môi trường đóng gói.
- `compileall` cho app, config, modules và tests: đạt.
- Môi trường đóng gói hiện tại không cài Streamlit, vì vậy chưa xác nhận chạy giao diện tương tác thực tế tại đây. Người dùng cài dependencies theo `requirements.txt` trước khi chạy.
- Chưa có xác nhận hiệu quả trên bộ dữ liệu thị trường thật độc lập đã được kiểm toán. Không xem kết quả demo tổng hợp là bằng chứng hiệu suất.
