# AI_BienSo

Nhận diện **biển số xe** bằng YOLO (phát hiện vùng biển số) + PaddleOCR (đọc ký tự) và demo quản lý **vào/ra bãi xe** bằng SQLite.

## Tính năng

- Phát hiện biển số từ **ảnh** hoặc **camera**.
- OCR ký tự biển số bằng **PaddleOCR**.
- Lưu trạng thái xe vào/ra bãi bằng SQLite (`parking_lot.db`):
  - Nếu biển số **chưa tồn tại** → ghi nhận **thời gian vào**.
  - Nếu biển số **đã tồn tại** → coi như **rời bãi** và xóa bản ghi.
- Lưu ảnh biển số đã cắt vào `output_plates/`.

## Yêu cầu

- Windows 10/11
- Python 3.9+ (khuyến nghị 3.10/3.11)
- (Tuỳ chọn) GPU CUDA nếu bạn muốn tăng tốc

## Cài đặt

Tạo môi trường ảo và cài dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install ultralytics opencv-python pillow paddleocr
```

Ghi chú:
- `paddleocr` sẽ kéo thêm một số gói phụ thuộc. Nếu cài đặt lỗi, hãy thử cập nhật `pip`/`setuptools` và cài lại.

## Train mô hình phát hiện biển số (YOLO)

File `main.py` hiện train YOLO với cấu hình:

- Model khởi tạo: `yolo11n.pt`
- Dataset: `mydata.yaml`
- Epochs: `75`

Chạy:

```bash
python main.py
```

### Lưu ý về `mydata.yaml`

Trong `mydata.yaml` đang để `path:` là đường dẫn tuyệt đối trên máy cũ:

- `path: C:\Users\duong\PycharmProjects\PythonProject1\datasets`

Bạn cần sửa `path:` trỏ về dataset của bạn (có `images/train`, `images/val`, `labels/...`) trước khi train.

## Chạy demo nhận diện + bãi xe (Tkinter)

Chạy:

```bash
python predict.py
```

App có các nút:
- **Chọn ảnh**: chọn ảnh để detect + OCR.
- **Chụp ảnh**: mở camera, nhấn **Enter** để chụp và xử lý, nhấn **ESC** để thoát.
- **Chọn video**: hiện mới là khung chức năng (chưa xử lý video).

### Quan trọng: sửa đường dẫn weights trong `predict.py`

`predict.py` đang load YOLO weights bằng đường dẫn tuyệt đối:

- `model = YOLO(r"D:\PythonProject2\runs\detect\train5\weights\best.pt")`

Bạn cần đổi sang:
- đường dẫn **tương đối** trong repo (ví dụ: `runs/detect/train5/weights/best.pt`), hoặc
- đặt weights ở một nơi trên máy bạn và trỏ lại đúng đường dẫn.

Khuyến nghị: dùng biến môi trường hoặc tham số dòng lệnh để cấu hình đường dẫn weights.

## Cấu trúc thư mục (tham khảo)

- `main.py`: train YOLO với `mydata.yaml`
- `predict.py`: giao diện Tkinter (ảnh/camera) + YOLO + PaddleOCR + SQLite
- `mydata.yaml`: cấu hình dataset YOLO
- `datasets/`, `dataset_bienso/`: dữ liệu (không nên commit lên GitHub nếu dung lượng lớn)
- `runs/`: output train của Ultralytics (không nên commit)
- `output_plates/`: ảnh biển số cắt ra khi chạy demo
- `parking_lot.db`: DB SQLite demo

## Gợi ý cải thiện

- Hoàn thiện xử lý **video** (frame-by-frame, tracking, debounce).
- Chuẩn hoá text OCR (loại bỏ khoảng trắng/ký tự nhiễu, regex theo format biển VN).
- Tách logic DB: thay vì `DELETE` khi ra bãi, có thể lưu `exit_time` để thống kê.

## License

Chưa khai báo.

