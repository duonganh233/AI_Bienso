import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
import os
import sqlite3
from datetime import datetime

# Khởi tạo mô hình YOLO
model = YOLO(r"D:\PythonProject2\runs\detect\train5\weights\best.pt")

# Khởi tạo PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Thư mục lưu ảnh biển số cắt ra
output_dir = "output_plates"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Kết nối cơ sở dữ liệu
conn = sqlite3.connect("parking_lot.db")
c = conn.cursor()

# Tạo bảng nếu chưa tồn tại
c.execute('''CREATE TABLE IF NOT EXISTS vehicles (
                plate TEXT PRIMARY KEY,
                entry_time TEXT,
                exit_time TEXT
            )''')
conn.commit()


# Hàm tiền xử lý ảnh
def preprocess_image(image):
    """Tiền xử lý hình ảnh để cải thiện khả năng nhận diện."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_image


# Hàm xử lý thông tin biển số
def handle_plate_info(plate):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Kiểm tra xem biển số đã tồn tại trong cơ sở dữ liệu chưa
    c.execute("SELECT * FROM vehicles WHERE plate = ?", (plate,))
    result = c.fetchone()

    if result:
        # Nếu biển số đã tồn tại, cập nhật thời gian rời bãi và xóa khỏi cơ sở dữ liệu
        c.execute("DELETE FROM vehicles WHERE plate = ?", (plate,))
        conn.commit()
        label_result.config(text=f"Xe {plate} đã rời bãi lúc {current_time}")
    else:
        # Nếu biển số chưa tồn tại, thêm vào cơ sở dữ liệu với thời gian vào
        c.execute("INSERT INTO vehicles (plate, entry_time) VALUES (?, ?)", (plate, current_time))
        conn.commit()
        label_result.config(text=f"Xe {plate} đã vào bãi lúc {current_time}")


# Hàm xử lý ảnh
def process_image():
    global original_img, processed_img, plate_img

    # Mở hộp thoại để chọn ảnh
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh cần nhận diện",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    # Đọc ảnh gốc
    image = cv2.imread(file_path)
    original_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    # Hiển thị ảnh gốc
    canvas_original.create_image(0, 0, anchor=tk.NW, image=original_img)

    # Chạy mô hình YOLO
    results = model(file_path)

    for r in results:
        for idx, box in enumerate(r.boxes):
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Cắt vùng biển số
            plate_region = image[y1:y2, x1:x2]

            # Lưu ảnh biển số vào file
            plate_filename = os.path.join(output_dir, f"plate_{idx}.png")
            cv2.imwrite(plate_filename, plate_region)

            # Tiền xử lý biển số cho OCR
            thresh_plate = preprocess_image(plate_region)

            # Nhận diện ký tự từ ảnh biển số với PaddleOCR
            ocr_results = ocr.ocr(thresh_plate, cls=True)

            # Lấy kết quả nhận diện
            detected_text = " ".join([line[1][0] for line in ocr_results[0]])

            # Xử lý thông tin biển số
            handle_plate_info(detected_text)

            # Hiển thị vùng biển số đã cắt
            plate_h, plate_w, _ = plate_region.shape
            canvas_plate.config(width=plate_w, height=plate_h)
            plate_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)))
            canvas_plate.create_image(0, 0, anchor=tk.NW, image=plate_img)

            # Hiển thị ảnh biển số sau xử lý
            processed_h, processed_w = thresh_plate.shape
            canvas_processed.config(width=processed_w, height=processed_h)
            processed_img = ImageTk.PhotoImage(Image.fromarray(thresh_plate))
            canvas_processed.create_image(0, 0, anchor=tk.NW, image=processed_img)

            print(f"Vùng biển số đã được lưu tại: {plate_filename}")
            print(f"Kết quả nhận diện: {detected_text}")
            return  # Dừng sau khi xử lý vùng biển số đầu tiên



# Hàm mở camera và chụp ảnh khi nhấn Enter
def capture_image_with_enter():
    global original_img

    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    print("Nhấn Enter để chụp ảnh, nhấn ESC để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ camera")
            break

        # Hiển thị khung hình
        cv2.imshow("Camera", frame)

        # Kiểm tra phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Phím Enter
            # Lưu ảnh tạm thời
            temp_path = "temp_captured_image.jpg"
            cv2.imwrite(temp_path, frame)

            # Hiển thị ảnh gốc
            original_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            canvas_original.create_image(0, 0, anchor=tk.NW, image=original_img)

            # Xử lý ảnh vừa chụp
            process_image_with_path(temp_path)
            break
        elif key == 27:  # Phím ESC
            break

    cap.release()
    cv2.destroyAllWindows()



# Hàm xử lý ảnh từ đường dẫn (dùng cho ảnh chụp)
def process_image_with_path(file_path):
    global original_img, processed_img, plate_img

    # Đọc ảnh gốc
    image = cv2.imread(file_path)

    # Chạy mô hình YOLO
    results = model(file_path)

    for r in results:
        for idx, box in enumerate(r.boxes):
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Cắt vùng biển số
            plate_region = image[y1:y2, x1:x2]

            # Lưu ảnh biển số vào file
            plate_filename = os.path.join(output_dir, f"plate_{idx}.png")
            cv2.imwrite(plate_filename, plate_region)

            # Tiền xử lý biển số cho OCR
            thresh_plate = preprocess_image(plate_region)

            # Nhận diện ký tự từ ảnh biển số với PaddleOCR
            ocr_results = ocr.ocr(thresh_plate, cls=True)

            # Lấy kết quả nhận diện
            detected_text = " ".join([line[1][0] for line in ocr_results[0]])

            # Xử lý thông tin biển số
            handle_plate_info(detected_text)

            # Hiển thị vùng biển số đã cắt
            plate_h, plate_w, _ = plate_region.shape
            canvas_plate.config(width=plate_w, height=plate_h)
            plate_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)))
            canvas_plate.create_image(0, 0, anchor=tk.NW, image=plate_img)

            # Hiển thị ảnh biển số sau xử lý
            processed_h, processed_w = thresh_plate.shape
            canvas_processed.config(width=processed_w, height=processed_h)
            processed_img = ImageTk.PhotoImage(Image.fromarray(thresh_plate))
            canvas_processed.create_image(0, 0, anchor=tk.NW, image=processed_img)

            print(f"Vùng biển số đã được lưu tại: {plate_filename}")
            print(f"Kết quả nhận diện: {detected_text}")
            return  # Dừng sau khi xử lý vùng biển số đầu tiên

def process_video_from_file():
    # Mở hộp thoại để chọn video
    file_path = filedialog.askopenfilename(
        title="Chọn video cần nhận diện",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not file_path:
        return

# Tạo giao diện
root = tk.Tk()
root.title("Nhận diện biển số xe")

# Khung ảnh gốc
frame_original = tk.Frame(root, width=500, height=400)
frame_original.pack(side=tk.LEFT, padx=10, pady=10)
canvas_original = Canvas(frame_original, width=500, height=400, bg="gray")
canvas_original.pack()

# Khung ảnh biển số
frame_plate = tk.Frame(root)
frame_plate.pack(side=tk.LEFT, padx=10, pady=10)
canvas_plate = Canvas(frame_plate, bg="gray")
canvas_plate.pack()

# Khung ảnh xử lý
frame_processed = tk.Frame(root)
frame_processed.pack(side=tk.LEFT, padx=10, pady=10)
canvas_processed = Canvas(frame_processed, bg="gray")
canvas_processed.pack()

# Nút chọn ảnh
button_select = Button(root, text="Chọn ảnh", command=process_image)
button_select.pack(side=tk.TOP, pady=10)

# Nút chụp ảnh với Enter
button_capture = Button(root, text="Chụp ảnh", command=capture_image_with_enter)
button_capture.pack(side=tk.TOP, pady=10)

# Nút chọn video
button_select_video = Button(root, text="Chọn video", command=process_video_from_file)
button_select_video.pack(side=tk.TOP, pady=10)

# Hiển thị thông tin biển số
label_result = Label(root, text="Biển số xe: ", font=("Arial", 14))
label_result.pack(side=tk.BOTTOM, pady=10)

# Chạy ứng dụng
root.mainloop()

# Đóng kết nối cơ sở dữ liệu khi thoát
conn.close()


# Tạo giao diện
root = tk.Tk()
root.title("Nhận diện biển số xe")

# Khung ảnh gốc
frame_original = tk.Frame(root, width=500, height=400)
frame_original.pack(side=tk.LEFT, padx=10, pady=10)
canvas_original = Canvas(frame_original, width=500, height=400, bg="gray")
canvas_original.pack()

# Khung ảnh biển số
frame_plate = tk.Frame(root)
frame_plate.pack(side=tk.LEFT, padx=10, pady=10)
canvas_plate = Canvas(frame_plate, bg="gray")
canvas_plate.pack()

# Khung ảnh xử lý
frame_processed = tk.Frame(root)
frame_processed.pack(side=tk.LEFT, padx=10, pady=10)
canvas_processed = Canvas(frame_processed, bg="gray")
canvas_processed.pack()

# Nút chọn ảnh
button_select = Button(root, text="Chọn ảnh", command=process_image)
button_select.pack(side=tk.TOP, pady=10)

# Nút chụp ảnh với Enter
button_capture = Button(root, text="Chụp ảnh", command=capture_image_with_enter)
button_capture.pack(side=tk.TOP, pady=10)

button_select_video = Button(root, text="Chọn video", command=process_video_from_file)
button_select_video.pack(side=tk.TOP, pady=10)

# Hiển thị thông tin biển số
label_result = Label(root, text="Biển số xe: ", font=("Arial", 14))
label_result.pack(side=tk.BOTTOM, pady=10)

# Chạy ứng dụng
root.mainloop()

# Đóng kết nối cơ sở dữ liệu khi thoát
conn.close()
