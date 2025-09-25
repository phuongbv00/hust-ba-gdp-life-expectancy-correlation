# Phân tích tương quan GDP và Tuổi thọ

Dự án này thực hiện làm sạch dữ liệu, đánh giá chất lượng, và phân tích mối quan hệ giữa GDP và tuổi thọ theo quốc gia,
đồng thời sinh các bảng báo cáo và hình vẽ minh họa. Toàn bộ môi trường và phụ thuộc được quản lý bằng uv.

## 1. Yêu cầu hệ thống

- Python: >= 3.11 (uv sẽ giúp bạn cài đúng phiên bản cục bộ nếu cần)
- Trình quản lý môi trường/đóng gói: [uv](https://docs.astral.sh/uv/) (thay thế pip/venv/poetry/pdm)
- Hệ điều hành: macOS / Linux / WSL / Windows (PowerShell)

## 2. Cài đặt uv

- macOS / Linux:
    - Nếu đã có Homebrew: `brew install uv`
    - Hoặc theo hướng dẫn chính thức: https://docs.astral.sh/uv/getting-started/installation/
- Windows:
    - Sử dụng winget: `winget install astral-sh.uv`
    - Hoặc theo hướng dẫn chính thức ở link trên.

Kiểm tra cài đặt: `uv --version`

## 3. Chuẩn bị dự án với uv

Tại thư mục gốc dự án (nơi có pyproject.toml):

1) Cài đúng phiên bản Python theo yêu cầu của dự án (nếu máy bạn chưa có):

```
uv python install 3.11
```

2) Cài dependencies theo lockfile (ưu tiên tái lập môi trường y hệt):

```
uv sync --frozen
```

- Nếu bạn muốn cài theo pyproject.toml (không dùng lock): `uv sync`
- Lần đầu chạy có thể mất vài phút để biên dịch/scipy/sklearn.

Mặc định, uv tạo môi trường ảo .venv trong thư mục dự án. Bạn không cần kích hoạt thủ công khi dùng `uv run`.

## 4. Dữ liệu đầu vào

- Tệp dữ liệu gốc: `data/Life Expectancy Data.csv`
- Giữ nguyên cấu trúc thư mục như trong repo. Script sẽ đọc trực tiếp từ đường dẫn này.

## 5. Chạy phân tích

Có hai cách tương đương:

- Cách 1: Dùng uv để chạy Python trực tiếp

```
uv run python main.py
```

- Cách 2: Kích hoạt môi trường (tùy chọn) rồi chạy

```
# (Tùy chọn) kích hoạt môi trường nếu bạn muốn chạy python mà không cần uv run
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

python main.py
```

Khi chạy xong, dự án sẽ tạo các báo cáo và hình ảnh trong thư mục `output/`:

- output/reports/*.csv, *.json
- output/figures/*.png
- output/cleaned_dataset.csv

## 6. Cấu trúc thư mục chính

- `main.py`: Điểm vào chính, thực hiện các bước: làm sạch dữ liệu, đánh giá chất lượng, phân tích và vẽ biểu đồ.
- `data/Life Expectancy Data.csv`: Dữ liệu nguồn.
- `output/`: Nơi lưu kết quả sinh ra (tự động tạo nếu chưa có).
- `pyproject.toml`: Khai báo metadata dự án và dependencies.
- `uv.lock`: Khóa phiên bản dependencies để tái lập môi trường lặp lại.
- `REQUIREMENTS.md`: Ghi chú thêm về phụ thuộc/yêu cầu (nếu có).
