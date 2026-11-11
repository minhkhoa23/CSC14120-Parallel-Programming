# CSC14120 – Lập trình song song

## Cây thư mục

```text
.
├── Dataset/
│   └──image{1..4}_{left|right}_{1280x720|1920x1080|3840x2160}.jpg
├── Midterm/
│   ├── src/
│   │   ├── v0/
│   │   │   ├── descriptor.cpp                          # Tạo descriptor BRIEF 256 bit cho các keypoint
│   │   │   ├── detector.cpp                            # Phát hiện cực trị DoG và lọc keypoint theo đáp ứng
│   │   │   ├── homography.cpp                          # Ước lượng homography B→A bằng RANSAC và DLT chuẩn hóa
│   │   │   ├── matching.cpp                            # Ghép descriptor bằng khoảng cách Hamming và kiểm tra tỉ lệ
│   │   │   ├── pipeline.cpp                            # Điều phối các giai đoạn V0 và đo thời gian từng giai đoạn
│   │   │   ├── preprocess.cpp                          # Chuyển ảnh sang xám và tạo các mức Gaussian
│   │   │   └── warp_blend.cpp                          # Warp ảnh, nội suy màu và hòa trộn thành panorama
│   │   ├── brief_pattern.h                             # Mẫu cố định 256 cặp điểm dùng trong BRIEF
│   │   ├── common.h                                    # Kiểu dữ liệu, tham số và giao diện chung của pipeline
│   │   ├── cuda_utils.cuh                              # Header giữ chỗ cho tiện ích CUDA trong phiên bản tương lai
│   │   ├── main.cu                                     # CLI stitch: đọc ảnh, chạy V0, ghi panorama và số liệu JSON
│   │   ├── stb_image.h                                 # Thư viện stb bên thứ ba để đọc ảnh
│   │   ├── stb_image_write.h                           # Thư viện stb bên thứ ba để ghi ảnh
│   │   ├── stb_impl.cpp                                # Cài đặt stb và hiện thực hàm đọc/ghi ảnh
│   │   └── versions.h                                  # Header tổng hợp giao diện các phiên bản, hiện dùng chung V0
│   ├── Group21_Midterm.ipynb                           # Report
│   └── src.zip                                         # Bản nén của mã nguồn trong src/
├── .gitignore
└── README.md
```