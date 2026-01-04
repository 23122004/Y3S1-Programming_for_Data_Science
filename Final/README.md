# Phân tích năng suất lập trình viên khi sử dụng AI  
**Đồ án cuối kỳ – Programming for Data Science – HCMUS**

## Thông tin nhóm (Nhóm 15)
- Đinh Đức Anh Khoa (23122001)  
- Nguyễn Lê Hoàng Trung (23122004)  
- Lê Hoàng Minh Huy (23122033)  

## Tổng quan dự án
Dự án nhằm phân tích tác động của các công cụ AI, mức độ căng thẳng, giấc ngủ và một số yếu tố con người khác đến năng suất làm việc và chất lượng mã nguồn của lập trình viên.

## Dữ liệu
- **Nguồn**: Kaggle – *AI Developer Performance*  
- **Mô tả**: Gồm 1000 bản ghi với 13 thuộc tính, bao quát các khía cạnh về năng suất, chất lượng code, mức độ sử dụng AI và điều kiện con người (stress, giấc ngủ, tải nhận thức).

## Câu hỏi nghiên cứu
1. Việc phụ thuộc nhiều vào AI ảnh hưởng tốt hay xấu năng suất?  
2. Mức độ căng thẳng ảnh hưởng thế nào đến hiệu suất làm việc?  
3. Những kiểu lập trình viên nào sẽ có hiệu suất cao nhất? 
4. Với từng cá nhân cụ thể, yếu tố nào đang kéo tụt hiệu suất của họ và yếu tố nào đang nâng họ lên?
5. Liệu việc gia tăng tiêu thụ cà phê có thực sự thúc đẩy năng suất làm việc hay sẽ tạo ra nhiều ảnh hưởng tiêu cực hơn? 
6. Hiệu quả của việc sử dụng công cụ AI đối với tỷ lệ thành công của nhiệm vụ thay đổi như thế nào giữa nhóm lập trình viên có trạng thái tinh thần ổn định so với nhóm đang bị quá tải ?

## Kết quả chính
- **Cân bằng**: Sử dụng AI ở mức vừa phải, kết hợp kiểm tra lại mã nguồn, cho hiệu quả tốt hơn so với phụ thuộc quá mức.  
- **Làm việc thông minh**: Lập trình viên có mức stress thấp, ngủ đủ và sử dụng AI hiệu quả có năng suất cao hơn so với chỉ làm việc nhiều giờ.  
- **Tải nhận thức**: Tải nhận thức cao là rào cản lớn đối với hiệu suất.  
- **Cà phê**: Không cải thiện đáng kể năng suất và có thể làm tăng số lỗi.  
- **Trạng thái tinh thần**: Lập trình viên có trạng thái tinh thần ổn định sử dụng AI hiệu quả hơn khoảng 26% so với nhóm quá tải.

## Cấu trúc thư mục
```
<root>
├── dataset/
│   ├── processed/    # Dữ liệu đã xử lý
│   └── raw/          # Dữ liệu gốc và mô tả
├── notebooks/
│   ├── 1_eda.ipynb       # EDA
│   ├── 2_questions.ipynb # Phân tích câu hỏi nghiên cứu
│   └── 3_summary.ipynb   # Tổng kết và kết luận
├── src/              # Mã nguồn xử lý và trực quan hóa
├── requirements.txt  # Các thư viện cần thiết
└── README.md

````

## Cách chạy dự án
1. Cài đặt các thư viện cần thiết:
```bash
   pip install -r requirements.txt
````

2. Chạy các notebook theo thứ tự:

   * `notebooks/1_eda.ipynb`
   * `notebooks/2_questions.ipynb`
   * `notebooks/3_summary.ipynb`

## Thư viện sử dụng

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* shap
* scipy

