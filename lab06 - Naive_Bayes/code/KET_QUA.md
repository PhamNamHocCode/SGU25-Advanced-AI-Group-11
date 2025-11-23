# KẾT QUẢ LAB 06 - NAÏVE BAYES CLASSIFIER

## 📊 Tổng quan

**Lab:** Naive Bayes Classifier  
**Môn:** Advanced AI  
**Trường:** Đại học Sài Gòn (SGU)  
**Thời gian thực hiện:** Tháng 11/2025

---

## ✅ Câu 1: Phân loại loài hoa Iris

### Thông tin dataset
- **Dataset:** UCI Iris Dataset
- **Tổng số mẫu:** 150
- **Số classes:** 3 (Setosa, Versicolor, Virginica)
- **Số features:** 4 (SepalLength, SepalWidth, PetalLength, PetalWidth)
- **Train/Test split:** 120/30 (80%/20%)

### Kết quả
- ✅ **Accuracy:** 93.33% (28/30 đúng)
- ⏱️ **Thời gian training:** < 1 giây
- 📈 **Đánh giá:** Rất tốt cho Naive Bayes với dataset nhỏ

### Phân tích
**Điểm mạnh:**
- Accuracy cao (>90%)
- Chạy nhanh, ổn định
- Phù hợp với dataset có features liên tục

**Các lỗi dự đoán:**
- 2 mẫu bị nhầm giữa Versicolor và Virginica
- Nguyên nhân: 2 loài này có đặc trưng tương tự nhau (overlap)

### File code
- `cau1_iris_demo.py` - Demo chi tiết với output đẹp
- `Naive_Bayes.py` - Class Naive Bayes (đã fix lỗi iteritems)

---

## ✅ Câu 2: Nhận dạng ký tự chữ cái (A-Z)

### Thông tin dataset
- **Dataset:** UCI Letter Recognition Dataset
- **Tổng số mẫu:** 20,000
- **Số classes:** 26 (A-Z)
- **Số features:** 16 (statistical moments, edge counts)
- **Train/Test split:** 16,000/4,000 (80%/20%)

### Kết quả
- ✅ **Accuracy:** 62.52% (2,501/4,000 đúng)
- ⏱️ **Thời gian training:** ~2-3 giây
- 📈 **Đánh giá:** Chấp nhận được cho Naive Bayes với bài toán phức tạp

### Phân tích theo từng ký tự

**Top 5 ký tự dự đoán TỐT NHẤT:**
1. M: 89.6%
2. A: 84.0%
3. I: 78.2%
4. W: 77.0%
5. V: 77.2%

**Top 5 ký tự dự đoán TỆ NHẤT:**
1. H: 27.2%
2. S: 28.0%
3. Y: 34.5%
4. E: 38.2%
5. K: 45.2%

**Phân tích:**
- Ký tự có hình dạng đơn giản, đặc trưng rõ ràng → accuracy cao (M, A, I)
- Ký tự phức tạp, dễ nhầm với ký tự khác → accuracy thấp (H, S, E)
- Ví dụ: H dễ nhầm với K, N; S dễ nhầm với 5, Z

### So sánh với các thuật toán khác

| Thuật toán | Accuracy | Ghi chú |
|-----------|----------|---------|
| **Naive Bayes** | **62.52%** | **Đơn giản, nhanh** |
| k-NN (k=5) | ~70-75% | Chậm hơn |
| Decision Tree | ~75-80% | Dễ overfit |
| SVM | ~85-90% | Chậm train |
| Neural Network | >95% | Cần nhiều data, chậm train |

**Kết luận:** Naive Bayes đạt accuracy trung bình nhưng rất nhanh và đơn giản.

### File code
- `cau2_letter_recognition.py` - Implementation đầy đủ với class riêng
- `LetterRecognitionNaiveBayes` - Class tự implement Gaussian Naive Bayes

---

## 🎯 Tổng kết

### Ưu điểm Naive Bayes
✅ **Tốc độ:** Training và prediction rất nhanh  
✅ **Đơn giản:** Dễ implement, dễ hiểu  
✅ **Hiệu quả với dataset nhỏ:** Câu 1 chỉ 120 samples train vẫn đạt 93%  
✅ **Probabilistic output:** Cho xác suất, không chỉ label  
✅ **Robust:** Ít bị overfit  

### Nhược điểm
❌ **Naive assumption:** Giả định features độc lập (thường sai)  
❌ **Accuracy trung bình:** Thấp hơn các thuật toán phức tạp (SVM, NN)  
❌ **Sensitive với outliers:** Gaussian assumption có thể sai  

### Khi nào dùng Naive Bayes?
✅ Baseline model (model đầu tiên để thử)  
✅ Dataset nhỏ (<10,000 samples)  
✅ Cần tốc độ (real-time classification)  
✅ Text classification (spam filter, sentiment analysis)  
✅ Multi-class problems với nhiều classes  

### Khi nào KHÔNG dùng?
❌ Cần accuracy cao (>90%)  
❌ Features có correlation mạnh  
❌ Dataset lớn (có thể dùng model phức tạp hơn)  

---

## 📁 Files đã tạo

```
lab06 - Naive_Bayes/code/
├── cau1_iris_demo.py              ✅ Câu 1: Iris demo
├── cau2_letter_recognition.py     ✅ Câu 2: Letter Recognition demo
├── run_all.py                     ✅ Script chạy tất cả
├── Naive_Bayes.py                 ✅ Class Naive Bayes (đã fix)
├── README_LAB06.md                ✅ Hướng dẫn chi tiết
├── KET_QUA.md                     ✅ File này
└── dataset/
    ├── Iris.csv                   ✅ Dataset Iris (có sẵn)
    └── letter-recognition.data    ✅ Dataset Letter (đã tải)
```

---

## 🚀 Cách chạy

### Chạy từng câu
```bash
# Câu 1
python cau1_iris_demo.py

# Câu 2
python cau2_letter_recognition.py
```

### Chạy tất cả
```bash
python run_all.py
```

### Output mẫu
```
LAB 06 - NAÏVE BAYES CLASSIFIER
Đại học Sài Gòn (SGU) - Advanced AI

📋 Danh sách bài tập:
   1. Câu 1: Phân loại loài hoa Iris (3 classes, 4 features)
   2. Câu 2: Nhận dạng ký tự A-Z (26 classes, 16 features)

⏱️  Tổng thời gian chạy: 5.51 giây

✅ Câu 1 (Iris): THÀNH CÔNG (Accuracy: 93.33%)
✅ Câu 2 (Letter Recognition): THÀNH CÔNG (Accuracy: 62.52%)

🎉 ĐÃ HOÀN THÀNH TẤT CẢ BÀI TẬP!
```

---

## 📚 Tài liệu tham khảo

1. **UCI Machine Learning Repository**
   - Iris: https://archive.ics.uci.edu/ml/datasets/iris
   - Letter: https://archive.ics.uci.edu/ml/datasets/letter+recognition

2. **Naive Bayes Theory**
   - https://alphacoder.xyz/naive-bayes
   - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

3. **Paper gốc**
   - Frey, P. W., & Slate, D. J. (1991). Letter recognition using Holland-style adaptive classifiers.

---

## 🎓 Bài học rút ra

1. **Naive Bayes phù hợp làm baseline model** - Nhanh, đơn giản, cho kết quả ban đầu
2. **Dataset nhỏ (Iris) → accuracy cao** - Naive Bayes hoạt động tốt với ít data
3. **Dataset phức tạp (Letter) → accuracy trung bình** - Cần thuật toán mạnh hơn (SVM, Neural Network)
4. **Preprocessing quan trọng** - Chuẩn hóa data, xử lý outliers giúp cải thiện accuracy
5. **Gaussian assumption không phải lúc nào cũng đúng** - Một số features có thể không follow normal distribution

---

**Ngày hoàn thành:** 23/11/2025  
**Trạng thái:** ✅ HOÀN THÀNH TẤT CẢ YÊU CẦU
