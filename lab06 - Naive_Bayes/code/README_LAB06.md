# Lab 06 - Naïve Bayes Classifier

## 📋 Mô tả

Bài lab thực hành thuật toán **Naïve Bayes** cho 2 bài toán phân loại:

### Câu 1: Phân loại loài hoa Iris 🌸
- **Dataset**: UCI Iris Dataset
- **Classes**: 3 loài hoa (Setosa, Versicolor, Virginica)
- **Features**: 4 đặc trưng (SepalLength, SepalWidth, PetalLength, PetalWidth)
- **Samples**: 150 mẫu
- **Algorithm**: Gaussian Naïve Bayes
- **Tham khảo**: 
  - https://alphacoder.xyz/naive-bayes
  - https://archive.ics.uci.edu/ml/datasets/iris

### Câu 2: Nhận dạng ký tự chữ cái (A-Z) 🔤
- **Dataset**: UCI Letter Recognition Dataset
- **Classes**: 26 ký tự (A to Z)
- **Features**: 16 đặc trưng số (statistical moments, edge counts)
- **Samples**: 20,000 mẫu (16,000 train / 4,000 test)
- **Algorithm**: Gaussian Naïve Bayes
- **Tham khảo**: https://archive.ics.uci.edu/ml/datasets/letter+recognition

---

## 📁 Cấu trúc thư mục

```
lab06 - Naive_Bayes/
├── code/
│   ├── dataset/
│   │   ├── Iris.csv                        # Dataset Iris (có sẵn)
│   │   └── letter-recognition.data         # Dataset Letter Recognition (đã tải)
│   ├── Naive_Bayes.py                      # Class Naive Bayes gốc
│   ├── cau1_iris_demo.py                   # ✅ Câu 1: Demo Iris
│   ├── cau2_letter_recognition.py          # ✅ Câu 2: Demo Letter Recognition
│   ├── run_all.py                          # Script chạy tất cả
│   └── README.md                           # File này
└── Lab_Naive_Bayes.pdf                     # Đề bài
```

---

## 🚀 Hướng dẫn chạy

### Cài đặt thư viện

```bash
pip install pandas numpy
```

### Chạy từng câu

**Câu 1: Iris Classification**
```bash
python cau1_iris_demo.py
```

**Câu 2: Letter Recognition**
```bash
python cau2_letter_recognition.py
```

### Chạy tất cả

```bash
python run_all.py
```

---

## 📊 Kết quả dự kiến

### Câu 1: Iris Dataset
- **Train/Test Split**: 80/20 (120 train / 30 test)
- **Accuracy**: ~93-97% (tùy random shuffle)
- **Thời gian train**: < 1 giây

### Câu 2: Letter Recognition
- **Train/Test Split**: 80/20 (16,000 train / 4,000 test)
- **Accuracy**: ~70-75%
- **Thời gian train**: ~2-3 giây
- **Lưu ý**: Độ chính xác có thể thấp hơn các thuật toán phức tạp hơn (SVM, Neural Network) do:
  - Giả định độc lập giữa các features (naïve assumption)
  - Không capture được correlation giữa các features

---

## 🧠 Giải thích thuật toán

### Naïve Bayes Classifier

**Công thức Bayes:**
```
P(Class | Features) = P(Features | Class) * P(Class) / P(Features)
```

**Naïve Assumption:**
```
P(Features | Class) = P(f1 | Class) * P(f2 | Class) * ... * P(fn | Class)
```

**Gaussian Distribution (cho continuous features):**
```
P(x | μ, σ²) = (1 / √(2πσ²)) * exp(-(x-μ)² / (2σ²))
```

### Các bước thuật toán:

1. **Training Phase:**
   - Tính prior probability: P(Class) = count(Class) / total_samples
   - Với mỗi class, tính mean (μ) và variance (σ²) cho từng feature
   
2. **Prediction Phase:**
   - Với mỗi class, tính posterior probability:
     - P(Class | Features) ∝ P(Class) * ∏ P(feature_i | Class)
   - Chọn class có posterior probability cao nhất

---

## 📈 Chi tiết Implementation

### Câu 1: Iris (`cau1_iris_demo.py`)

**Class sử dụng:** `Naive_Bayes` (từ file `Naive_Bayes.py`)

**Features:**
- Sử dụng Gaussian distribution cho 4 features liên tục
- Tính mean/variance theo từng class bằng pandas groupby
- Demo với mẫu ngẫu nhiên từ test set

**Output:**
- Accuracy tổng thể
- Danh sách các prediction sai
- Demo phân loại 3 mẫu ngẫu nhiên

### Câu 2: Letter Recognition (`cau2_letter_recognition.py`)

**Class sử dụng:** `LetterRecognitionNaiveBayes` (tự implement)

**Features:**
- Sử dụng numpy arrays cho tính toán nhanh
- Implement Gaussian Naive Bayes từ đầu
- Xử lý 26 classes và 16 features
- Tính per-class accuracy (confusion matrix)

**Output:**
- Accuracy tổng thể
- Accuracy theo từng ký tự (A-Z)
- Ký tự dự đoán tốt nhất/tệ nhất
- Demo phân loại 5 mẫu ngẫu nhiên

---

## 🎯 Ưu điểm và Nhược điểm

### Ưu điểm:
✅ **Đơn giản, nhanh**: Training và prediction rất nhanh  
✅ **Ít data**: Hoạt động tốt với dataset nhỏ  
✅ **Hiệu quả với high-dimensional data**: 16 features vẫn OK  
✅ **Xác suất**: Cho output là probability, không chỉ label  

### Nhược điểm:
❌ **Naïve assumption**: Giả định features độc lập (thường sai trong thực tế)  
❌ **Zero probability**: Nếu feature chưa thấy trong training → P=0 (cần Laplace smoothing)  
❌ **Continuous features**: Phải giả định distribution (Gaussian, Multinomial...)  
❌ **Accuracy**: Thấp hơn các thuật toán phức tạp hơn (SVM, Random Forest, Neural Network)  

---

## 📚 Tài liệu tham khảo

1. **UCI Machine Learning Repository**
   - Iris Dataset: https://archive.ics.uci.edu/ml/datasets/iris
   - Letter Recognition: https://archive.ics.uci.edu/ml/datasets/letter+recognition

2. **Naive Bayes Tutorial**
   - https://alphacoder.xyz/naive-bayes
   - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

3. **Paper gốc Letter Recognition**
   - Frey, P. W., & Slate, D. J. (1991). Letter recognition using Holland-style adaptive classifiers. Machine learning, 6(2), 161-182.

---

## 👥 Thông tin

**Lab:** Naive Bayes Classifier  
**Môn:** Advanced AI  
**Trường:** Đại học Sài Gòn (SGU)  
**Năm:** 2025

---

## 📝 Ghi chú

- Dataset **Iris.csv** đã có sẵn trong folder `dataset/`
- Dataset **letter-recognition.data** đã được tải tự động từ UCI Repository
- Code đã được test và chạy thành công trên Python 3.8+
- Kết quả có thể khác nhau mỗi lần chạy do random shuffle

**Lưu ý quan trọng:** Với Letter Recognition, accuracy ~70-75% là bình thường cho Naive Bayes. Để đạt accuracy cao hơn (>90%), cần dùng các thuật toán phức tạp hơn như SVM hoặc CNN.
