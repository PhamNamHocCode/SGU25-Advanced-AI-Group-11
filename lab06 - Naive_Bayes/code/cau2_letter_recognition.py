"""
CÂU 2: Nhận dạng ký tự (A-Z) dùng thuật toán Naïve Bayes
Dataset: UCI Letter Recognition Dataset
Classes: 26 classes (A to Z)
Features: 16-D feature vectors
Samples: 20,000 samples
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class LetterRecognitionNaiveBayes:
    """
    Naive Bayes Classifier cho Letter Recognition
    Sử dụng Gaussian Naive Bayes với 16 features liên tục
    """
    
    def __init__(self):
        self.classes = []
        self.class_priors = {}
        self.feature_means = {}  # Mean của từng feature theo từng class
        self.feature_variances = {}  # Variance của từng feature theo từng class
        
    def fit(self, X_train, y_train):
        """
        Huấn luyện model
        X_train: numpy array shape (n_samples, 16)
        y_train: numpy array shape (n_samples,)
        """
        print("   🔧 Đang tính toán statistics...")
        
        # Lấy danh sách classes
        self.classes = np.unique(y_train)
        n_samples = len(y_train)
        
        # Tính prior probability cho mỗi class
        for cls in self.classes:
            class_samples = np.sum(y_train == cls)
            self.class_priors[cls] = class_samples / n_samples
        
        # Tính mean và variance cho từng feature theo từng class
        for cls in self.classes:
            # Lấy tất cả samples thuộc class này
            X_class = X_train[y_train == cls]
            
            # Tính mean và variance cho từng feature
            self.feature_means[cls] = np.mean(X_class, axis=0)
            self.feature_variances[cls] = np.var(X_class, axis=0) + 1e-9  # Thêm epsilon tránh chia 0
        
        print(f"   ✅ Đã train cho {len(self.classes)} classes")
        print(f"   ✅ Mỗi class có {len(self.feature_means[self.classes[0]])} features")
    
    def _calculate_likelihood(self, x, mean, variance):
        """
        Tính Gaussian probability density function
        P(x | μ, σ²) = (1 / √(2πσ²)) * exp(-(x-μ)² / (2σ²))
        """
        coefficient = 1.0 / np.sqrt(2 * np.pi * variance)
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return coefficient * exponent
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất cho mỗi class
        Trả về: numpy array shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            for j, cls in enumerate(self.classes):
                # Tính P(class)
                prior = self.class_priors[cls]
                
                # Tính P(features | class) = ∏ P(feature_i | class)
                likelihood = 1.0
                for k in range(len(x)):
                    likelihood *= self._calculate_likelihood(
                        x[k],
                        self.feature_means[cls][k],
                        self.feature_variances[cls][k]
                    )
                
                # P(class | features) ∝ P(class) * P(features | class)
                probabilities[i, j] = prior * likelihood
        
        # Normalize (optional, không cần thiết cho classification)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = np.divide(probabilities, row_sums, 
                                  out=np.zeros_like(probabilities), 
                                  where=row_sums!=0)
        
        return probabilities
    
    def predict(self, X):
        """
        Dự đoán class cho mỗi sample
        Trả về: numpy array shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        predictions = np.array([self.classes[np.argmax(prob)] for prob in probabilities])
        return predictions
    
    def score(self, X_test, y_test):
        """
        Tính accuracy trên test set
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy


def load_letter_recognition_data(filepath):
    """
    Load UCI Letter Recognition dataset
    Format: letter,x1,x2,...,x16
    """
    print(f"   📂 Đọc file: {filepath}")
    
    # Đọc dataset (không có header)
    df = pd.read_csv(filepath, header=None)
    
    # Cột 0 là letter (class), cột 1-16 là features
    y = df.iloc[:, 0].values  # Letters (A-Z)
    X = df.iloc[:, 1:].values  # 16 features
    
    print(f"   ✅ Đã load {len(X)} samples")
    print(f"   ✅ Số features: {X.shape[1]}")
    print(f"   ✅ Số classes: {len(np.unique(y))}")
    
    return X, y


def main():
    print("=" * 80)
    print("CÂU 2: NHẬN DẠNG KÝ TỰ (A-Z) BẰNG NAÏVE BAYES")
    print("=" * 80)
    
    # 1. Load dataset
    print("\n📊 BƯỚC 1: Tải UCI Letter Recognition Dataset")
    try:
        X, y = load_letter_recognition_data("dataset/letter-recognition.data")
    except FileNotFoundError:
        print("   ❌ KHÔNG TÌM THẤY FILE: dataset/letter-recognition.data")
        print("   💡 Hướng dẫn tải:")
        print("      1. Truy cập: https://archive.ics.uci.edu/ml/datasets/letter+recognition")
        print("      2. Tải file 'letter-recognition.data'")
        print("      3. Lưu vào folder: lab06 - Naive_Bayes/code/dataset/")
        return
    
    # 2. Thống kê dataset
    print("\n📈 BƯỚC 2: Thống kê dữ liệu")
    unique_letters, counts = np.unique(y, return_counts=True)
    print(f"   - Tổng số mẫu: {len(X)}")
    print(f"   - Số ký tự (classes): {len(unique_letters)}")
    print(f"   - Danh sách ký tự: {', '.join(unique_letters)}")
    print(f"   - Số mẫu mỗi class: {counts[0]} (cân bằng: {len(set(counts)) == 1})")
    
    # 3. Split train/test
    print("\n✂️ BƯỚC 3: Chia tập train/test")
    # Dataset gốc đã được shuffle sẵn theo đề bài
    # 16,000 samples đầu là train, 4,000 samples sau là test
    train_size = 16000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"   - Train set: {len(X_train)} mẫu (80%)")
    print(f"   - Test set: {len(X_test)} mẫu (20%)")
    
    # 4. Train Naive Bayes
    print("\n🧠 BƯỚC 4: Huấn luyện Naïve Bayes Classifier")
    nb = LetterRecognitionNaiveBayes()
    nb.fit(X_train, y_train)
    
    print("\n   Prior Probabilities (P(class)):")
    for i, (letter, prob) in enumerate(nb.class_priors.items()):
        if i % 5 == 0:
            print("   ", end="")
        print(f"P({letter})={prob:.4f}", end="  ")
        if (i + 1) % 5 == 0:
            print()
    print()
    
    # 5. Evaluate on test set
    print("\n🎯 BƯỚC 5: Đánh giá trên tập test")
    print("   ⏳ Đang dự đoán...")
    
    y_pred = nb.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\n📈 KẾT QUẢ TỔNG QUÁT:")
    print(f"   - Tổng số mẫu test: {len(y_test)}")
    print(f"   - Dự đoán đúng: {np.sum(y_pred == y_test)}")
    print(f"   - Dự đoán sai: {np.sum(y_pred != y_test)}")
    print(f"   - ĐỘ CHÍNH XÁC (ACCURACY): {accuracy:.2%}")
    
    # 6. Confusion matrix (per-class accuracy)
    print("\n📊 BƯỚC 6: Độ chính xác theo từng ký tự")
    print("-" * 80)
    
    class_accuracies = {}
    for letter in unique_letters:
        mask = y_test == letter
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            class_accuracies[letter] = class_acc
    
    # Hiển thị theo hàng 5 ký tự
    for i, (letter, acc) in enumerate(class_accuracies.items()):
        if i % 5 == 0:
            print("   ", end="")
        print(f"{letter}: {acc:>5.1%}", end="  ")
        if (i + 1) % 5 == 0:
            print()
    print()
    
    # Tìm ký tự dự đoán tốt nhất và tệ nhất
    best_letter = max(class_accuracies, key=class_accuracies.get)
    worst_letter = min(class_accuracies, key=class_accuracies.get)
    
    print(f"\n   ✅ Ký tự dự đoán TỐT NHẤT: {best_letter} ({class_accuracies[best_letter]:.1%})")
    print(f"   ❌ Ký tự dự đoán TỆ NHẤT: {worst_letter} ({class_accuracies[worst_letter]:.1%})")
    
    # 7. Demo phân loại
    print("\n" + "=" * 80)
    print("🔤 DEMO: Nhận dạng ký tự mới")
    print("=" * 80)
    
    # Lấy 5 mẫu ngẫu nhiên để demo
    demo_indices = np.random.choice(len(X_test), size=5, replace=False)
    
    for idx in demo_indices:
        features = X_test[idx]
        actual = y_test[idx]
        predicted = nb.predict(features.reshape(1, -1))[0]
        
        print(f"\nMẫu: {features[:8]}... (16 features)")
        print(f"   Dự đoán: {predicted}")
        print(f"   Thực tế: {actual}")
        print(f"   Kết quả: {'✅ ĐÚNG' if predicted == actual else '❌ SAI'}")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH CÂU 2")
    print("=" * 80)


if __name__ == "__main__":
    main()
