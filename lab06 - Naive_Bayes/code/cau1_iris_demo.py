"""
CÂU 1: Phân loại loài hoa Iris dựa trên thuật toán Naïve Bayes
Dataset: UCI Iris Dataset
Classes: 3 loài hoa (Setosa, Versicolor, Virginica)
Features: 4 đặc trưng (SepalLength, SepalWidth, PetalLength, PetalWidth)
"""

import pandas as pd
import numpy as np
import random
from Naive_Bayes import Naive_Bayes

def main():
    print("=" * 80)
    print("CÂU 1: PHÂN LOẠI LOÀI HOA IRIS BẰNG NAÏVE BAYES")
    print("=" * 80)
    
    # 1. Load dataset
    print("\n📊 BƯỚC 1: Tải dữ liệu Iris")
    df = pd.read_csv("dataset/Iris.csv")
    print(f"   - Tổng số mẫu: {len(df)}")
    print(f"   - Số features: {len(df.columns) - 2} (bỏ Id và Species)")
    print(f"   - Các loài hoa: {df['Species'].unique()}")
    
    # 2. Preprocessing
    print("\n🔧 BƯỚC 2: Tiền xử lý dữ liệu")
    df.drop(['Id'], axis=1, inplace=True)
    data_set = df.values.tolist()
    random.shuffle(data_set)
    print("   - Đã xóa cột Id")
    print("   - Đã shuffle dữ liệu ngẫu nhiên")
    
    # 3. Split train/test (80/20)
    print("\n✂️ BƯỚC 3: Chia tập train/test")
    train_size = int(0.8 * len(data_set))
    train_data = pd.DataFrame(data_set[:train_size])
    test_data = pd.DataFrame(data_set[train_size:])
    print(f"   - Train set: {len(train_data)} mẫu (80%)")
    print(f"   - Test set: {len(test_data)} mẫu (20%)")
    
    # 4. Train Naive Bayes
    print("\n🧠 BƯỚC 4: Huấn luyện Naïve Bayes Classifier")
    nb = Naive_Bayes(train_data)
    print("   - Đã tính mean và variance cho từng feature theo từng class")
    print("   - Đã tính prior probabilities:")
    for species, prob in nb.class_probabilities.items():
        print(f"      P({species}) = {prob:.4f}")
    
    # 5. Test & Evaluate
    print("\n🎯 BƯỚC 5: Đánh giá trên tập test")
    print("-" * 80)
    
    correct = 0
    total = 0
    wrong_predictions = []
    
    for row in test_data.itertuples():
        feature_set = row[1:5]
        actual_class = row[5]
        predicted_class = nb.predict(feature_set)
        
        if predicted_class == actual_class:
            correct += 1
        else:
            wrong_predictions.append({
                'features': feature_set,
                'predicted': predicted_class,
                'actual': actual_class
            })
        total += 1
    
    accuracy = correct / total
    
    print(f"\n📈 KẾT QUẢ:")
    print(f"   - Tổng số mẫu test: {total}")
    print(f"   - Dự đoán đúng: {correct}")
    print(f"   - Dự đoán sai: {len(wrong_predictions)}")
    print(f"   - ĐỘ CHÍNH XÁC (ACCURACY): {accuracy:.2%}")
    
    if wrong_predictions:
        print(f"\n❌ CÁC DỰ ĐOÁN SAI:")
        for i, wp in enumerate(wrong_predictions[:5], 1):  # Chỉ hiển thị 5 sai đầu tiên
            print(f"   {i}. Features: {wp['features']}")
            print(f"      Predicted: {wp['predicted']}, Actual: {wp['actual']}")
    
    # 6. Demo phân loại mới
    print("\n" + "=" * 80)
    print("🌸 DEMO: Phân loại mẫu hoa mới")
    print("=" * 80)
    
    # Lấy 3 mẫu ngẫu nhiên từ test set để demo
    demo_samples = test_data.sample(n=3)
    
    for idx, row in enumerate(demo_samples.itertuples(), 1):
        features = row[1:5]
        actual = row[5]
        predicted = nb.predict(features)
        
        print(f"\nMẫu {idx}:")
        print(f"   SepalLength={features[0]:.1f}, SepalWidth={features[1]:.1f}")
        print(f"   PetalLength={features[2]:.1f}, PetalWidth={features[3]:.1f}")
        print(f"   Dự đoán: {predicted}")
        print(f"   Thực tế: {actual}")
        print(f"   Kết quả: {'✅ ĐÚNG' if predicted == actual else '❌ SAI'}")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH CÂU 1")
    print("=" * 80)

if __name__ == "__main__":
    main()
