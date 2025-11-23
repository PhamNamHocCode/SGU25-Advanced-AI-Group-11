"""
Script chạy tất cả các bài tập Lab 06 - Naïve Bayes
Chạy tuần tự: Câu 1 (Iris) → Câu 2 (Letter Recognition)
"""

import sys
import time

def print_separator():
    print("\n" + "=" * 80)
    print("=" * 80 + "\n")

def run_cau1():
    """Chạy Câu 1: Iris Classification"""
    try:
        print_separator()
        print("🚀 BẮT ĐẦU CHẠY CÂU 1: IRIS CLASSIFICATION")
        print_separator()
        
        import cau1_iris_demo
        cau1_iris_demo.main()
        
        return True
    except Exception as e:
        print(f"\n❌ LỖI KHI CHẠY CÂU 1: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_cau2():
    """Chạy Câu 2: Letter Recognition"""
    try:
        print_separator()
        print("🚀 BẮT ĐẦU CHẠY CÂU 2: LETTER RECOGNITION")
        print_separator()
        
        import cau2_letter_recognition
        cau2_letter_recognition.main()
        
        return True
    except Exception as e:
        print(f"\n❌ LỖI KHI CHẠY CÂU 2: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("LAB 06 - NAÏVE BAYES CLASSIFIER")
    print("Đại học Sài Gòn (SGU) - Advanced AI")
    print("=" * 80)
    print("\n📋 Danh sách bài tập:")
    print("   1. Câu 1: Phân loại loài hoa Iris (3 classes, 4 features)")
    print("   2. Câu 2: Nhận dạng ký tự A-Z (26 classes, 16 features)")
    print("\n⏳ Đang chạy tất cả bài tập...")
    
    start_time = time.time()
    
    # Chạy Câu 1
    success_cau1 = run_cau1()
    
    # Nghỉ 2 giây giữa 2 câu
    time.sleep(2)
    
    # Chạy Câu 2
    success_cau2 = run_cau2()
    
    # Tổng kết
    end_time = time.time()
    elapsed = end_time - start_time
    
    print_separator()
    print("📊 TỔNG KẾT")
    print("=" * 80)
    print(f"\n⏱️  Tổng thời gian chạy: {elapsed:.2f} giây")
    print(f"\n✅ Câu 1 (Iris): {'THÀNH CÔNG' if success_cau1 else 'THẤT BẠI'}")
    print(f"✅ Câu 2 (Letter Recognition): {'THÀNH CÔNG' if success_cau2 else 'THẤT BẠI'}")
    
    if success_cau1 and success_cau2:
        print("\n🎉 ĐÃ HOÀN THÀNH TẤT CẢ BÀI TẬP!")
        print("=" * 80)
        return 0
    else:
        print("\n⚠️  CÓ MỘT SỐ BÀI TẬP THẤT BẠI")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
