import numpy as np

def accuracy_evaluate(model, test_data):
    """
    Default model: f(x)
    Default test_data = [(data, label), (data, label),...]
    return accuracy
    
    Ưu điểm: Công thức tường minh, dễ diễn giải ý nghĩa
    
    Nhược điểm: Tính toán đo lường trên tất cả các nhãn chứ không tập trung quan tâm đến độ chính xác của một nhãn riêng lẻ nào cả
    
    Ví dụ: Trong bài toán phân loại hồ sơ bệnh án, giả sử ta cần quan tâm đặc biệt đến việc phân loại đúng những bệnh nhân bị ung thư thay vì
    quan tâm đến việc phân loại các hồ sơ bệnh án thông thường.
    """
    check_correct = np.array([1 if model(data) == label else 0 for data, label in test_data]).sum()
    return float(check_correct)/test_data.shape[0]

