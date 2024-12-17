import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    # TODO: เขียนฟังก์ชัน activation function แบบ step function
    def step_function(self, x):
         
    
    # TODO: เขียนฟังก์ชันสำหรับการ train model
    def fit(self, X, y):
        # รับข้อมูล input X และ target y
        n_samples, n_features = X.shape
        
        # สร้าง weights และ bias เริ่มต้นด้วยค่าสุ่ม
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # วนรอบการเรียนรู้
        
        
    # TODO: เขียนฟังก์ชันสำหรับการทำนายผลลัพธ์
    def predict(self, X):
        # เขียนโค้ดตรงนี้
        

# ตัวอย่างการใช้งาน: ข้อมูลการตัดสินใจซื้อบ้าน
# Features: [รายได้(ล้านบาท/ปี), มีครอบครัวหรือไม่(0,1)]
X = np.array([[2.5, 0], [3.2, 1], [1.8, 0], [4.5, 1]])
y = np.array([0, 1, 0, 1])  # 0=ไม่ซื้อ, 1=ซื้อ

model = Perceptron()
model.fit(X, y)
predictions = model.predict(X)
print("ผลการทำนาย:", predictions)
print("Weights:", model.weights)
print("Bias:", model.bias)
