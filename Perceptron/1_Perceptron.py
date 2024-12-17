import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def step_function(self, x):
        # เขียนโค้ดฟังก์ชันกระตุ้นแบบขั้นบันได
        
    
    def fit(self, X, y):
        # รับข้อมูล input X และ target y
        n_samples, n_features = X.shape
        
        # สร้าง weights และ bias 
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # วนรอบการเรียนรู้
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # เขียนโค้ดคำนวณ linear regression
                 
              
                # เขียนโค้ดเรียกใช้ activation function (Step function)
                
                
                # เขียนโค้ดปรับค่า weights และ bias ตาม perceptron learning rule
                 
                
    def predict(self, X):
        # คำนวณ output สำหรับ input X
        

# ทดสอบกับ AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

model = Perceptron(learning_rate=0.1, n_iterations=100)
model.fit(X_and, y_and)
predictions = model.predict(X_and)
print("AND Gate ผลการทำนาย:", predictions)
print("AND Gate Weights:", model.weights)
print("AND Gate Bias:", model.bias)
