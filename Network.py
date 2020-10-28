 # Network Start----------------------------------------------


## Yapay Sinir Ağı Python Class'ı (Python Sınıfı)
class Network: 
    
    
    ## Ağırlıklarımızı ve bias değerlerimizi burada oluşturulmaktadır.
    def __init__(self):
        
        # Ağ üzerinden 3 adet nöron olduğu için 
        # 6 adet ağırlık ve 3 adet bias değeri olmalı
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
     
    ## Sigmoid fonksiyonu 
    def sigmoid(self , x): 
        
        # Sigmoid aktivasyon fonksiyonu : f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))
    
    
    ## Sigmoid fonksiyonunun türevi
    def sigmoid_turev(self , x):
        
        # Sigmoid fonksiyonunun türevi: f'(x) = f(x) * (1 - f(x))
        sig    = self.sigmoid(x)
        result = sig * (1 - sig)
        
        
        return result
            
    def mse_loss(self , y_real , y_prediction):
        
        # y_real ve y_prediction aynı boyutta numpy arrayleri olmalıdır. 
        return ((y_real - y_prediction) ** 2).mean()
    
    ## İleri beslemeli nöronlar üzerinden tahmin
    ## değerinin elde edilmesi 
    
    def feedforward(self , row):
        
        # h1 nöronunun değeri
        h1 = self.sigmoid((self.w1 * row[0]) + (self.w2 * row[1]) + self.b1 )
        
        # h2 nöronunun değeri
        h2 = self.sigmoid((self.w3 * row[0]) + (self.w4 * row[1]) + self.b2 )
        
        # Tahmin değeri 01 nöronun değeri
        o1 = self.sigmoid((self.w5 * h1 ) + (self.w6 * h2 ) + self.b3 )
        
        return o1 
    
    

        
    
    ## Belitiler iteresyon sayısı kadar model eğitimi
    def train(self , data , labels):
    
        learning_rate = 0.001
        epochs = 10000
        
        for epoch in range(epochs):
            
            for x, y in zip(data , labels):
                    
                # Neuron H1
                sumH1 = (self.w1 * x[0]) + (self.w2 * x[1]) + self.b1 
                H1    = self.sigmoid(sumH1)
                
                # Neuron H2
                sumH2 = (self.w3 * x[0]) + (self.w4 * x[1]) + self.b2
                H2    = self.sigmoid(sumH2)
                
                # Neuron O1
                sumO1 = (self.w5 * H1) + (self.w6 * H2) + self.b3
                O1    = self.sigmoid(sumO1)
                
                # Tahmin değerimiz
                prediction = O1
                
                
                # Türevlerin Hesaplanması 
                # dL/dYpred :  y = doğru değer | prediciton: tahmin değeri
                dLoss_dPrediction = -2*(y - prediction)
                
                # Nöron H1 için ağırlık ve bias türevleri 
                dH1_dW1 = x[0] * self.sigmoid_turev(sumH1)
                dH1_dW2 = x[1] * self.sigmoid_turev(sumH1)
                dH1_dB1 = self.sigmoid_turev(sumH1)
                
                
                # Nöron H2 için ağırlık ve bias türevleri
                dH2_dW3 = x[0] * self.sigmoid_turev(sumH2)
                dH2_dW4 = x[1] * self.sigmoid_turev(sumH2)
                dH2_dB2 = self.sigmoid_turev(sumH2)
                
                # Nöron O1 (output) için ağırlık ve bias türevleri
                dPrediction_dW5 = H1 * self.sigmoid_turev(sumO1) 
                dPrediction_dW6 = H2 * self.sigmoid_turev(sumO1) 
                dPrediction_dB3 = self.sigmoid_turev(sumO1) 
                
                # Aynı zamanda tahmin değerinin H1 ve H2'ye göre türevlerinin de
                # hesaplanması gerekmektedir. 
                dPrediction_dH1 = self.w5 * self.sigmoid_turev(sumO1)
                dPrediction_dH2 = self.w6 * self.sigmoid_turev(sumO1)
                
                ## Ağırlık ve biasların güncellenmesi 
                
                # H1 nöronu için güncelleme
                self.w1 = self.w1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW1)
                self.w2 = self.w2 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW2)
                self.b1 = self.b1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dB1)
                
                # H2 nöronu için güncelleme 
                self.w3 = self.w3 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW3)
                self.w4 = self.w4 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW4)
                self.b2 = self.b2 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dB2)
                
                # O1 nöronu için güncelleme 
                self.w5 = self.w5 - (learning_rate * dLoss_dPrediction * dPrediction_dW5) 
                self.w6 = self.w6 - (learning_rate * dLoss_dPrediction * dPrediction_dW6) 
                self.b3 = self.b3 - (learning_rate * dLoss_dPrediction * dPrediction_dB3) 
                
            predictions = np.apply_along_axis(self.feedforward ,1, data)
            loss = self.mse_loss(labels , predictions)
            print("Devir %d loss: %.7f" % (epoch, loss))
            
                     
        
 # Network Ends----------------------------------------------


data = np.array([ #Boy #Kilo
                  [5,    4 ],  # Ahmet
                  [15,   5 ],  # Veli
                  [-6,  -26],  # Nazan 
                  [-24, -15],  # Elif
                  [-15, -6 ],  # Kamil 
                  [7,  -15]    # Suzan
])
labels = np.array([
  0, # Ahmet
  0, # Veli
  1, # Nazan
  1, # Elif
  0, # Kamil
  1, # Suzan
])

network = Network()
network.train(data, labels)

X = [20 , 9] # 20 + 160 = 180 Boy , 9 + 71 = 80 Kilo  

prediction = network.feedforward(Osman)

if(prediction > 0.5): 
    print('Cinsiyet : Kadın  |' , 'Value : ' , prediction)
else: 
    print('Cinsiyet : Erkek  |' , 'Value : ' , prediction)
