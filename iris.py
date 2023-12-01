import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Iris veri setini yükleyelim
file_path = r'C:\Users\seytim13\Desktop\Iris\iris.csv'
df = pd.read_csv(file_path)

# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['species'])


# Veriyi özellikler (X) ve etiketler (y) olarak ayırma
X = df.drop('species', axis=1)
y = label_encoder.fit_transform(df['species'])


# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Veriyi standartlaştıralım
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Yapay sinir ağı modelini oluşturalım
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Modeli derleyelim
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitelim
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

# Modelin performansını değerlendirelim
accuracy = model.evaluate(X_test, y_test, verbose=2)[1]
print(f"Model Doğruluk Oranı: {accuracy}")
