import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data1 = pd.read_csv("../data1/healthcare-dataset-stroke-data.csv")
df = pd.DataFrame(data1)

# Select features and target
x_features = ["gender", "age", "hypertension", "heart_disease", "work_type",
              "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
y = df["stroke"]
X = df[x_features]

# Print first few rows to verify data
print("Target variable (first 5 rows):")
print(y.head())
print("\nFeatures (first 5 rows):")
print(X.head().to_string())

# Encode categorical variables
s = (X.dtypes == "object")
obj_cols = list(s[s].index)
encoder = OrdinalEncoder()
X[obj_cols] = encoder.fit_transform(X[obj_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create the model
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(9,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'\nTest Accuracy: {test_accuracy:.4f}')

y_pred = model.predict(X_test_scaled)
print("\nA few of the predictions (prob of stroke):")
print(y_pred[:5])




plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


"""# Function to make predictions for new patients
def predict_stroke_risk(patient_data):
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data])

    # Encode categorical variables
    patient_df[obj_cols] = encoder.transform(patient_df[obj_cols])

    # Scale features
    patient_scaled = scaler.transform(patient_df)

    # Make prediction
    prediction = model.predict(patient_scaled)[0][0]
    return prediction


# Example usage
sample_patient = {
    "gender": "Male",
    "age": 65,
    "hypertension": 1,
    "heart_disease": 1,
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 200,
    "bmi": 28,
    "smoking_status": "formerly smoked"
}

risk = predict_stroke_risk(sample_patient)
print(f"\nStroke risk for sample patient: {risk:.2%}")"""