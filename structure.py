from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('D:\Project\Healthcure\models\Brain_Tumor_model.h5')

# Print the summary of the model
model.summary()
