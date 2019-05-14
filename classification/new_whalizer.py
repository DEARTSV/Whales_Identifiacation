from core.siamese import Siamese

model = Siamese(input_shape=(224, 224, 3), n_classes=5004)
model.load_weights('data/new_whalizer.h5')

model.predict_new_whales('data/test', 'data/submission_no_new_whales.csv')
