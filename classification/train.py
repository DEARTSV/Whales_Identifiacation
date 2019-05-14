from core.siamese import Siamese

model = Siamese(input_shape=(224, 224, 3), train_hidden_layers=True, n_classes=5004, mode='cosface')
#model = Siamese(input_shape=(224, 224, 3), train_hidden_layers=False, n_classes=2)
#model = Siamese(input_shape=(224, 224, 3), train_hidden_layers=False, n_classes=5004, mode='arcface')
#model.load_weights('model/mobilenet_imagenet.h5')
model.load_weights('trained/final_weights.h5')


model.train('data/train', 'data/train.csv', meta_dir='data/meta', epochs=500, batch_size=5, learning_rate=0.0001)


