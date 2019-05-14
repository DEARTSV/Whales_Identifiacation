import sys
sys.path.insert(0, '../')

from core.siamese import Siamese

#model = Siamese(input_shape=(224, 224, 3), train_hidden_layers=False, n_classes=8, mode='classification')
model = Siamese(input_shape=(224, 224, 3), train_hidden_layers=True, n_classes=8, mode='arcface')
#model.load_weights('../trained/final_weights.h5')
model.load_weights('trained/final_weights.h5')
model.train('../data/train', 'train.csv', meta_dir='../data/meta', epochs=70, batch_size=5, learning_rate=0.0001)
