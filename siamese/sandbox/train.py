import sys
sys.path.insert(0, '../')

from core.siamese import Siamese

model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128, train_hidden_layers=True)
#model.load_weights('../trained/final_weights.h5')
model.load_weights('trained/final_weights.h5')
model.train('../data/train', 'train.csv', meta_dir='../data/meta', epochs=40, batch_size=20, learning_rate=0.001, margin=1.0)
