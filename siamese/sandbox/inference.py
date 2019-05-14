import sys
sys.path.insert(0, '../')

from core.siamese import Siamese

model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128)
model.load_weights('trained/final_weights.h5')

model.make_embeddings('../data/train', 'train.csv', mappings_filename='../data/meta/whales_to_idx_mapping.npy', batch_size=25)
#model.load_embeddings('trained/embeddings.pkl')

model.predict('../data/train', 'val.csv')
#model.predict('../data/train', 'train.csv')
