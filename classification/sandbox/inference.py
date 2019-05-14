# inference to get cos angle embeddings

import sys
sys.path.insert(0, '../')

from core.siamese import Siamese

model = Siamese(input_shape=(224, 224, 3), n_classes=8, mode='arcface', train=False)
model.load_weights('trained/final_weights.h5')

model.make_embeddings('../data/train', 'train.csv', mappings_filename='../data/meta/whales_to_idx_mapping.npy', batch_size=25)
#model.load_embeddings('trained/embeddings.pkl')

model.predict_using_embeddings('../data/train', 'val.csv')
#model.predict_using_embeddings('../data/train', 'train.csv')
