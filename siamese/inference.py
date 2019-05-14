from core.siamese import Siamese

model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128)
model.load_weights('trained/final_weights.h5')

#model.make_embeddings('data/train', 'data/train.csv', batch_size=5, meta_dir='data/meta')
model.load_embeddings('trained/embeddings.pkl')

model.predict('data/test', meta_dir='data/meta')
#model.load_predictions('trained/predictions.pkl')

model.make_kaggle_csv('data/meta/idx_to_whales_mapping.npy')
