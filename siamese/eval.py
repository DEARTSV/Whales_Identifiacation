from core.siamese import Siamese


model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128, strategy='batch_hard')
model.load_weights('cache/cache-190213-095205/training/checkpoint-07.h5')
model.load_embeddings('cache/cache-190213-113426/embeddings.pkl')
# model.make_embeddings('data/train', 'data/train.csv', batch_size=32)
# model.predict('data/test')
model.predict('data/train_subset')
model.make_csv('cache/cache-190208-093118/idx_to_whales_mapping.npy')





