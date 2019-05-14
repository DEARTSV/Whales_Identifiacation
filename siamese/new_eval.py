from core.siamese import Siamese

model = Siamese.restore_from_config('cache/cache-190226-111833/config.json')
model.make_embeddings('data/train', 'data/train.csv', batch_size=24)

# model.predict('data/train')
# model.make_csv('cache/cache-190225-233850/idx_to_whales_mapping.npy')

model.predict('data/test')
model.make_kaggle_csv('data/meta/idx_to_whales_mapping.npy')



