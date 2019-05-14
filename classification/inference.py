from core.siamese import Siamese

# using softmax loss classification
# model = Siamese(input_shape=(224, 224, 3), n_classes=5004)
# model.load_weights('trained/final_weights.h5')
# model.predict('data/test')
# model.make_kaggle_csv('data/meta/idx_to_whales_mapping.npy')



# using cos_angular embeddings
model = Siamese(input_shape=(224, 224, 3), n_classes=5004, mode='cosface', train=False)
model.load_weights('trained/final_weights.h5')

model.make_embeddings('data/train', 'data/train.csv', batch_size=25, meta_dir='data/meta')
#model.load_embeddings('trained/embeddings.pkl')

model.predict_using_embeddings('data/test', meta_dir='data/meta')
#model.load_predictions('trained/predictions.pkl')

model.make_kaggle_csv('data/meta/idx_to_whales_mapping.npy')
