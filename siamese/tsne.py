from core.siamese import Siamese


model = Siamese('resnet_like_33', input_shape=(384, 512, 3), embedding_size=128, strategy='batch_all')
model.load_weights('trained/checkpoint-05.h5')
model.load_embeddings('trained/embeddings.pkl')
model.load_predictions('trained/predictions.pkl')

model.draw_tsne(model.predictions.values[:, 1:])


