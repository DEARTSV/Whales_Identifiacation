from core.siamese import Siamese

model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128)
model.load_weights('model/mobilenet_imagenet.h5')
#model.load_weights('trained/final_weights.h5')

#model.train('data/train', 'data/split_train.csv', meta_dir='D:\\IdeaProjects\\whales\\data\\meta', epochs=20, batch_size=24, learning_rate=0.001)
model.train('data/train', 'data/train.csv', meta_dir='data/meta', epochs=500, batch_size=25, learning_rate=0.0005, margin=1.0)
