from core.siamese import Siamese


model = Siamese('shallow_mnist', input_shape=(28, 28, 3), embedding_size=64, strategy='batch_all')
#model.train('data_mnist/train.csv', 'data_mnist/train', 'C:/Users/Sergei/Desktop/humpback-whale-identification/data_mnist/meta',
#            epochs=10, batch_size=100, learning_rate=0.0001, margin=1.0)

model.train('data_mnist/train', 'data_mnist/train.csv', 'C:/Users/Sergei/Desktop/humpback-whale-identification/data_mnist/meta',
            epochs=10, batch_size=100, learning_rate=0.0001, margin=0.5)
