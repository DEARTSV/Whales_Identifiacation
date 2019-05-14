import pandas as pd


if __name__ == '__main__':
    whales = pd.read_csv('data_mnist/train.csv')
    pred = pd.read_csv('cache/cache-190220-203237/prediction.csv')

    total = len(pred)
    counter = 0

    for idx, p in pred.iterrows():
        res = whales.loc[whales['Image'] == p['Image']]
        if res['Id'].values[0] == p['Id']:
            counter += 1
        print('%d/%d' % (idx+1, total))

    print(counter / total)






