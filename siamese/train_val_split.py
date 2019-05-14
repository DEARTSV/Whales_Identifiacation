import pandas as pd

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    train = train.sample(frac=1).reset_index(drop=True)  # shuffle rows

    num_new_whales = 500
    new_whale_imgs = train.loc[train['Id'] == 'new_whale']
    val_part = new_whale_imgs[:num_new_whales]
    val_imgs = val_part['Image'].tolist()
    val_labels = ['new_whale' for _ in range(num_new_whales)]
    train.drop(val_part.index, inplace=True)

    ids = train['Id'].unique()
    for whale_id in ids:
        group_imgs = train.loc[train['Id'] == whale_id]
        if 10 <= len(group_imgs):
            if whale_id == 'new_whale':
                continue
            val_part = group_imgs[:2]
            val_imgs += val_part['Image'].tolist()
            val_labels += [whale_id, whale_id]
            train.drop(val_part.index, inplace=True)
        elif 5 <= len(group_imgs):
            val_part = group_imgs[:1]
            val_imgs += val_part['Image'].tolist()
            val_labels += [whale_id]
            train.drop(val_part.index, inplace=True)

    val = pd.DataFrame({'Image': val_imgs, 'Id': val_labels})

    train.to_csv('data/split_train.csv', index=False)
    val.to_csv('data/split_val.csv', index=False)

    print('train images: ', len(train))
    print('val images: ', len(val))
