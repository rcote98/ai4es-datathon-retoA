import rasterio
import numpy as np
import pandas as pd


def load_image(image_path, remove_negs=True, normalization=False):
    # Load the image as a numpy array
    img = rasterio.open(image_path)
    concat_list = []
    if img.count == 5: # 1=B, 2=G, 3=R, 4=RE, 5=NIR
        channels_list = [1,2,3,4,5]
    elif img.count == 10: # 1=B, 3=G, 5=R, 7=RE, 10=NIR
        channels_list = [1,3,5,7,10]
    else:
        raise Exception("Unexpected number of channels in image %s" %image_path)
    for i in channels_list:
        ch_image = img.read(i)
        if remove_negs:
            ch_image[ch_image < 0] = 0.0
        concat_list.append(np.expand_dims(ch_image, -1))
    image = np.concatenate(concat_list, axis=-1).astype(float) # float 0-1
    if normalization:
        max_pixel_value = np.max(image, axis=2)
        max_pixel_value = np.repeat(max_pixel_value[:, :, np.newaxis], 5, axis=2)
        image = np.divide(image, max_pixel_value)
        image = image[:, :, 0:4]
        image[np.isnan(image)] = 0.0

    return image

def predict(image):
    import random
    return [random.randrange(0, 101), random.randrange(0, 101), random.randrange(0, 101)]


if __name__=="__main__":
    test_df = pd.read_csv("test.csv")
    for image_path in test_df["PLOT_FILE"]:
        image = load_image(image_path=image_path, remove_negs=True, normalization=True)
        prediction = predict(image=image)
        test_df.loc[test_df.PLOT_FILE == image_path, ['DISEASE1', 'DISEASE2', 'DISEASE3']] = prediction
    participant = "John_Smith"
    version = "v1"
    results_filename = '-'.join([participant, 'test_results', version])
    test_df.to_csv(results_filename, index=False)

