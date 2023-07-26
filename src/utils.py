import os
import shutil
import numpy as np

from PIL import Image
from itertools import groupby
from operator import itemgetter


def load_train(path):
    """Returns loaded images that will be used in similarity analysis.

    Args:
        path (str): path to folder directory containing the images.
    """
    # Directories
    dir_ = []
    for x in os.walk(path):
        try:
            dir_.append(x[0].split('\\')[1])
        except Exception as _:
            pass

    # Loop through folders.
    files_in_folders, files_in_target_folder = [], []
    for i, folder in enumerate(dir_):
        folder_path = os.path.join(path, folder)

        # Collect data names in current folder.
        files = []
        for file in os.walk(folder_path):
            files.append(file)
            files = [os.path.join(folder_path, name) for name in files[0][2]]

            # Make sure each image file has correct format.
            cnt = 0
            for j, f in enumerate(files):
                formt = f.split('\\')[-1].split('.')[-1]
                if formt not in ['png', 'jpg', 'PNG', 'JPG']:
                    files.pop(cnt)

                cnt += 1
                msg = f"processed (getting data): {i + 1}/{len(dir_)}, file {j + 1}/{len(files)}"
                print(msg, end="\r")

        # Group images given respective index.
        ints = [int(name.split('_')[-1].split('.')[0]) for name in files]
        ints = [[idx, int_] for idx, int_ in zip(range(len(ints)), ints)]
        idxs = sorted(ints, key=itemgetter(1))
        idxs = np.array(idxs)[:, 0]
        files_ = [files[i] for i in idxs]
        files_ = [list(i) for j, i in groupby(files_, lambda a: a.split('_')[-1].split('.')[0])]
        files_in_folders.append(files_)

    # Flatten list of images.
    images = [e for sub in files_in_folders for e in sub]
    images = [e for sub in images for e in sub]

    return images, dir_


def remove_transparency(x, bg_colour=(255, 255, 255)):
    """Returns x image without alpha channel. Only process if
    image has transparency.

    Args:
        x: x image.
        bg_colour (tuple): background color.
    """
    if x.mode in ('RGBA', 'LA') or (x.mode == 'P' and 'transparency' in x.info):
        # Convert to RGBA if LA format due to a bug in PIL (see http://stackoverflow.com/a/1963146).
        alpha = x.convert('RGBA').split()[-1]

        # Create a new background image of our matt color. It must be RGBA
        # because 'paste' requires both images have the same format.
        # See: http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208.
        bg = Image.new("RGBA", x.size, bg_colour + (255,))
        bg.paste(x, mask=alpha)
        return bg.convert('RGB')
    else:
        # Returns image if no alpha channel.
        return x


def resize_array(im, size, resample=True):
    """Returns array resize to defined size.

    Args:
        im: x image to resize.
        size (int or float): finql size of image.
        resample (bool): if True, resize with resampling.
    """
    # Convert to PIL onbject.
    img = Image.fromarray(im)
    # Resize image.
    if type(size) == int:
        img = img.resize((size, size), resample)
    elif (type(size) == int) or (type(size) == float):
        img = img.resize(size, resample)
    return np.array(img)


def get_array_from_image(img, img_size, normalize=True):
    """Returns array of x image.

    Args:
        img: x image.
        img_size (int): fina size of image.
        normalize (bool): if True, normalize pixel values of
            image.
    """
    # Remove transparency from image (if existing).
    img = remove_transparency(Image.open(img))

    # Resize and convert to array.
    img = img.resize((img_size, img_size))
    img = np.array(img).astype(np.float32)

    # Normalize values of image pixels.
    if normalize is True:
        if len(img.shape) == 3:
            # if 3 bands
            img = np.multiply(img, 1.0 / 255.0)
        else:
            # if 1 band
            img = ((img - np.min(img[img != 0])) / (np.max(img) - np.min(img[img != 0]))).clip(min=0)
        img = np.array(img)

    return img


def check_directory(directory, show=True):
    """Checks if directory exists, otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        if show is True:
            print('New folder created: {}'.format(directory))


def check_directory_and_replace(directory, show=True):
    """Checks if directory exists, then replace it with new one
    for saving new content."""
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)  # delete directory and its content
        if show is True:
            print('Folder {} and its content removed.'.format(directory))

    # Create new directory.
    check_directory(directory, show=False)


def weighting(x):
    """Returns normalized x."""
    if type(x) == list:
        x = np.array(x)
    return x / np.sum(x)


def clear_txt_file(path, cls, name):
    """Returns empty text file."""
    file_name = path + '/' + '{}_'.format(cls) + name + '.txt'
    _ = open(file_name, 'w+')
    return file_name
