import numpy as np

class ImageProperties:

    def __init__(self, py_type):
        self.py_type = py_type
        self.numpy_dtype = None
        self.shape = None
        self.min_val = None
        self.max_val = None
        self.bins = None
        self.hist = None
        self.num_unique = None

    def __str__(self):
        err = 'Not recognized'
        str_data = f'Type:\t\t{self.py_type}\n'
        if self.numpy_dtype is None:
            str_data += f'Numpy type:\t{err}\n'
        else:
            str_data += f'Numpy type:\t{self.numpy_dtype}\n'

        if self.shape is None:
            str_data += f'Shape:\t\t{err}\n'
        else:
            str_data += f'Shape:\t\t{self.shape}\n'

        if self.min_val is None:
            str_data += f'Minimum:\t{err}\n'
        else:
            str_data += f'Minimum:\t{self.min_val}\n'

        if self.max_val is None:
            str_data += f'Maximum:\t{err}\n'
        else:
            str_data += f'Maximum:\t{self.max_val}\n'

        if self.bins is None:
            str_data += f'Bins:\t\t{err}\n'
        else:
            str_data += f'Bins:\t\t{self.bins}\n'

        if self.hist is None:
            str_data += f'Occurrence:\t{err}\n'
        else:
            str_data += f'Occurrence:\t{self.hist}\n'

        if self.num_unique is None:
            str_data += f'#Unique:\t{err}\n'
        else:
            str_data += f'#Unique:\t{self.num_unique}\n'

        return str_data

def props(img, name=''):

    img_type = type(img)
    imgp = ImageProperties(img_type)

    # Convert to numpy
    try:
        img = np.array(img)
        converted = True
    except Exception:
        converted = False

    if converted:
        dtype = img.dtype
        shape = img.shape
        imgp.numpy_dtype = dtype
        imgp.shape = shape
        # Check if array contains objects
        if not dtype.hasobject:
            min_val, max_val = np.min(img), np.max(img)
            imgp.min_val = min_val
            imgp.max_val = max_val

            hist, bins = np.histogram(img, bins=5)
            imgp.bins = bins
            imgp.hist = hist

            unique = np.unique(img)
            imgp.num_unique = len(unique)

            warnings = _checks(img)

    print(f'{name} properties:')
    print(imgp, end='')
    if warnings=='':
        print('')   # Blank line
    else:
        print(warnings)

def _checks(img):

    shape = img.shape
    ndim = img.ndim
    min_size = min(shape)
    msg = ''
    if min_size==1:
        msg = 'Warning, image has one dimension with size 1.\n'
    elif min_size==2:
        pass
    elif (min_size==3 or min_size==4) and ndim>=3:
        if shape[-1]==min_size:
            c01_close = np.allclose(img[...,0], img[...,1])
            c02_close = np.allclose(img[...,0], img[...,2])
        elif shape[0]==min_size:
            c01_close = np.allclose(img[0], img[1])
            c02_close = np.allclose(img[0], img[2])
        if c01_close and c02_close:
            msg = 'Warning, image has more than one channel but appears to be grayscale.\n'
        if min_size==4:
            msg += 'Warning, image has an alpha channel\n'

    return msg