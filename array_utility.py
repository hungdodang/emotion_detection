def scale_array(array):
    """
    Convert values in range [0:255] in array into [-1:1]
    :param array: input array
    :return: scaled array
    """
    array = array.astype('float32')
    array = ((array / 255.0) - 0.5) * 2.0
    return array
