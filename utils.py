import numpy as np


def calc_angle(first_vector, second_vector):
    len1 = np.sqrt(np.sum(np.array(first_vector) ** 2))
    len2 = np.sqrt(np.sum(np.array(second_vector) ** 2))

    first_vector[0] /= len1
    first_vector[1] /= len1

    second_vector[0] /= len2
    second_vector[1] /= len2

    dot_product = first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1]
    len_product = np.sqrt(np.sum(np.array(first_vector) ** 2)) * np.sqrt(np.sum(np.array(second_vector) ** 2))
    rad = np.arccos(dot_product / len_product)
    deg = (rad / np.pi) * 180.0
    if (first_vector[0] + second_vector[0] <= 0.0) and (first_vector[1] + second_vector[1] >= 0.0):
        return deg, True
    else:
        return deg, False
