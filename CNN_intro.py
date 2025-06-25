import numpy as np


def convolve2d(image, kernel):
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape

    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1

    output = np.zeros((o_height, o_width))
    for y in range(o_height):
        for x in range(o_width):
            output[y, x] = np.sum(image[y:y + k_height, x:x + k_width] * kernel)
    return output
image = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])
kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])
output = convolve2d(image, kernel)



print("Input test Img):")
print(image)
print("\nKernel:")
print(kernel)
print("\nOutput after convolution:")
print(output)