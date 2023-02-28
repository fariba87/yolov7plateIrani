import numpy as np
import cv2
def crop(image, coord):
    cropped_image = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    return cropped_image
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=2.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
def pascal_voc_to_coco(x1y1x2y2):
    x1, y1, x2, y2 = x1y1x2y2
    return [x1, y1, x2 - x1, y2 - y1]

def get_best_ocr(preds, rec_conf, ocr_res, track_id):
    for info in preds:
        # Check if it is current track id
        if info['track_id'] == track_id:
            # Check if the ocr confidenence is maximum or not
            if info['ocr_conf'] < rec_conf:
                info['ocr_conf'] = rec_conf
                info['ocr_txt'] = ocr_res
            else:
                rec_conf = info['ocr_conf']
                ocr_res = info['ocr_txt']
            break
    return preds, rec_conf, ocr_res

def applyNoise (plate):
    background = plate.convert("RGBA")
    noisyTemplates = []
    for noise in noises:
        newPlate = Image.new('RGBA', (600,132), (0, 0, 0, 0))
        newPlate.paste(background, (0,0))
        noise = Image.open(os.path.join('Noises/', noise)).convert("RGBA")
        newPlate.paste(noise, (0, 0), mask=noise)
        noisyTemplates.append(newPlate)
    return noisyTemplates


# Generate Transformations of plates
def applyTransforms(plate):
    transformedTemplates = []
    plate = np.array(plate)

    # Rotating to clockwise
    for _ in range(3):
        result = imutils.rotate_bound(plate, random.randint(2, 7))
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # Rotating to anticlockwise
    for _ in range(3):
        result = imutils.rotate_bound(plate, random.randint(-7, -2))
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # Scaling up
    for _ in range(3):
        height, width, _ = plate.shape
        randScale = random.uniform(1.1, 1.3)
        result = cv2.resize(plate, None, fx=randScale, fy=randScale, interpolation=cv2.INTER_CUBIC)
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # Scaling down
    for _ in range(3):
        height, width, _ = plate.shape
        randScale = random.uniform(0.2, 0.6)
        result = cv2.resize(plate, None, fx=randScale, fy=randScale, interpolation=cv2.INTER_CUBIC)
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # # Adding perspective transformations
    # for _ in range(3):
    #     rows,cols,ch = plate.shape
    #     background = Image.fromarray(np.zeros(cols + 100, rows + 100, 3))
    #     pts1 = np.float32([[50,50],[200,50],[50,200]])
    #     pts2 = np.float32([[10,100],[200,50],[100,250]])
    #     M = cv2.getAffineTransform(pts1,pts2)
    #     result = cv2.warpAffine(plate,M,(cols,rows))
    #     result = Image.fromarray(result)
    #     transformedTemplates.append(result)

    return transformedTemplates
def plateFromName(nameStr):
    numbers = []
    letters = []

    for char in nameStr:
        if char.isnumeric():
            numbers.append(char)
        else:
            letters.append(char)

    return [*numbers[:2], ''.join(letters), *numbers[2:]]
def getPlateName(n1, n2, l, n3, n4, n5):
    return f'{n1}{n2}{l}{n3}{n4}{n5}'
import random
def getNewPlate(letters, numbers):
    return [random.choice(letters),  # numbers[5],
            random.choice(letters),  # numbers[6],
            random.choice(letters),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers)]
# Generateplate array from string
# (37GAF853 -> ['3', '7', 'GAF', '8', '5', '3'])
# def plateFromName(nameStr):
#     numbers = []
#     letters = []
#
#     for char in nameStr:
#         if char.isnumeric():
#             numbers.append(char)
#         else:
#             letters.append(char)
#
#     return [*numbers[:2], ''.join(letters), *numbers[2:]]
def transform_data(m):
    dpix, dpiy = m.shape
    x_c, y_c = np.unravel_index(np.argmax(m), m.shape)
    angles = np.linspace(0, np.pi*2, min(dpix, dpiy))
    mrc = min(abs(x_c - dpix), abs(y_c - dpiy), x_c, y_c)
    radiuses = np.linspace(0, mrc, max(dpix, dpiy))
    A, R = np.meshgrid(angles, radiuses)
    X = R * np.cos(A)
    Y = R * np.sin(A)
    return A, R, m[X.astype(int) + mrc - 1, Y.astype(int) + mrc - 1]
def align_by_fft(img):
    fft = np.fft.fft2(img)
    max_peak = np.max(np.abs(fft))
    fft[fft < (max_peak * 0.25)] = 0
    abs_data = 1 + np.abs(fft)
    c = 255.0 / np.log(1 + max_peak)
    log_data = c * np.log(abs_data)
    max_scaled_peak = np.max(log_data)
    rows, cols = np.where(log_data > (max_scaled_peak * 0.90))
    min_col, max_col = np.min(cols), np.max(cols)
    min_row, max_row = np.min(rows), np.max(rows)
    dy, dx = max_col - min_col, max_row - min_row
    theta = np.arctan(dy / float(dx))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))


    angles, radiuses, m = transform_data(magnitude_spectrum)
    c=m
    sample_angles = np.linspace(0,  2 * np.pi, len(c.sum(axis=0))) / np.pi*180
    turn_angle_in_degrees = 90 - sample_angles[np.argmax(c.sum(axis=0))]
    return turn_angle_in_degrees