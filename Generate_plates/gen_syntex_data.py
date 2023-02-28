from PIL import Image
import os
import random
import numpy as np
import cv2
import imutils
import csv
import time
from tqdm import tqdm
numbers = [str(i) for i in range(0, 10)]
letters = ['ALEF', 'BE', 'PE', 'TE', 'SE', 'JIM', 'CHE', 'HE', 'KHE', 'DAL', 'ZAL', 'RE', 'ZE', 'ZHE', 'SIN','SHIN', 'SAD', 'ZAD', 'TA', 'ZA', 'EIN', 'GHEIN', 'FE', 'GHAF', 'KAF', 'GAF', 'LAM', 'MIM', 'NON', 'VAV', 'HA', 'YE']
import numpy as np
import cv2
from PIL import Image
import os
import imutils



def applyNoise (plate,noises):
    background = plate.convert("RGBA")
    noisyTemplates = []
  #  for noise in noises:
    newPlate = Image.new('RGBA', (600,132), (0, 0, 0, 0))
    newPlate.paste(background, (0,0))
    noise = Image.open(os.path.join('Noises/', noises)).convert("RGBA")
    newPlate.paste(noise, (0, 0), mask=noise)
    noisyTemplates.append(newPlate)
    return noisyTemplates


# Generate Transformations of plates
def applyTransforms(plate):
    transformedTemplates = []
    plate = np.array(plate)

    # Rotating to clockwise
    for _ in range(1):
        result = imutils.rotate_bound(plate, random.randint(2, 7))
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # Rotating to anticlockwise
    for _ in range(1):
        result = imutils.rotate_bound(plate, random.randint(-7, -2))
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # Scaling up
    for _ in range(1):
        height, width, _ = plate.shape
        randScale = random.uniform(1.1, 1.3)
        result = cv2.resize(plate, None, fx=randScale, fy=randScale, interpolation=cv2.INTER_CUBIC)
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # Scaling down
    for _ in range(1):
        height, width, _ = plate.shape
        randScale = random.uniform(0.2, 0.6)
        result = cv2.resize(plate, None, fx=randScale, fy=randScale, interpolation=cv2.INTER_CUBIC)
        result = Image.fromarray(result)
        transformedTemplates.append(result)
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
def getPlateName(n1, n2, l, n3, n4, n5,n6,n7):
    return f'{n1}{n2}{l}{n3}{n4}{n5}{n6}{n7}'
import random
# def getNewPlate(letters, numbers):
#     return [random.choice(letters),  # numbers[5],
#             random.choice(letters),  # numbers[6],
#             random.choice(letters),
#             random.choice(numbers),
#             random.choice(numbers),
#             random.choice(numbers)]
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
fonts = ['roya_bold']
templates = [os.path.basename(os.path.splitext(template)[0]) for template in os.listdir('Templates') if template.endswith('.png') and template not in ['tashrifat.png', 'template-sepah.png', 'template-police.png']]
# Noises
noises = os.listdir('Noises')
#noises=noises[np.random.randint(0,len(noises))]
# transformations
transformations = [ 'zoom_in']
#transformations = []
# Count of permutations
permutations = 3




# Returns a plate as a string
# def getPlateName(n1, n2, l, n3, n4, n5):
#     return f'{n1}{n2}{l}{n3}{n4}{n5}'


# Returns Address of a glyph image given font, and glyph name
def getGlyphAddress(font, glyphName):
    return f'Glyphs/{font}/{glyphName}_trim.png'


# Returns an array containing a plate's letter and numbers:
# [number1, number2 , letter, number3, number4, number5]
# return plateFromName('37GAF853')

def getNewPlate():
    return [ random.choice(numbers),  # numbers[5],
            random.choice(numbers),  # numbers[6],
            random.choice(letters1),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers),
             random.choice(numbers), random.choice(numbers)
             ]
idCounter = 0
fontsProgBar = tqdm(total=len(fonts)*len(templates)*permutations*len(noises)*(len(transformations)-1)*3, desc='Generating Plate...')
for i in range(100):
    for font in fonts:
        # Create font directory if not exists
        if not os.path.exists(font): os.mkdir(font)
        # time.sleep(0.1)

        # Getting the letters list from nameMap csv
        letters = []
        with open(f'Fonts/{font}_namesMap.csv') as nameMapCsv:
            reader = csv.reader(nameMapCsv)
            next(reader)  # Skipping header
            letters = [rows[1] for rows in reader]
      #  print(letters)
        letters1=[]
        signs =['#','&','@',')','(','$']
        for i in letters:
            if i.isnumeric():
                continue
            if i in signs:
                continue
            letters1.append(i)
        for template in templates:
            for i in range(permutations):
                idCounter += 1

                # Generate a plate as an array
                # e.g. ['3', '7', 'GAF', '8', '5', '3']
                plate = getNewPlate()
                # Get the plate name as string
                # e.g. 37_GAF_853
                plateName = label = getPlateName(*plate)

                # Get Glyph images of plate characters
                glyphImages = []
                for glyph in plate:
                    glyphImage = Image.open(getGlyphAddress(font, glyph)).convert("RGBA")
                    # number.putalpha(255)
                    glyphImages.append(glyphImage)

                # Create a blank image with size of templates
                # and add the background and glyph images
                newPlate = Image.new('RGBA', (600, 132), (0, 0, 0, 0))
                background = Image.open(f'Templates/{template}.png').convert("RGBA")
                newPlate.paste(background, (0, 0))
                # adding glyph images with 11 pixel margin
                w = 0
                for i, glyph in enumerate(glyphImages):
                    if i == 2:
                        newPlate.paste(glyph, (70 + w, 30), mask=glyph)
                    elif i==6:
                        newPlate.paste(glyph, (480, 25), mask=glyph)
                    elif i==7:
                        newPlate.paste(glyph, (530, 25), mask=glyph)
                    else:
                        newPlate.paste(glyph, (70 + w, 25), mask=glyph)
                    w += glyph.size[0] +13

                idCounter += 1
                # Save Simple Plate
                _newPlate = newPlate.resize((312, 70), Image.ANTIALIAS)
                fontsProgBar.update(1)
                _newPlate.save(f"{font}/{plateName}_{template.split('-')[1]}{random.randint(0, 20)}{idCounter}.png")
                # newPlate.show(f"{font}/{plateName}_{template.split('-')[1]}.png")

                idCounter += 1
                noise1 = noises[np.random.randint(0, len(noises))]
                print(noise1)
                noisyTemplates = applyNoise(newPlate,noise1)
                for noisyTemplate in noisyTemplates:
                    idCounter += 1
                    fontsProgBar.update(1)
                    _noisyTemplate = noisyTemplate.resize((312, 70), Image.ANTIALIAS)
                    _noisyTemplate.save(
                        f"{font}/{plateName}_{template.split('-')[1]}{random.randint(0, 20)}{idCounter}.png")
                    transformedTemplates = applyTransforms(noisyTemplate)
                    for transformedTemplate in transformedTemplates:
                        idCounter += 1
                        _transformedTemplate = transformedTemplate.resize((312, 70), Image.ANTIALIAS)
                        fontsProgBar.update(1)
                        _transformedTemplate.save(
                            f"{font}/{plateName}_{template.split('-')[1]}{random.randint(0, 20)}{idCounter}.png")
                """"""
            fontsProgBar.update(1)
        fontsProgBar.update(1)
fontsProgBar.update(1)

fontsProgBar.close()
