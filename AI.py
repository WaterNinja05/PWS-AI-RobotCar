import math

import cv2
import numpy as np
import time
from PIL import ImageGrab

# screen = np.array(ImageGrab.grab(bbox=(-1280, 0, 0, 1024), include_layered_windows=False, all_screens=True))
#
# screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB, 20)
# image = screen[(-1280 + 100):300, (-1280 + 800):600]
# cv2.imshow("test", image)
# cv2.waitKey(0)


def lab_green_filter(imagee, amount):



    # lab = cv2.cvtColor(imagee, cv2.COLOR_BGR2Lab)
    # lab[:, :, 1] = lab[:, :, 1] * -amount
    # lab[:, :, 1][lab[:, :, 1] > 255] = 255
    # imagee = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return imagee


def contrast_brigthness(imagee, alpha, beta):
    new_image = np.zeros(imagee.shape, imagee.dtype)
    # Initialize values
    try:
        alpha = float(alpha)
        beta = int(beta)
    except ValueError:
        print('Error, not a number, contrast_brigthness')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    new_image = cv2.convertScaleAbs(imagee, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    return new_image


def saturate(screenz):
    hsv = cv2.cvtColor(screenz, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * 2
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    # np.logical_and(hsv[:, :, 0] > 75, hsv[:, :, 0] < 160) = 0
    hsv = np.array(hsv, dtype=np.uint8)
    screenz = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return screenz


while True:
    # image = cv2.imread("Images/20180722_170202.jpg")
    # screen = np.array(ImageGrab.grab(bbox=(-1280, 60, 0, 1074), include_layered_windows=False, all_screens=True))
    screen = cv2.imread("Images/img.png")
    screen = screen[50:1074, 0:1280]

    # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB, 20)
    image = screen
    image_contrast = contrast_brigthness(image, 1, 30)
    image_sat = saturate(image_contrast)

    gray = cv2.cvtColor(image_sat, cv2.COLOR_BGR2GRAY, 20)
    kernel_size = 5
    blur_gray = cv2.medianBlur(gray, 7, 2)
    blur_gray = cv2.cvtColor(blur_gray, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(blur_gray,
                      threshold1=120,
                      threshold2=150,
                      apertureSize=3)

    edges_copy = edges[0:800, 0:800]

    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        5,  # Distance resolution in pixels
        np.pi / 200,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=3,  # Min allowed length of line
        maxLineGap=80  # Max allowed gap between line for joining them
    )

    line_image = np.copy(image) * 0
    lines_list = []
    # Iterate over points

    for points in lines:

        # Extracted points nested in the list

        x1, y1, x2, y2 = points[0]

        # deg = 0
        # rc = 0
        # Draw the lines joing the points
        # On the original image
        # rc = y2 - y1 / x2 - x1
        # print(f"RC:    {rc}")
        # deg = math.degrees(math.atan(rc))
        # print(f"DEG:   {deg}")
        if 1:
            cv2.line(line_image, (x1, y1), (x2, y2), (120, 220, 0), 4)
            # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])

    lines_edges = cv2.addWeighted(image, 1, line_image, 0.7, 0)

    # Save the result image
    cv2.imwrite('detectedLines.png', lines_edges)

    # gray = cv2.resize(gray, (700, 380))
    edges = cv2.resize(edges, (700, 380))
    lines_edges = cv2.resize(lines_edges, (700, 380))
    # screen = cv2.resize(screen, (700, 380))
    # blur_gray = cv2.resize(blur_gray, (700, 380))

    lower_frame = cv2.hconcat([image_sat, image_contrast])
    upper_frame = cv2.hconcat([image, blur_gray])
    total_frame = cv2.vconcat([upper_frame, lower_frame])
    total_frame = cv2.resize(total_frame, (1280, 1024))
    cv2.imshow('Total Frame', total_frame)

    # cv2.imshow("gray", gray)
    # cv2.imshow("blur_gray", blur_gray)
    # cv2.imshow("edges", edges)
    # cv2.imshow("edges_copy", lines_edges)
    # cv2.imshow('Python Window', screen)
    # cv2.imshow("image", image)
    # cv2.imshow("image_sat", image_sat)
    # cv2.imshow("image_contrast", image_contrast)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
        break
