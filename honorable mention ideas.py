import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
import vgamepad as gp
import time
import pickle
import statistics as stat
from PIL import ImageGrab


def custom_pts_with_sliders(frame, framex, framey, savefile_name):
    global pts
    pts = {}
    frame_shaped = cv2.resize(frame, (960, 640))
    cv2.namedWindow('Settings')
    cv2.resizeWindow('Settings', frame_shaped.shape[0], frame_shaped.shape[1])
    # settings_window = cv2.imread('Images/BlackPixel.png')
    # settings_window = cv2.resize(settings_window, (frame_shaped.shape[0], 1))
    cv2.imshow('Custom Points', frame_shaped)
    try:
        custom_pts_old_savefile = open(savefile_name, 'rb')

        # print(pickle.load(custom_pts_old_savefile))

        i = 1

        for point in pickle.load(custom_pts_old_savefile):
            pts[f"point {i}"] = point
            i += 1
    except:
        print('Error in "TRY" above >>>>')

    def change(val):
        line_image = np.copy(frame) * 0
        i = 1

        pts_pos = {}
        for point in pts:
            j = 1
            for coordinate in point:
                if j == 1:
                    pts_pos[f"x{i}"] = int(cv2.getTrackbarPos(f"x{i}", "Settings"))
                    # print(f"coordinate: {coordinate}, point: {point}, pts: {pts}, x: {i} j: {j}")
                if j == 2:
                    pts_pos[f"y{i}"] = int(cv2.getTrackbarPos(f"y{i}", "Settings"))
                    # print(f"coordinate: {coordinate}, point: {point}, pts: {pts}, y: {i} j: {j}")
                j += 1
            pts[f"point {i}"] = [pts_pos[f"x{i}"], pts_pos[f"y{i}"]]
            i += 1

        roi_test_pts = np.array(list(pts.values()))

        cv2.polylines(line_image, [roi_test_pts], True, (255, 150, 255), 8)
        gaming = cv2.addWeighted(frame, 1, line_image, 0.9, 0)
        gaming = cv2.resize(gaming, (960, 640))
        cv2.imshow('Custom Points', gaming)

    i = 1

    pts_pos = {}
    for point in pts:
        j = 1
        for coordinate in point:
            if j == 1:
                cv2.createTrackbar(f"x{i}", "Settings", pts[f"point {i}"][0], framex, change)
            if j == 2:
                cv2.createTrackbar(f"y{i}", "Settings", pts[f"point {i}"][1], framey, change)
            j += 1
        i += 1
    cv2.waitKey(0)

    i = 1

    pts_pos = {}
    for point in pts:
        j = 1
        for coordinate in point:
            if j == 1:
                pts_pos[f"x{i}"] = int(cv2.getTrackbarPos(f"x{i}", "Settings"))
                # print(f"coordinate: {coordinate}, point: {point}, pts: {pts}, x: {i} j: {j}")
            if j == 2:
                pts_pos[f"y{i}"] = int(cv2.getTrackbarPos(f"y{i}", "Settings"))
                # print(f"coordinate: {coordinate}, point: {point}, pts: {pts}, y: {i} j: {j}")
            j += 1
        pts[f"point {i}"] = [pts_pos[f"x{i}"], pts_pos[f"y{i}"]]
        i += 1

    roi_test_pts = np.array(list(pts.values()))

    custom_pts_savefile = open(savefile_name, 'wb')

    pickle.dump(list(pts.values()), custom_pts_savefile)

    cv2.destroyAllWindows()

    return roi_test_pts


# Import the image for testing, and crop it
frame = cv2.imread("Images/537775-L-3222752829.png")
# frame = cv2.imread("Images/actualFrame.png")
# cv2.imwrite('Images/actualFrame.png', frame)

# Define the Region Of Interest
roi = np.copy(frame) * 0
framey = frame.shape[0]
framex = frame.shape[1]

# Define the Region Of Interest
roi_adjusted_pts = custom_pts_with_sliders(frame, framex, framey, 'roi_savefile.pkl')
roi_pts = roi_adjusted_pts
# roi_pts = np.array([[int(framex*0), int(framey*0.73242188)], [int(framex*0), int(framey*0.605)],
#                     [int(framex*0.4140625), int(framey*0.43945312)], [int(framex*0.5859375), int(framey*0.43945312)],
#                     [int(framex*1), int(framey*0.605)], [int(framex*1), int(framey*0.73142188)]])

# Make a mask of the Region Of Interest
roi_mask = cv2.fillPoly(roi, [roi_pts], (255, 255, 255))
roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)
cv2.resize(roi_mask, (1920, 1080))

# Define Point for the steering Arrow
arrow_pts = [int(framex*0.5), int(framey*0.29296875)]

# Define the key out effect area
mask_adjusted_pts = custom_pts_with_sliders(frame, framex, framey, 'mask_savefile.pkl')
mask_pts = mask_adjusted_pts
# mask_pts = np.array([[int(framex*0.3125), int(framey*0.78125)], [int(framex*0.390625), int(framey*0.78125)],
#                      [int(framex*0.390625), int(framey*0.87890625)], [int(framex*0.3125), int(framey*0.87890625)]])

# Start timing startup time
begin = time.time()

# Define controller
try:
    vgp = gp.VX360Gamepad()
except:
    print("failed once \n")
    vgp = gp.VX360Gamepad()


def on_change_angle_slider(val):
    mask_parameter_function()


def on_change_lower_slider(val):
    mask_parameter_function()


def on_change_higher_slider(val):
    mask_parameter_function()


def on_change_switch_slider(val):
    mask_parameter_function()


def canny(img):
    # Blur and grey out the image, and crop out the ROI
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.medianBlur(gray, 5)
    edges_roi = cv2.bitwise_and(blur_gray, blur_gray, mask=roi_mask)

    # Put on Canny filter
    edges = cv2.Canny(image=edges_roi,
                      threshold1=120,
                      threshold2=150,
                      apertureSize=3)

    # Do a Houghlines algorithm
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        5,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=20,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )

    # Variables for combining the lines given
    line_image = np.copy(img) * 0
    lines_list = []

    slope_ave_r = []
    amount_of_lines_r = 0
    intercept_ave_r = []
    intercept_ave_r_left = []
    tot_length_r = 0

    slope_ave_l = []
    amount_of_lines_l = 0
    intercept_ave_l = []
    tot_length_l = 0

    i = 0

    # Iterate over points
    for points in lines:
        x1, y1, x2, y2 = points[0]
        if x2 < x1:
            temp_x1 = x1
            temp_y1 = y1
            x1 = x2
            y1 = y2
            x2 = temp_x1
            y2 = temp_y1

        if x1 != x2:
            slope = (y1 - y2) / (x1 - x2)
            intercept_y_axis = y1 - (x1 * slope)

        else:
            slope = 0
            intercept_y_axis = 0

        # and x1 <= img.shape[1] / 2 and x2 <= img.shape[1] / 2

        if slope < 0 and x2 <= roi_pts[3][0]:

            length_l = math.sqrt(((y1 - y2) ** 2) + ((x1 - x2) ** 2))
            # tot_length_l += length_l
            for i in range(int(length_l)):
                slope_ave_l.append(slope)
            intercept_ave_l.append(intercept_y_axis)

            amount_of_lines_l += 1

            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # and x1 >= img.shape[1] / 2 and x2 >= img.shape[1] / 2

        elif slope >= 0 and x1 >= roi_pts[2][0]:

            # length_r = math.sqrt(((y1 - y2) ** 2) + ((x1 - x2) ** 2))
            # tot_length_r += length_r
            slope_ave_r.append(slope)
            intercept_ave_r_left.append(intercept_y_axis)
            intercept_ave_r.append((framex*slope + intercept_y_axis))

            amount_of_lines_r += 1

            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 4)

        else:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

        i += 1
        if i == len(lines) and amount_of_lines_l > 0:
            # slope_ave_l = (slope_ave_l / amount_of_lines_l) / (tot_length_l / amount_of_lines_l)
            # slope_ave_r = (slope_ave_r / amount_of_lines_r) / (tot_length_r / amount_of_lines_r)
            beg = time.time()
            slope_ave_l.sort()
            en = time.time()
            print(en - beg, " sec, ", beg, " beg, ", en, " en.")
            slope_ave_l = stat.median(slope_ave_l)
            slope_ave_r = stat.median(slope_ave_r)

            # intercept_ave_l = int((intercept_ave_l / amount_of_lines_l) / (tot_length_l / amount_of_lines_l))
            # intercept_ave_r = int((intercept_ave_r / amount_of_lines_r) / (tot_length_r / amount_of_lines_r))
            # intercept_ave_r_left = int((intercept_ave_r_left / amount_of_lines_r) / (tot_length_r / amount_of_lines_r))
            intercept_ave_l = int(stat.median(intercept_ave_l))
            intercept_ave_r = int(stat.median(intercept_ave_r))
            intercept_ave_r_left = int(stat.median(intercept_ave_r_left))

            line_l_x2 = int((roi_pts[3][1] - intercept_ave_l)/slope_ave_l)
            color_left = (249, 171, 59)
            cv2.line(line_image, (0, intercept_ave_l), (line_l_x2, int(roi_pts[3][1])), color_left, 8)

            line_r_x2 = abs(int((roi_pts[3][1] - intercept_ave_r_left) / slope_ave_r))
            color_right = (126, 138, 249)
            cv2.line(line_image, (int(framex), intercept_ave_r), (line_r_x2, int(roi_pts[3][1])), color_right, 8)

            intercept_middle = int((intercept_ave_r + intercept_ave_l)/2)
            line_middle_x2 = int((line_r_x2 + line_l_x2)/2)
            color_middle = (92, 247, 133)
            cv2.line(line_image, (int(framex*0.5), intercept_middle), (line_middle_x2, int(roi_pts[3][1])), color_middle, 20)
            middle_line_angle = (math.degrees(math.atan((intercept_middle - int(roi_pts[3][1]))/(int(framex*0.5) - line_middle_x2))))
            cv2.putText(line_image, text=f'{middle_line_angle}',
                        org=(int(framex*0.5), intercept_middle),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color_middle,
                        thickness=5)
            print("real angle: " + str(middle_line_angle.__round__(4)))
            try:
                # given_angle = cv2.getTrackbarPos('angle', 'window')
                given_angle = middle_line_angle
            except:
                given_angle = middle_line_angle
            print(f"given angle: {given_angle.__round__(4)}")

            middle_line_angle_corrected = (((given_angle * 2)/180) - 1).__round__(4)
            cv2.arrowedLine(line_image, arrow_pts, [arrow_pts[0] + int(300 * middle_line_angle_corrected), arrow_pts[1]],
                            (0, 209, 250), thickness=8)
            cv2.putText(line_image, text=f'{middle_line_angle_corrected}',
                        org=arrow_pts,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 209, 250),
                        thickness=5)

            if middle_line_angle > 100:
                vgp.left_joystick_float(x_value_float=middle_line_angle_corrected, y_value_float=0)
                vgp.update()
                print(f"steer right: {middle_line_angle_corrected}")
            elif middle_line_angle < 80:
                vgp.left_joystick_float(x_value_float=middle_line_angle_corrected, y_value_float=0)
                vgp.update()
                print(f"steer left: {middle_line_angle_corrected}")
            else:
                print(f"dont steer: {middle_line_angle_corrected}")

        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])

    cv2.polylines(line_image, [mask_pts], True, (0, 137, 255), 8)
    cv2.polylines(line_image, [roi_pts], True, (255, 150, 255), 4)
    cv2.circle(line_image, arrow_pts, 8, (0, 255, 255), lineType=cv2.FILLED, thickness=-8)
    lines_edges = cv2.addWeighted(frame, 1, line_image, 0.7, 0)

    return lines_edges


def keyout_mask_function(img):
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_state = 0  # h = 0, s = 1, v = 2
    h = []
    s = []
    v = []
    # h_lowest = 360
    # h_highest = 0

    img = img[mask_pts[0][1]:mask_pts[2][1], mask_pts[0][0]:mask_pts[2][0]]

    for pixel in np.nditer(img):
        if hsv_state == 0:
            h.append(pixel)

            # if pixel <= h_lowest:
            #     h_lowest = pixel
            # if pixel >= h_highest:
            #     h_highest = pixel
        elif hsv_state == 1:
            s.append(pixel)
        elif hsv_state == 2:
            v.append(pixel)

        if hsv_state != 2:
            hsv_state += 1
        else:
            hsv_state = 0

    offset_of_median = 50

    h_median = stat.median(h)
    s_median = stat.median(s)
    v_median = stat.median(v)

    h_highest = int(h_median + offset_of_median)
    h_lowest = int(h_median - offset_of_median)

    s_highest = int(s_median + offset_of_median)
    s_lowest = int(s_median - offset_of_median)

    v_highest = int(v_median + offset_of_median)
    v_lowest = int(v_median - offset_of_median)
    print(f"h_median: {h_median}, s_median: {s_median}, v_median: {v_median}")
    return h_highest, h_lowest, s_highest, s_lowest, v_highest, v_lowest


def mask_parameter_function():
    try:
        switch = cv2.getTrackbarPos('switch', 'window')
    except:
        switch = 0
    if switch == 1:
        lowerr = cv2.getTrackbarPos('lower', 'window')
        higherr = cv2.getTrackbarPos('higher', 'window')
        hsv_ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue_ = np.array([lowerr, 0, 0])
        upper_blue_ = np.array([higherr, 255, 255])

        # Here we are defining range of bluecolor in HSV
        # This creates a mask of blue coloured
        # objects found in the frame.
        mask_ = cv2.inRange(hsv_, lower_blue_, upper_blue_)
        mask_ = cv2.medianBlur(mask_, 5)

        # The bitwise and of the frame and mask is done so
        # that only the blue coloured objects are highlighted
        # and stored in res
        res_ = cv2.bitwise_and(frame, frame, mask=mask_)

        res_ = frame - res_

        edges = canny(res_)
        combined_ = cv2.hconcat([res_, frame, edges])
        combined_ = cv2.resize(combined_, (1280, 640))
        cv2.imshow('window', combined_)
        cv2.imwrite('Result.png', combined_)

    else:

        lowerr = cv2.getTrackbarPos('lower', 'window')
        higherr = cv2.getTrackbarPos('higher', 'window')
        hsv_ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue_ = np.array([lowerr, 0, 0])
        upper_blue_ = np.array([higherr, 255, 255])

        # Here we are defining range of bluecolor in HSV
        # This creates a mask of blue coloured
        # objects found in the frame.
        mask_ = cv2.inRange(hsv_, lower_blue_, upper_blue_)
        mask_ = cv2.medianBlur(mask_, 5)

        # The bitwise and of the frame and mask is done so
        # that only the blue coloured objects are highlighted
        # and stored in res
        res_ = cv2.bitwise_and(frame, frame, mask=mask_)

        edges = canny(res_)
        combined_ = cv2.hconcat([res_, frame, edges])
        combined_ = cv2.resize(combined_, (1280, 640))
        cv2.imshow('window', combined_)
        cv2.imwrite('Result.png', combined_)


# Custom key out effect
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

h_higher, h_lower, s_higher, s_lower, v_higher, v_lower = keyout_mask_function(frame)

print(f"higher: {[h_lower, s_lower, v_lower]}, lower: {[h_higher, s_higher, v_higher]}")

lower_h_mask = np.array([h_lower, s_lower, v_lower])
upper_h_mask = np.array([h_higher, s_higher, v_higher])

mask = cv2.inRange(hsv, lower_h_mask, upper_h_mask)
res = cv2.bitwise_and(frame, frame, mask=mask)

# Put together the images, and resize them
combined = cv2.hconcat([res, frame])
combined = cv2.resize(combined, (1280, 640))

# Display the window, and create sliders
cv2.imshow('window', combined)
cv2.createTrackbar('lower', 'window', h_lower, 360, on_change_lower_slider)
cv2.createTrackbar('higher', 'window', h_higher, 360, on_change_higher_slider)
cv2.createTrackbar('switch', 'window', 0, 1, on_change_switch_slider)
cv2.createTrackbar('angle', 'window', 90, 180, on_change_angle_slider)

end = time.time()
difference = end - begin
print(f"startup time: {difference.__round__(3)} sec")

cv2.waitKey(0)

cv2.destroyAllWindows()










































if middle_line_angle_bisection > 100:
    vgp.left_joystick_float(x_value_float=middle_line_angle_bisection_corrected, y_value_float=0)
    vgp.update()
    print(f"steer right: {middle_line_angle_bisection_corrected}")
elif middle_line_angle_bisection < 80:
    vgp.left_joystick_float(x_value_float=middle_line_angle_bisection_corrected, y_value_float=0)
    vgp.update()
    print(f"steer left: {middle_line_angle_bisection_corrected}")
else:
    print(f"dont steer: {middle_line_angle_bisection_corrected}")