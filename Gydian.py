import math
import cv2
import numpy as np
import time
import pickle
from ServoHatModule import *

global h_higher, h_lower, s_higher, s_lower, v_higher, v_lower
global white

pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)

avg_time = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cap = cv2.VideoCapture(0)
success, image = cap.read()
cv2.imshow('test', image)
cv2.waitKey(0)
if cap.isOpened() is False:
    cap.open()
if cap.isOpened() is False:
    print("Cap fault-----------------")
    exit()


def get_current_frame(continuous=False):
    mon = {'top': 0, 'left': -1280, 'width': 1280, 'height': 1024}
    if continuous:
        while True:
            last_time = time.time()
            mask_parameter_function()
            print(f'fps: {1 / (time.time() - last_time)}')
            if cv2.waitKey(5) & 0xFF == ord('q'):
                pwm.setServoPulse(0, 1500)
                pwm.setServoPulse(1, 500)
                cv2.destroyAllWindows()
                break
    else:
        begin = time.time()
        if cap.isOpened():
            success, image = cap.read()
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # image.flags.writeable = False
            # img = cv2.resize(image, (1280, 1024))
        print(f'fps_one_time: {1 / (time.time() - begin)}')
        return image


def correct_angle(angle):
    corrected_angle = (((angle * 2)/180) - 1).__round__(4)
    return corrected_angle


def correct_dist(dist, framex):
    corrected_dist = (((dist * 2)/framex) - 1).__round__(4)
    return corrected_dist


def median(list):
    if list is None or list == []:
        print(list)
        return 1
    list.sort()
    mid = len(list) // 2
    if len(list) % 2 == 0:
        median = list[mid]
    else:
        median = (list[mid] + list[~mid]) / 2

    return median


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


def on_change_angle_slider(val):
    mask_parameter_function()


def on_change_switch_slider(val):
    mask_parameter_function()


def canny(img):
    beg = time.time()

    # Blur and grey out the image, and crop out the ROI
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.medianBlur(gray, 5)
    edges_roi = cv2.bitwise_and(blur_gray, blur_gray, mask=roi_mask)
    # cv2.imwrite('roi.png', edges_roi)
    # cv2.imwrite('img.png', img)
    # edges_roi = cv2.pyrDown(edges_roi)
    # edges_roi = cv2.resize(edges_roi, scale)

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
        threshold=200,  # Min number of votes for valid line
        minLineLength=50,  # Min allowed length of line
        maxLineGap=20  # Max allowed gap between line for joining them
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
    try:
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

                # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 4)

            # and x1 >= img.shape[1] / 2 and x2 >= img.shape[1] / 2

            elif slope > 0 and x1 >= roi_pts[2][0]:

                length_r = math.sqrt(((y1 - y2) ** 2) + ((x1 - x2) ** 2))
                # tot_length_r += length_r
                for i in range(int(length_r)):
                    slope_ave_r.append(slope)
                    intercept_ave_r_left.append(intercept_y_axis)
                    intercept_ave_r.append((framex*slope + intercept_y_axis))

                amount_of_lines_r += 1

                # cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 4)

            else:
                # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                o = 0

            # Maintain a simples lookup list for points
            # lines_list.append([(x1, y1), (x2, y2)])
    except:
        print("No lines :(")
        print(edges)
        return line_image

    # slope_ave_l = (slope_ave_l / amount_of_lines_l) / (tot_length_l / amount_of_lines_l)
    # slope_ave_r = (slope_ave_r / amount_of_lines_r) / (tot_length_r / amount_of_lines_r)

    slope_ave_l = median(slope_ave_l)
    slope_ave_r = median(slope_ave_r)

    # intercept_ave_l = int((intercept_ave_l / amount_of_lines_l) / (tot_length_l / amount_of_lines_l))
    # intercept_ave_r = int((intercept_ave_r / amount_of_lines_r) / (tot_length_r / amount_of_lines_r))
    # intercept_ave_r_left = int((intercept_ave_r_left / amount_of_lines_r) / (tot_length_r / amount_of_lines_r))
    intercept_ave_l = int(median(intercept_ave_l))
    intercept_ave_r = int(median(intercept_ave_r))
    intercept_ave_r_left = int(median(intercept_ave_r_left))

    line_l_x2 = int((roi_pts[3][1] - intercept_ave_l)/slope_ave_l)
    color_left = (249, 171, 59)
    cv2.line(line_image, (0, intercept_ave_l), (line_l_x2, int(roi_pts[3][1])), color_left, 8)

    line_r_x2 = abs(int((roi_pts[3][1] - intercept_ave_r_left) / slope_ave_r))
    color_right = (126, 138, 249)
    cv2.line(line_image, (int(framex), intercept_ave_r), (line_r_x2, int(roi_pts[3][1])), color_right, 8)

    s_mid_x_bisection = (intercept_ave_l - intercept_ave_r_left) / (slope_ave_r - slope_ave_l)
    s_mid_y_bisection = slope_ave_l * s_mid_x_bisection + intercept_ave_l
    angle_between_lines_bisection = 180 - (math.degrees(math.atan(abs(slope_ave_l))) + math.degrees(math.atan(abs(slope_ave_r))))
    middle_line_angle_bisection = (0.5 * angle_between_lines_bisection) + math.degrees(math.atan(abs(slope_ave_r)))
    slope_ave_mid_bisection = math.tan(math.radians(middle_line_angle_bisection))
    intercept_ave_mid_bisection = int(s_mid_y_bisection - (slope_ave_mid_bisection * s_mid_x_bisection))

    x_mid_roi = int((roi_pts[3][1] - intercept_ave_mid_bisection) / slope_ave_mid_bisection)
    x_mid_framey_bisection = int((framey - intercept_ave_mid_bisection) / slope_ave_mid_bisection)
    x_l_framey = int((framey - intercept_ave_l) / slope_ave_l)
    x_r_framey = int((framey - intercept_ave_r_left) / slope_ave_r)
    x_mid_framey = int((x_r_framey + x_l_framey) / 2)
    middle_line_angle = -math.degrees(math.atan((s_mid_y_bisection - framey) / (s_mid_x_bisection - x_mid_framey)))

    color_middle = (92, 247, 133)
    color_middle_bisection = (170, 0, 255)

    # print(f"Slope Left: {slope_ave_l}, Intercept Y-axis Left: {intercept_ave_l}"
    #       f"\nSlope Right: {slope_ave_r}, Intercept FrameX Right: {intercept_ave_r}, "
    #       f"Intercept Y-Axis Right: {intercept_ave_r_left}"
    #       f"\nSlope Mid: {slope_ave_mid_bisection}, Intercept Y-Axis Mid: {intercept_ave_mid_bisection}")
    # print(f"S({s_mid_x_bisection};{s_mid_y_bisection})")
    # print(f"framey: {framey}, x roi: {x_mid_roi}, x framey: {x_mid_framey}, roi y {roi_pts[3][1]}")

    cv2.line(line_image, (int(s_mid_x_bisection), int(s_mid_y_bisection)),
             (x_mid_framey, framey), color_middle, 20)
    cv2.line(line_image, (int(s_mid_x_bisection), int(s_mid_y_bisection)),
             (x_mid_framey_bisection, framey), color_middle_bisection, 20)

    # cv2.putText(line_image, text=f'{middle_line_angle_bisection.__round__(2)}',
    #             org=(int(x_mid_framey_bisection), framey - 100),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color_middle_bisection,
    #             thickness=5)
    # cv2.putText(line_image, text=f'{middle_line_angle.__round__(2)}',
    #             org=(int(x_mid_framey), framey - 100),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color_middle,
    #             thickness=5)

    switch = cv2.getTrackbarPos('angle', 'window')

    angle_steer = 0
    dist_steer = 0

    if switch == 1:
        # angle steer computing
        print("real angle: " + str(middle_line_angle_bisection.__round__(4)))
        middle_line_angle_bisection_corrected = correct_angle(middle_line_angle_bisection)
        if middle_line_angle_bisection <= 80:
            print(f"steer left: {middle_line_angle_bisection_corrected}")
        elif middle_line_angle_bisection >= 100:
            print(f"steer right: {middle_line_angle_bisection_corrected}")

        angle_steer = middle_line_angle_bisection_corrected

        # cv2.arrowedLine(line_image, arrow1_pts,
        #                 [arrow1_pts[0] + int(300 * middle_line_angle_bisection_corrected), arrow1_pts[1]],
        #                 color_middle_bisection, thickness=8)
        # cv2.putText(line_image, text=f'{middle_line_angle_bisection_corrected}',
        #             org=arrow1_pts,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color_middle_bisection,
        #             thickness=5)

        # distance steer computing
        print("real dist: " + str(x_mid_framey_bisection.__round__(4)))
        x_mid_framey_bisection_corrected = correct_dist(x_mid_framey_bisection, framex)
        if x_mid_framey_bisection_corrected <= (1 - (1 / 9)):
            print(f"steer left to the middle: {x_mid_framey_bisection_corrected}")
        elif x_mid_framey_bisection_corrected >= (1 + (1 / 9)):
            print(f"steer right to the middle: {x_mid_framey_bisection_corrected}")

        dist_steer = x_mid_framey_bisection_corrected

        # cv2.arrowedLine(line_image, arrow2_pts,
        #                 [arrow2_pts[0] + int(300 * x_mid_framey_bisection_corrected), arrow2_pts[1]],
        #                 color_middle_bisection, thickness=8)
        # cv2.putText(line_image, text=f'{x_mid_framey_bisection_corrected}',
        #             org=arrow2_pts,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color_middle_bisection,
        #             thickness=5)
    else:
        # angle steer computing
        print("real angle: " + str(middle_line_angle.__round__(4)))
        middle_line_angle_corrected = correct_angle(middle_line_angle)
        if middle_line_angle <= 80:
            print(f"steer left: {middle_line_angle_corrected}")
        elif middle_line_angle >= 100:
            print(f"steer right: {middle_line_angle_corrected}")

        angle_steer = middle_line_angle_corrected

        # cv2.arrowedLine(line_image, arrow1_pts,
        #                 [arrow1_pts[0] + int(300 * middle_line_angle_corrected), arrow1_pts[1]],
        #                 color_middle, thickness=8)
        # cv2.putText(line_image, text=f'{middle_line_angle_corrected}',
        #             org=arrow1_pts,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color_middle,
        #             thickness=5)

        # distance steer computing
        print("real dist: " + str(x_mid_framey.__round__(4)))
        x_mid_framey_corrected = correct_dist(x_mid_framey, framex)
        if x_mid_framey_corrected <= (1 - (1 / 9)):
            print(f"steer left to the middle: {x_mid_framey_corrected}")
        elif x_mid_framey_corrected >= (1 + (1 / 9)):
            print(f"steer right to the middle: {x_mid_framey_corrected}")

        dist_steer = x_mid_framey_corrected

        # cv2.arrowedLine(line_image, arrow2_pts,
        #                 [arrow2_pts[0] + int(300 * x_mid_framey_corrected), arrow2_pts[1]],
        #                 color_middle, thickness=8)
        # cv2.putText(line_image, text=f'{x_mid_framey_corrected}',
        #             org=arrow2_pts,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color_middle,
        #             thickness=5)

    angle_weight = cv2.getTrackbarPos('angle weight', 'window')
    dist_weight = cv2.getTrackbarPos('dist weight', 'window')

    steering_ave = (((-angle_steer * angle_weight) + (dist_steer * dist_weight)) / (angle_weight + dist_weight)).__round__(2)
    steer(steering_ave)

    # cv2.arrowedLine(line_image, arrow3_pts,
    #                 [arrow3_pts[0] + int(300 * steering_ave), arrow3_pts[1]],
    #                 (0, 209, 255), thickness=8)
    # cv2.putText(line_image, text=f'{steering_ave}',
    #             org=arrow3_pts,
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 209, 255),
    #             thickness=5)

    # cv2.polylines(line_image, [mask_pts], True, (0, 137, 255), 8)
    # cv2.polylines(line_image, [roi_pts], True, (255, 150, 255), 4)
    # cv2.imwrite("line_image.png", line_image)
    lines_edges = cv2.addWeighted(img, .5, line_image, 0.7, 0)

    en = time.time()
    total_time = (en - beg).__round__(4)
    print(f"Edge Detection time: {1/total_time} fps \n---------------------------------")

    return lines_edges


def keyout_mask_function(img):
    begin = time.time()
    hsv_state = 0  # h = 0, s = 1, v = 2
    h = []
    s = []
    v = []

    img = img[mask_pts[0][1]:mask_pts[2][1], mask_pts[0][0]:mask_pts[2][0]]

    for pixel in np.nditer(img):
        # print(pixel, hsv_state)
        if hsv_state == 0:
            h.append(pixel)
        elif hsv_state == 1:
            s.append(pixel)
        elif hsv_state == 2:
            v.append(pixel)

        if hsv_state != 2:
            hsv_state += 1
        else:
            hsv_state = 0

    offset_of_median = 85

    h_median = median(h)
    s_median = median(s)
    v_median = median(v)

    h_highest = int(h_median + offset_of_median)
    h_lowest = int(h_median - offset_of_median)

    s_highest = int(s_median + offset_of_median)
    s_lowest = int(s_median - offset_of_median)

    v_highest = int(v_median + offset_of_median)
    v_lowest = int(v_median - offset_of_median)
    print(f"h_median: {h_median}, s_median: {s_median}, v_median: {v_median}")
    end = time.time()
    print(f"Mask time: {round(end - begin, 2)} sec")
    return h_highest, h_lowest, s_highest, s_lowest, v_highest, v_lowest


def mask_parameter_function():
    frame = get_current_frame()
    try:
        switch = cv2.getTrackbarPos('switch', 'window')
    except:
        switch = 0
    if switch == 1:
        hsv_ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_ = cv2.bitwise_and(hsv_, hsv_, mask=roi_mask)
        lower_blue_ = np.array([h_lower, s_lower, 0])
        upper_blue_ = np.array([h_higher, s_higher, 255])

        mask_ = cv2.inRange(hsv_, lower_blue_, upper_blue_)
        mask_ = cv2.medianBlur(mask_, 5)
        mask_ = cv2.bitwise_not(mask_)
        res_ = cv2.bitwise_and(white, white, mask=mask_)

        edges = canny(res_)
        # combined_ = cv2.hconcat([res_, frame, edges])
        # # combined_ = cv2.hconcat([edges])
        # combined_ = cv2.resize(combined_, (1280, 640))
        # cv2.imshow('window', combined_)
        # cv2.imwrite('Result.png', combined_)

    else:
        hsv_ = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv_ = cv2.bitwise_and(hsv_, hsv_, mask=roi_mask)
        lower_blue_ = np.array([h_lower, s_lower, 0])
        upper_blue_ = np.array([h_higher, s_higher, 255])

        mask_ = cv2.inRange(hsv_, lower_blue_, upper_blue_)
        mask_ = cv2.medianBlur(mask_, 5)

        res_ = cv2.bitwise_and(white, white, mask=mask_)

        edges = canny(res_)
        # combined_ = cv2.hconcat([res_, frame, edges])
        combined_ = cv2.hconcat([edges])
        # combined_ = cv2.resize(combined_, (1280, 640))
        cv2.imshow('window', combined_)
        # cv2.imwrite('Result.png', combined_)


def steer(steering):
    steering = steering * 0.71
    if steering >= 1:
        steering = 1
    elif steering <= -1:
        steering = -1
    steering = ((steering + 1)*1000)+500-72
    print(f"steer pulse: {steering} pwm, gas pulse: {1500} pwm")
    pwm.setServoPulse(0, steering)
    pwm.setServoPulse(1, 1510)


# Import the image for testing, and crop it
frame = get_current_frame()
# frame = cv2.imread("Images/actualFrame.png")
# cv2.imwrite('Images/actualFrame.png', frame)

# Define the Region Of Interest
roi = np.copy(frame) * 0
framey = frame.shape[0]
framex = frame.shape[1]
white = cv2.imread('Images/White.png')
white = cv2.resize(white, (framex, framey))

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
arrow1_pts = [int(framex * 0.5), int(framey * 0.29296875)]
arrow2_pts = [int(framex*0.5), int(framey*0.29296875 + 100)]
arrow3_pts = [int(framex*0.5), int(framey*0.29296875 - 100)]

# Define the key out effect area
mask_adjusted_pts = custom_pts_with_sliders(frame, framex, framey, 'mask_savefile.pkl')
mask_pts = mask_adjusted_pts
# mask_pts = np.array([[int(framex*0.3125), int(framey*0.78125)], [int(framex*0.390625), int(framey*0.78125)],
#                      [int(framex*0.390625), int(framey*0.87890625)], [int(framex*0.3125), int(framey*0.87890625)]])

# Start timing startup time
begin = time.time()

# Custom key out effect
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

h_higher, h_lower, s_higher, s_lower, v_higher, v_lower = keyout_mask_function(hsv)

lower_h_mask = np.array([h_lower, s_lower, v_lower])
upper_h_mask = np.array([h_higher, s_higher, v_higher])

mask = cv2.inRange(hsv, lower_h_mask, upper_h_mask)
res = cv2.bitwise_and(frame, frame, mask=mask)

# Put together the images, and resize them
combined = cv2.hconcat([res, frame])
combined = cv2.resize(combined, (1280, 640))

# Display the window, and create sliders
cv2.imshow('window', combined)
cv2.createTrackbar('switch', 'window', 0, 1, on_change_switch_slider)
cv2.createTrackbar('angle', 'window', 1, 1, on_change_angle_slider)
cv2.createTrackbar('angle weight', 'window', 5, 100, on_change_angle_slider)
cv2.createTrackbar('dist weight', 'window', 3, 100, on_change_angle_slider)

end = time.time()
difference = end - begin
print(f"startup time: {difference.__round__(3)} sec")

# cv2.waitKey(0)

get_current_frame(True)

cv2.waitKey(0)

cv2.destroyAllWindows()

cap.release()
