# -------------------------------------------------------------------------------------------------------------------------------------- #
import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture("Shot-Predictor-Video.mp4")


# Video
video_width = int(cap.get(3))
video_height = int(cap.get(4))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
   
size = (video_width, video_height)
fps = 60
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.mp4', fourcc,frame_rate,size)

# Color variables
lower_green = np.array([56,161,38])
uppper_green = np.array([71,255,94])

lower_field = np.array([56,131,4])
upper_field = np.array([75,221,216])

lower_que = np.array([10,14,144])
upper_que = np.array([100,42,255])

#Kernal 
k_size = 5
k_shape = cv2.MORPH_RECT
kernal_open = cv2.getStructuringElement(k_shape,(k_size,k_size))
kernal_close = cv2.getStructuringElement(k_shape,(k_size+3,k_size+3))

#State Variables
player = np.array([9999,9999])
timeout = 1
wait = False
going_in = False

#Shot Time 
shot_time_outs = [95,130,123,150,74,175,120,135,102,190]
shot_nr  = 0

#Prediction Text
text_in = "Prediction: IN"
text_out = "Prediction: OUT"

#Text
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 3
color_in = (0,255,0) # Green Color
color_out = (0,0,255) # Green Color
text_size_in = cv2.getTextSize(text_in,font,scale,2)[0]
text_size_in = cv2.getTextSize(text_out,font,scale,2)[0]


#Functions
def get_intersection(movement_dir, top_left, bot_right, target):
    # calculates the intersection point between a movement direction
    # and a rectangle defined by its top-left and bottom-right corners

    theta = math.atan2(movement_dir[0], movement_dir[1])
    a = math.sin(theta)
    b = math.cos(theta)
    w = bot_right[0]
    h = bot_right[1]
    rx = top_left[0]
    ry = top_left[1]

    t1 = (w-target[0])/a
    t2 = (h-target[1])/b
    t3 = (rx-target[0])/a
    t4 = (ry-target[1])/b

    curr_t = 9999999
    for t in [t1,t2,t3,t4]:
        if t > 0 and t < curr_t:
            curr_t = t

    if curr_t == t1 or curr_t == t3:
        normal = np.array([1,0])
    else:
        normal = np.array([0,1])
    intersection = (int(target[0] + curr_t*a), int(target[1] + curr_t*b))
    return intersection, normal

def predict_in(pockets, pred):
    for p in pockets:
        if np.linalg.norm(p-pred) < 59:
            return True
        
        elif np.linalg.norm(p-pred) == 921.4732768778484:
            return True
        
    return False
while True:
    s,img = cap.read()
    if s:
        
        if timeout > 0:
            img = cv2.resize(img,(0,0),None,0.7,0.7) #Resizing Window
            video.write(img)
            cv2.imshow("Window",img)
            cv2.waitKey(50)
            timeout -= 1
            continue

        # Converting BGR image to HSV
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        # Getting the image threshold of green pixel
        green_mask = cv2.inRange(hsv_img,lower_green,uppper_green)

        # Remove small objects from an image while preserving the shape and size of larger objects
        img_open = cv2.morphologyEx(green_mask,cv2.MORPH_OPEN,kernal_open)
        img_close = cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,kernal_close)

        # Hough transform to detect lines in the img_close image.
        lines = cv2.HoughLinesP(img_close,1,np.pi/180,100,maxLineGap=10)

        angle_threshold = 1
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1,x2-x1)*180/np.pi)
            if angle < angle_threshold:
                horizontal_lines.append(line)
            elif angle > 90 - angle_threshold and angle < 90 + angle_threshold:
                vertical_lines.append(line)
        
        middle_y = img.shape[0] // 2
        top_lines = [line for line in horizontal_lines if line [0][1] < middle_y]
        bot_lines = [line for line in horizontal_lines if line [0][1] > middle_y]

        top_line = sorted(top_lines, key=lambda x:x[0][1])[0]
        bot_line = sorted(bot_lines, key=lambda x:x[0][1])[-1]

        middle_x = img.shape[1]//2
        left_lines = [line for line in vertical_lines if line[0][0] < middle_x]
        right_lines = [line for line in vertical_lines if line[0][0] > middle_x]

        left_line = sorted(left_lines, key=lambda x:x[0][1])[0]
        right_line = sorted(right_lines, key=lambda x:x[0][1])[-1]

        top_left = [left_line[0][0],top_line[0][1]]
        bot_left = [left_line[0][0],bot_line[0][1]]
        top_right = [right_line[0][0],top_line[0][1]]
        bot_right = [right_line[0][0],bot_line[0][1]]

        corners = [top_left,bot_left,bot_right,top_right]
        offset = 60.1
        b = 15

        w_rect = [[top_left[0]+b, top_left[1]+offset],[bot_left[0]+b, bot_left[1]-offset],[bot_right[0]-b, bot_right[1]-offset],[top_right[0]-b, top_right[1]+offset]]
        h_rect = [[top_left[0]+offset, top_left[1]+b],[bot_left[0]+offset, bot_left[1]-b],[bot_right[0]-offset, bot_right[1]-b],[top_right[0]-offset, top_right[1]+b]]
        
        mid = int((top_left[0]+top_right[0])/2)

        pockets = [top_left, top_right, bot_left, bot_right, [mid, top_left[1]], [mid, bot_left[1]]]
        pockets = np.array(pockets)

        p_size = 60.1
        top_p_rect = [[mid-p_size, top_left[1]], [mid-p_size, top_left[1]+p_size], [mid+p_size,top_left[1]+p_size], [mid+p_size, top_left[1]]]
        bot_p_rect = [[mid-p_size, bot_left[1]], [mid-p_size, bot_left[1]-p_size], [mid+p_size,bot_left[1]-p_size], [mid+p_size, bot_left[1]]]
        # Connect top left and top right points
        cv2.line(img, (top_left[0], top_left[1]), (top_right[0], top_right[1]), (255, 0, 0), 2)
        # Connect top right and bottom right points
        cv2.line(img, (top_right[0], top_right[1]), (bot_right[0], bot_right[1]), (255, 0, 0), 2)
        # Connect bottom right and bottom left points
        cv2.line(img, (bot_right[0], bot_right[1]), (bot_left[0], bot_left[1]), (255, 0, 0), 2)
        # Connect bottom left and top left points
        cv2.line(img, (bot_left[0], bot_left[1]), (top_left[0], top_left[1]), (255, 0, 0), 2)

        # Threshold the image to get only white pixels
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

        # Create a filled polygon with the four corners to define the region of interest
        w_corners = np.array([w_rect], dtype=np.int32)
        h_corners = np.array([h_rect], dtype=np.int32)
        cv2.fillPoly(mask, w_corners, 255)
        cv2.fillPoly(mask, h_corners, 255)
        top_p_rect = np.array([top_p_rect], dtype=np.int32)
        bot_p_rect = np.array([bot_p_rect], dtype=np.int32)
        cv2.fillPoly(mask, top_p_rect, 0)
        cv2.fillPoly(mask, bot_p_rect, 0)


        # Apply the mask to the thresholding operation
        board_mask = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
        
        white_mask = cv2.inRange(board_mask, lower_que, upper_que)
        board_mask = cv2.inRange(board_mask, lower_field, upper_field)
        mask_inv = cv2.bitwise_not(board_mask, mask=mask)*4
        
        # Apply the Hough transform to detect straight lines
        lines = cv2.HoughLinesP(white_mask,rho = 1,theta = 1*np.pi/180,threshold = 50,minLineLength = 100,maxLineGap = 50)

        best_line = None
        if lines is not None:
            best_len = 0
            for line in lines:
                x1,y1,x2,y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > best_len:
                    best_len = length
                    best_line = (x1, y1, x2, y2)

            # Draw the best line on the image
            cv2.line(img, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (255, 255, 0),3)

        contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fit circles to the detected contours
        circles = 0
        circ_pos = []
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            if radius > 22 and radius < 35:
                cv2.circle(img,center,radius,(0,255,0),2)
                circles += 1
                circ_pos.append([np.array(center), radius])

        if circles < 2:
            gray = cv2.bitwise_and(img, img, mask=mask)
            ball = cv2.imread('green_ball.png')
            w_ball = cv2.imread('green_white_ball.png')
            for pattern in [ball, w_ball]:
                w, h = pattern.shape[:-1]
                res = cv2.matchTemplate(gray, pattern, cv2.TM_CCOEFF_NORMED)
                threshold = 0.3
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val >= threshold:
                    center = (max_loc[0] + w//2, max_loc[1] + h//2)
                    circles += 1
                    cv2.circle(img, center, 25, (0, 255, 0), 2)
                    circ_pos.append([np.array(center), 25])
                    break

        if circles == 2 and best_line is not None:
            #Calculate resulting direction of circle furthest away from line
            avg_cue = np.array([(best_line[2]+best_line[0])/2, (best_line[3]+best_line[1])/2])
            if np.linalg.norm(circ_pos[0][0] - avg_cue) < np.linalg.norm(circ_pos[1][0] - avg_cue):
                ball = circ_pos[0]
                target = circ_pos[1]
            else:
                ball = circ_pos[1]
                target = circ_pos[0]
            if np.linalg.norm(player - ball[0]) > 5 and np.linalg.norm(player - ball[0]) < 111:
                print("Locked in predictions")
            player = ball[0]
            cue_start = np.array([best_line[0], best_line[1]])
            cue_end = np.array([best_line[2], best_line[3]])
            if np.linalg.norm(cue_start-ball[0]) < np.linalg.norm(cue_end-ball[0]):
                temp = cue_end
                cue_end = cue_start
                cue_start = temp

            cue_dir = (cue_end - cue_start) / np.linalg.norm(cue_end - cue_start)
            v = ball[0] - cue_end
            proj = np.dot(v, cue_dir) * cue_dir
            ball = [cue_end+proj,ball[1]]

            dist_along_cue = np.dot(target[0] - ball[0], cue_dir) 
            if dist_along_cue < 1:
                collision_point = ball[0] + (dist_along_cue + ball[1] + target[1]) * cue_dir
            else:
                collision_point = ball[0] + (dist_along_cue - ball[1] - target[1]) * cue_dir
            cv2.line(img, (int(ball[0][0]), int(ball[0][1])), (int(collision_point[0]), int(collision_point[1])), (0, 0, 255), thickness=2)

            movement_dir = target[0] - collision_point
            intersection, normal = get_intersection(movement_dir, top_left, bot_right, target[0])
            hit = predict_in(pockets, intersection)
            if not hit:
                #See if the ball target ball can reflect off a surface
                perp = np.dot(movement_dir, normal) * normal
                reflection = movement_dir - 2 * perp 
                new_dir = reflection / np.linalg.norm(reflection)
                old_intersection=intersection
                intersection, normal = get_intersection(new_dir, top_left, bot_right, intersection-movement_dir*0.1)
                if predict_in(pockets, intersection):
                    cv2.circle(img, intersection, 25, (0, 255, 0), 2)
                    got_in = True
                cv2.line(img, (int(collision_point[0]), int(collision_point[1])), (int(old_intersection[0]), int(old_intersection[1])), (0, 255, 255), thickness=3)
                cv2.line(img, (int(old_intersection[0]), int(old_intersection[1])), (int(intersection[0]), int(intersection[1])), (0, 255, 255), thickness=3)
            else:
                cv2.circle(img, intersection, 25, (0, 255, 0), 2)
                got_in = True
                cv2.line(img, (int(collision_point[0]), int(collision_point[1])), (int(intersection[0]), int(intersection[1])), (0, 255, 255), thickness=3)
            if not got_in:
                cv2.circle(img, intersection, 25, (0, 0, 255), 2)
                text_size = cv2.getTextSize(text_out, font, scale, 2)[0]
                text_x = int((middle_x - text_size[0]) / 2)
                text_y = int((middle_y + text_size[1]) / 2)
                cv2.putText(img, text_out, (text_x, text_y), font, scale, color_out, 2)
                print("OUT")
            else:
                text_size = cv2.getTextSize(text_in, font, scale, 2)[0]
                text_x = int((middle_x - text_size[0]) / 2)
                text_y = int((middle_y + text_size[1]) / 2)
                cv2.putText(img, text_in, (text_x, text_y), font, scale, color_in, 2)
                print("IN")
            
            timeout = shot_time_outs[shot_nr]
            shot_nr += 1
            wait = True

        # Show the final image with all contours
        img = cv2.resize(img,(0,0),None,0.7,0.7)
        cv2.imshow("Window", img)
        if wait:
            for i in range(90):
                wait = False
                got_in = False
        cv2.waitKey(100)
    else:
        break
cap.release()
cv2.destroyAllWindows()  

# -------------------------------------------------------------------------------------------------------------------------------------- #