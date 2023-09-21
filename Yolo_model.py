from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)

    framewidth = int(cap.get(3))
    frameheight = int(cap.get(4))

    out = cv2.VideoWriter("YOLO running/Output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (framewidth, frameheight))


    model = YOLO('best.pt')

    classNames = ['-', 'ALL_MOTOR_VEHICLE_PROHIBITED', 'AXLE_LOAD_LIMIT', 'BARRIER_AHEAD', 'BUMPS', 'CATTLE', 'COMPULSARY_AHEAD_OR_TURN_LEFT', 'COMPULSARY_AHEAD_OR_TURN_RIGHT', 'COMPULSARY_CYCLE_TRACK', 'COMPULSARY_KEEP_LEFT', 'COMPULSARY_KEEP_RIGHT', 'COMPULSARY_MINIMUM_SPEED', 'COMPULSARY_SOUND_HORN', 'COMPULSARY_TURN_LEFT', 'COMPULSARY_TURN_LEFT_AHEAD', 'COMPULSARY_TURN_RIGHT', 'COMPULSARY_TURN_RIGHT_AHEAD', 'COMPULSORY_AHEAD', 'CROSS_ROAD', 'CYCLE_CROSSING', 'CYCLE_PROHIBITED', 'DANGEROUS_DIP', 'DIRECTION', 'FALLING_ROCKS', 'GAP_IN_MEDIAN', 'GIVE_WAY', 'GUARDED_LEVEL_CROSSING', 'HEIGHT_LIMIT', 'HORN_PROHIBITED', 'LEFT_HAIR_PIN_BEND', 'LEFT_HAND_CURVE', 'LEFT_REVERSE_BEND', 'LEFT_TURN_PROHIBITED', 'LENGTH_LIMIT', 'LOOSE_GRAVEL', 'MEN_AT_WORK', 'NARROW_BRIDGE', 'NARROW_ROAD_AHEAD', 'NO_ENTRY', 'NO_PARKING', 'NO_STOPPING_OR_STANDING', 'OBJECT_HAZARD', 'ONE_WAY_TRAFFIC', 'OVERTAKING_PROHIBITED', 'PASS_EITHER_SIDE', 'PEDESTRIAN_CROSSING', 'PEDESTRIAN_PROHIBITED', 'RESTRICTION_ENDS', 'RIGHT_HAIR_PIN_BEND', 'RIGHT_HAND_CURVE', 'RIGHT_REVERSE_BEND', 'RIGHT_TURN_PROHIBITED', 'ROUNDABOUT', 'SCHOOL_AHEAD', 'SIDE_ROAD_LEFT', 'SIDE_ROAD_RIGHT', 'SLIPPERY_ROAD', 'SPEED_LIMIT_100', 'SPEED_LIMIT_120', 'SPEED_LIMIT_15', 'SPEED_LIMIT_20', 'SPEED_LIMIT_30', 'SPEED_LIMIT_40', 'SPEED_LIMIT_5', 'SPEED_LIMIT_50', 'SPEED_LIMIT_60', 'SPEED_LIMIT_70', 'SPEED_LIMIT_80', 'SPEED_LIMIT_90', 'STAGGERED_INTERSECTION', 'STEEP_ASCENT', 'STEEP_DESCENT', 'STOP', 'STRAIGHT_PROHIBITED', 'TRAFFIC_SIGNAL', 'TRUCK_PROHIBITED', 'T_INTERSECTION', 'U_TURN_PROHIBITED', 'WIDTH_LIMIT', 'Y_INTERSECTION']

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                # print(x1, y1, x2, y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # print(box.conf[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                # print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        yield img
        # out.write(img)
        # cv2.imshow("Image", img)
        # if cv2.waitKey(1) & 0xFF==ord('1'):
        #     break
    # out.release()
cv2.destroyAllWindows()

