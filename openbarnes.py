import cv2
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime
import time

class Maze(object):
    video_file = None
    distance_traveled = 0
    video = None
    df = None
    _first_frame = None
    mouse_coordinates = []
    frame_count = 0

    def __init__(self, video_file=None, output_file=None):
        if not video_file:
            raise BaseException('Video file must be specified')
        elif not os.path.isfile(video_file):
            raise BaseException('File does not exist')

        self.video_file = video_file
        self.video = cv2.VideoCapture(self.video_file)
        self.output_file = output_file

        video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps       = self.video.get(cv2.CAP_PROP_FPS)
        #print(video_fps)
        self.video_writer = cv2.VideoWriter(self.output_file, video_FourCC, video_fps, (600, 1600))

        self.first_frame()

    def first_frame(self):
        if self._first_frame is None:
            ret, self._first_frame = self.video.read()
            self.frame_count += 1

        return self._first_frame

    def next_frame(self):
        ret, frame = self.video.read()
        return frame

    #def write_frame(self):
#        self.output_video

    def detect(self):
        pass

class BarnesMaze(Maze):
    thresh_min = 175
    thresh_max = 255
    hole_count = 20
    rotation_offset = -90
    plot_circles = []
    perspective = None
    warped_frame = None
    circle_centers = []
    reset_frame_count = 0
    radius_cm = 46
    mouse_down = False
    manual_coordinates = []
    previous_distance = 0
    slow = False
    last_frame = None
    video_finished = False

    def set_manual_coordinates(self, event, x, y, flags, param):
        if event == 1:
            self.mouse_down = True
        if event == 4:
            self.mouse_down = False

        if self.mouse_down:
            print(x, y)
            self.manual_coordinates.append((x, y))

    def closest_node(self, node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index]

    def plot(self):
        print('Called plot()')
        h, w, _ = self._first_frame.shape
        sns.set(rc={'figure.figsize':(w / 100, h / 100)})
        sns.set_style("whitegrid")

        fig=plt.figure()
        ax=fig.add_subplot(1, 1, 1)
        #print(self._first_frame.shape)
        ax.set(ylim=(h, 1), xlim=(1, w))
        #for c in self.plot_circles:
        #    x, y, r = c
        #    circle = plt.Circle((x, y), r, color="black", fill=False)
        #    ax.add_patch(circle)

        prev_coordinates = None
        for c in self.mouse_coordinates:
            x, y = c
            #circle = plt.Circle((x, y), 1, color="blue", fill=True)
            #ax.add_patch(circle)

            if prev_coordinates is None:
                prev_coordinates = (x, y)
                continue

            xmin, ymin = prev_coordinates

            l = mlines.Line2D([xmin, x], [ymin, y], color='blue', linewidth=4, linestyle='-')
            ax.add_line(l)
            prev_coordinates = (x, y)

        for c in self.plot_circles:
            x, y, r = c
            circle = plt.Circle((x, y), r, color="black", fill=False)
            ax.add_patch(circle)


        sns.despine(left=True, bottom=True)
        ax.set(yticks=[], xticks=[])
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #sns.pointplot(data=self.df, join=True, color='red', ax=ax, zorder=1)
        print('Saving to plot.svg...')
        plt.savefig("plot.svg", format="svg", bbox_inches='tight')
        return plt

    def scan(self):
        height, width = self._first_frame.shape[:2]
        roi_mask = np.zeros((height, width), np.uint8)
        (x, y, r) = self.plot_circles[0]
        cv2.circle(roi_mask, (int(x), int(y)), int(r) - 15, 255, thickness=-1, lineType=cv2.LINE_AA)
        plot_centers = [(x, y) for (x,y,r) in self.plot_circles]
        corrected_frame = None
        last_frame = None

        while True:
            ret, frame = self.video.read()
            if self.slow:
                time.sleep(1)
            #print('.')
            if not ret:
                frame = last_frame.copy()
                self.video_finished = True
            else:
                last_frame = frame.copy()
                #break
            if self.video_finished is False:
                self.frame_count += 1
            dst = cv2.warpPerspective(frame, self.perspective, (width, height))
            corrected_frame = dst.copy()

            #dst = frame.copy()
            #print(len(self.plot_circles))
            for i, (x, y, r) in enumerate(self.plot_circles):
                # This should be based on i, not the radius.
                if int(r) == 15:
                    cv2.circle(dst, (int(x), int(y)), 15, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(dst, (int(x), int(y)), int(r), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                else:
                    # Recheck why I subtracted 15 here.
                    cv2.circle(dst, (int(x), int(y)), int(r), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                font = cv2.FONT_HERSHEY_SIMPLEX

                if i == 0:
                    continue

                if i >= 10:
                    cv2.putText(dst, '%d' % (i), (int(x)-9, int(y)+5), font, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(dst, '%d' % (i), (int(x)-5, int(y)+5), font, 0.47, (255, 255, 255), 1, cv2.LINE_AA)


            roi = cv2.bitwise_and(dst, dst, mask = roi_mask)
            roi = dst.copy()

            frame_gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(frame_gray, 125, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            for i, c in enumerate(contours):
                if i == 0:
                    continue
                if cv2.contourArea(c) <= 25:
                    continue
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                x2, y2 = self.closest_node((x, y), plot_centers)
                dist = distance.cdist([[x, y]], [[x2, y2]])

                if dist <= 10:
                    continue

                #cv2.circle(roi, (int(x), int(y)), 5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

                extLeft = c[c[:, :, 0].argmin()][0]
                extRight = c[c[:, :, 0].argmax()][0]
                extTop = c[c[:, :, 1].argmin()][0]
                extBottom = c[c[:, :, 1].argmax()][0]

                #cv2.drawContours(roi, [c], 0, (255, 0, 0), 2, lineType=cv2.LINE_AA)
                # Find nearest hole and draw a line
                x2, y2 = self.closest_node((x, y), plot_centers[1:])

                if distance.cdist([extLeft], [[x2, y2]]) <= 30 or distance.cdist([extRight], [[x2, y2]]) <= 30 or \
                    distance.cdist([extTop], [[x2, y2]]) <= 30 or distance.cdist([extBottom], [[x2, y2]]) <= 30:

                    cv2.circle(roi, (int(x2), int(y2)), 15, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                self.mouse_coordinates.append((x, y))

                # This can be disabled for performance
            prev_coordinates = None
            self.distance_traveled = self.previous_distance
            for (x, y) in self.mouse_coordinates:
                if prev_coordinates is None:
                    prev_coordinates = (int(x), int(y))
                    continue
                #self.distance_traveled += distance.euclidean(prev_coordinates, (x,y))
                if distance.cdist([list(prev_coordinates)], [[x, y]]) <= 120:
                    cv2.line(roi, (int(x), int(y)), prev_coordinates, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                #else:
                #    cv2.line(roi, (int(x), int(y)), prev_coordinates, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                #else:
                #    cv2.line(roi, (int(x), int(y)), prev_coordinates, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                prev_coordinates = (int(x), int(y))

                #cv2.imshow('preview', roi)
                #key = cv2.waitKey(1) & 0xFF
            #print('write frame')

            prev_manual_coordinates = None#(self.manual_coordinates[0][0], self.manual_coordinates

            for i in range(1, len(self.manual_coordinates)):
                if prev_manual_coordinates is None:
                    prev_manual_coordinates = self.manual_coordinates[0]
                x, y = self.manual_coordinates[i]
                self.distance_traveled += distance.euclidean(prev_manual_coordinates, (x, y))
                cv2.line(roi, (int(x), int(y)), prev_manual_coordinates, color=(35, 94, 240), thickness=2, lineType=cv2.LINE_AA)
                prev_manual_coordinates = (int(x), int(y))
            #print('Distance:', self.distance_traveled * self.radius_cm / self.maze_radius)

            seconds = int((self.frame_count - self.reset_frame_count)/ 23.98)
            timestamp = str(datetime.timedelta(seconds=seconds))
            cv2.putText(roi, timestamp, (10, 25), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            distance_multiplier = self.radius_cm / self.maze_radius
            distance_traveled_cm = self.distance_traveled * distance_multiplier
            #cv2.putText(roi, "%d cm" % (distance_traveled_cm), (10, 55), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            #self.video_writer.write(np.hstack([frame, dst]))
            #cv2.imshow('preview', np.hstack([frame, dst]))
            cv2.namedWindow("OpenBarnes")
            cv2.setMouseCallback("OpenBarnes", self.set_manual_coordinates)
            cv2.imwrite('visualize-%03d.jpg' % self.frame_count, roi)
            cv2.imshow('OpenBarnes', roi)
        #self.df = pd.DataFrame(self.mouse_coordinates)

            #return dst
            #cv2.imshow('preview', dst)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                #cv2.destroyAllWindows()
                #self.plot_circles = []
                break
            elif key == ord('R'):
                self.reset_frame_count = self.frame_count
            elif key == ord('r'):
                #self.reset_frame_count = self.frame_count
                self.mouse_coordinates = []
            elif key == ord('s'):
                if not self.slow:
                    self.slow = True
                else:
                    self.slow = False
            elif key == ord('d'):
                self.manual_coordinates = []
                self.previous_distance = self.distance_traveled

        #cv2.destroyAllWindows()

    def detect_correct(self):
        frame_gray = cv2.cvtColor(self.warped_frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame_gray, self.thresh_min, self.thresh_max, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for i, contour in enumerate(contours):
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            #circle_centers.append((x, y))
            #circle_centers[0] = (x, y)
            self.plot_circles[0] = (x, y, radius)
            self.maze_radius = radius
            break

    def detect(self):
        #if self.warped_frame is not None:
            #self._first_frame = self.warped_frame.copy()
            #self.plot_circles = []
            #self.perspective = None
            #self.warped_frame = None

        first_frame_copy = self._first_frame.copy()
        frame_gray = cv2.cvtColor(self._first_frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame_gray, self.thresh_min, self.thresh_max, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        height, width = self._first_frame.shape[:2]
        white_image = cv2.bitwise_not(np.zeros((height, width), np.uint8))
        white_image = cv2.cvtColor(white_image, cv2.COLOR_GRAY2BGR)


        #total_radius = 0

        for i, contour in enumerate(contours):
            if i == 111:
                continue
            area = cv2.contourArea(contour)
            if area  / (height * width) >= 0.80:
                continue
            if area < 100:
                continue
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            #total
            self.circle_centers.append((x, y))
            #cv2.drawContours(white_image, contour, -1, (255, 0, 0), 1)

            #cv2.circle(white_image, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            #cv2.circle(self._first_frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

            #cv2.line(white_image, (3, 3), (9, 9), color=(0, 0, 255), thickness=2)

        #x,y,w,h = cv2.boundingRect(contours[0])
        #x-= 20
        #y-= 20

        ((x, y), radius) = cv2.minEnclosingCircle(contours[0])

        self.plot_circles.append((x, y, radius))

        angle = 360 / self.hole_count

        pts_src = []
        pts_dst = []
        #h, status = cv2.findHomography(pts_src, pts_dst)

        for i in range(self.hole_count):
            theta = ((angle * i) + self.rotation_offset) * np.pi / 180
            x2 = x + radius * np.cos(theta)
            y2 = y + radius * np.sin(theta)
            cv2.line(self._first_frame, (int(x), int(y)), (int(x2), int(y2)), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            x2 = x + (radius - 43) * np.cos(theta)
            y2 = y + (radius - 43) * np.sin(theta)

            #Ideal Circle
            cv2.circle(self._first_frame, (int(x2), int(y2)), 15, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            self.plot_circles.append((x2, y2, 15))
            #pts_src.append([int(x2), int(y2)])

            #Actual circle
            x3, y3 = self.closest_node((x2, y2), self.circle_centers)

            if distance.cdist([[x2, y2]], [[x3, y3]]) <= 25:
                #cv2.circle(dst, (int(x3), int(y3)), 15, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(self._first_frame, (int(x3), int(y3)), 15, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                pts_src.append([int(x2), int(y2)])
                pts_dst.append([int(x3), int(y3)])


            #Actual Circle


        #print(pts_src)
        #print(pts_dst)

        #print(len(pts_src))

        src = np.array([pts_src[0], pts_src[5], pts_src[10], pts_src[15]]).astype(np.float32)
        dst = np.array([pts_dst[0], pts_dst[5], pts_dst[10], pts_dst[15]]).astype(np.float32)

        #src = np.array([pts_src[0], pts_src[2], pts_src[4], pts_src[5]]).astype(np.float32)
        #dst = np.array([pts_dst[0], pts_dst[2], pts_dst[4], pts_dst[5]]).astype(np.float32)

        #pts_src.reshape(-1,1,2).astype(np.float32)
        #pts_dst.reshape(-1,1,2).astype(np.float32)

        #affine_matrix = cv2.getAffineTransform(pts_src, pts_dst)

        #print(h, status)

        #warped = cv2.warpPerspective(self._first_frame, h, (width+250, height+250))

        #warped = cv2.warpAffine(self._first_frame, affine_matrix, (width, height))

        #cv2.circle(first_frame_copy, (int(x), int(y)), int(radius), (0, 255, 0), 4)
        #with_lines = self._first_frame.copy()
        #self._first_frame = first_frame_copy.copy()

        #pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
        #pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

        self.perspective = cv2.getPerspectiveTransform(dst, src)
        #if self.warped_frame is None:
        dst = cv2.warpPerspective(first_frame_copy, self.perspective, (width, height))
        self.warped_frame = dst.copy()

        cv2.circle(dst, (int(x), int(y)), int(radius) - 15, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

        for i in range(self.hole_count):
            theta = ((angle * i) + self.rotation_offset) * np.pi / 180

            x2 = x + (radius-15) * np.cos(theta)
            y2 = y + (radius-15) * np.sin(theta)

            cv2.line(dst, (int(x), int(y)), (int(x2), int(y2)), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            x2 = x + (radius - 43) * np.cos(theta)
            y2 = y + (radius - 43) * np.sin(theta)

            #Ideal Circle
            cv2.circle(dst, (int(x2), int(y2)), 15, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(dst, (int(x2), int(y2)), 15, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if i >= 9:
                cv2.putText(dst, '%d' % (i+1), (int(x2)-9, int(y2)+5), font, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(dst, '%d' % (i+1), (int(x2)-5, int(y2)+5), font, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
            #pts_src.append([int(x2), int(y2)])
        #guides = self._first_frame.copy()
        #self._first_frame = dst.copy()


        #for i in range(self.hole_count):
        #    cv2.line(self.warped_frame, (int(x), int(y)), (9, 9), color=(0, 255, 0), thickness=4)

        # Not all the circles may have been detected due to reflections and distortions.
        # We will attempt to use the partial detections to fill in the rest.
        #return derp
        #return dst

        #x,y,w,h = cv2.boundingRect(contours[0])
        #ROI = image[y:y+h, x:x+w]
        #return self._first_frame[y:y+h, x:x+w]
        #return self.warped_frame
        #return dst
        return np.hstack([self._first_frame, dst])
        #return np.hstack([self._first_frame[y:y+h, x:x+w], dst[y:y+h, x:x+w]])
