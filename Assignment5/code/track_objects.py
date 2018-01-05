"""A very simple tracking algorithm.

For more serious tracking, look-up the papers in the projects pdf.
"""

import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt

FRAME_DIR = '../data/frames/'
DETECTION_DIR = '../data/detections/'


def compute_similarity(current_detections, next_detections,
                       current_image, next_image):
  """Compute similarity score for detections in adjacent frames.

  Args:
    current_detections: Detections in the current frame
    next_detections:    Detections in the next frame
    current_image:      PIL.Image object for the current frame
    next_image:         PIL.Image object for the next frame

  Returns:
    sim:                Similarity matrix.
  """
  num_current_detections = len(current_detections)
  num_next_detections = len(next_detections)
  sim = np.zeros((num_current_detections, num_next_detections))

  current_detection_areas = compute_area(current_detections)
  next_detection_areas = compute_area(next_detections)

  current_detection_centers = compute_center(current_detections)
  next_detection_centers = compute_center(next_detections)

  current_image = np.array(current_image)
  next_image = np.array(next_image)

  weights = [1, 1, 2]

  for detection_id in range(num_current_detections):
    # Compare boxes sizes
    current_box_area = current_detection_areas[detection_id]
    sim[detection_id] = (np.minimum(next_detection_areas, current_box_area)
                         / np.maximum(next_detection_areas, current_box_area)
                         * weights[0])

    # Penalize distance (would be good to look-up flow, but it's slow to
    # compute for images of this size.
    center_distances = (next_detection_centers
                        - current_detection_centers[detection_id])
    sim[detection_id] += np.exp(- 0.5 * np.sum(center_distances ** 2, axis=1)
                                / 5 ** 2) * weights[1]

    # Compute patch similarity
    current_patch = grab_image_patch(current_image,
                                     current_detections[detection_id])

    for next_detection_id in range(num_next_detections):
      distance = np.linalg.norm(current_detection_centers[detection_id]
                                - next_detection_centers[next_detection_id])
      if distance > 60:
        sim[detection_id, next_detection_id] = 0.
        continue

      next_patch = grab_image_patch(next_image,
                                    next_detections[next_detection_id],
                                    current_patch.shape[:2])

      sim[detection_id, next_detection_id] += (
          weights[2] * np.sum(current_patch * next_patch))
  return sim


def grab_image_patch(image, box, target_shape=None):
  """Get the image patch based on the given box.

  Args:
    image:          np.array object for input image.
    box:            Pixel coordinate for output patch.
    target_shape:   None or size-2 tuple for target patch shape. Output patch
                    will be resized to target shape if not set to None.

  Returns:
    patch:    np.array object for cropped image patch
  """
  box = np.round(box).astype(np.int32)

  # Python index starts at 0.
  box[:2] = np.maximum(1, box[:2]) - 1
  # xmax and ymax are exclusive.
  box[2] = min(box[2], image.shape[1])
  box[3] = min(box[3], image.shape[0])

  patch = image[box[1]:box[3], box[0]:box[2], :]

  if target_shape is not None:
    patch = scipy.misc.imresize(patch, target_shape)

  patch = patch.astype(np.double)
  patch /= np.linalg.norm(patch)

  return patch


def compute_area(detections):
  """Compute the area for each detection.

  Args:
    detections:   A matrix of shape N * 4, recording the pixel coordinate of
                  each detection bounding box (inclusive).

  Returns:
    area:         A vector of length N, representing the area for each input
                  bounding box.
  """
  area = (np.maximum(0, detections[:, 2] - detections[:, 0] + 1)
          * np.maximum(0, detections[:, 3] - detections[:, 1] + 1))
  return area


def compute_center(detections):
  """Compute the center for each detection.

  Args:
    detections:   A matrix of shape N * 4, recording the pixel coordinate of
                  each detection bounding box (inclusive).

  Returns:
    center:       A matrix of shape N I 2, representing the (x, y) coordinate
                  for each input bounding box center.
  """
  center = detections[:, [0, 1]] + detections[:, [2, 3]]
  center /= 2.
  return center


def load_image_and_detections(frame_id):
  """Helper function for loading image and detections.
  """
  image_path = os.path.join(FRAME_DIR, '%06d.jpg' % frame_id)
  image = Image.open(image_path)

  mat_path = os.path.join(DETECTION_DIR, '%06d_dets.mat' % frame_id)
  mat_content = scipy.io.loadmat(mat_path)
  detections = mat_content['dets']
  return image, detections




def get_tracks():

  start_frame = 62
  end_frame = 72

  sims = []
  tracks = []
  for frame_id in range(start_frame, end_frame):
    print("frame_id is: {}".format(frame_id))
    current_image, current_detections = load_image_and_detections(frame_id)

    next_image, next_detections = load_image_and_detections(frame_id + 1)

    # sim has as many rows as len(current_detections) and as many columns as
    # len(next_detections).
    # sim[k, t] is the similarity between detection k in frame i, and detection
    # t in frame j.
    # sim[k, t] == 0 indicates that k and t should probably not be the same track.
    sim = compute_similarity(current_detections, next_detections,
                            current_image, next_image)


    current_detections[:, 4] = frame_id
    next_detections[:, 4] = frame_id + 1


    for k in range(sim.shape[0]):
      most_similar_index = np.argmax(sim[k])
      current_similar_detection_index = k
      next_similar_detection_index = most_similar_index
      print("current_similar_detection_index is: {}".format(k))
      print("next similar detection index is: {}".format(next_similar_detection_index))
      track = [current_detections[current_similar_detection_index], next_detections[next_similar_detection_index]]
      # if tracks list not empty:
      exist_flag = False
      if not len(tracks) == 0:
        previous_count = 0
        for previous_track in tracks:
          previous_count += 1
          if exist_flag == False:
          # if previous truck's last detection id is same as current track's first detection id and bounding box is same
            if \
                        previous_track[-1][0] == track[0][0] \
                    and previous_track[-1][1] == track[0][1] \
                    and previous_track[-1][2] == track[0][2] \
                    and previous_track[-1][3] == track[0][3] \
                    and previous_track[-1][4] == track[0][4]:
              # append the new detection to previous list
              previous_track.append(track[1])
              exist_flag = True
              print("previous count is: {}".format(previous_count))

      # if no previous detection can be tracked
      if exist_flag == False:
        tracks.append(track)

      sims.append(sim)
  print("done!")
  return tracks, sims


def get_desired_index(tracks):
  indices_list = []
  for i in range(len(tracks)):
    if len(tracks[i]) > 5:
      indices_list.append(i)
  return indices_list


def pre_processing(tracks, indices_list):
  images = {}
  selected_tracks = []
  color_list = [0, 1, 2, 3, 4, 5]
  for i in range(len(indices_list)):
    desired_track = tracks[indices_list[i]]
    desired_track = np.reshape(np.array(desired_track), (-1, 6))
    desired_track[:, 5] = color_list[i]
    selected_tracks.append(desired_track)



  id_bl = []
  for i in range(len(selected_tracks)):
    for detector in selected_tracks[i]:
      left, top, right, bottom, id,  color = detector[0], detector[1], detector[2], detector[3], int(detector[4]), detector[5]
      if id not in id_bl:
        images["{}".format(id)] = [[left, top, right, bottom, color]]
        id_bl.append(id)
      else:
        images["{}".format(id)] += [[left, top, right, bottom, color]]


  return  images




def visualization(images, path):
    # color_list = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "gold", "silver", "magenta",
    #             "cyan"]

    keys = list(images.keys())
    for i in range(len(keys)):
      id = "0000{}.jpg".format(keys[i])
      content = images[keys[i]]
      content = np.reshape(np.array(content), (-1, 5))
      image = Image.open("../data/frames/{}".format(id))
      new_path = path + "/{}".format(id)
      # color = color_list[i]
      showboxes(image, content, output_figure_path=new_path)



def showboxes(image, boxes, output_figure_path=None):
  """Draw bounding boxes on top of an image.

  Args:
    image:               PIL.Image object
    boxes:               A N * 4 matrix for box coordinate.
    output_figure_path:  String or None. The figure will be saved to
                         output_figure_path if not None.
  """
  color_list = ["red", "blue",  "yellow", "orange", "purple", "gold", "silver", "magenta",
                "cyan", "black", "white", "pink"]
  figure = plt.figure()
  axis = figure.add_subplot(111, aspect='equal')
  plt.imshow(image)
  # color = "yellow"
  for box in boxes:
    axis.add_patch(patches.Rectangle(box[:2],
                                     box[2] - box[0],
                                     box[3] - box[1],
                                     fill=None,
                                     ec=color_list[box[4].astype(int)],
                                     lw=2))

  if output_figure_path is not None:
    plt.savefig(output_figure_path)





def main():
  tracks, sims = get_tracks()
  indices_list = get_desired_index(tracks)
  images_dict = pre_processing(tracks, indices_list)
  visualization(images_dict, "../output")


if __name__ == "__main__":

  main()







