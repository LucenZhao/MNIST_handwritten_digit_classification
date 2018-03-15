import numpy as np
import skimage.measure

def get_pose(coord):
   x = coord[0]
   y = coord[1]
   pose = 0
   if x > 4:
      pose = pose + 3
   if x > 10:
      pose = pose + 3
   if y > 4:
      pose = pose + 1
   if y > 10:
      pose = pose + 1
   return pose

class mnist_features:
   
   def __init__(self, img, plain, pool, hist, grad, chain):

      feats = np.array([], dtype=np.float32)
      # choose the types of features to be selected
      if plain == True:
         new_feats = self.get_plain(img)
         feats = np.concatenate((feats, new_feats))
      if pool['take'] == True:
         new_feats = self.get_pool(img, pool)
         feats = np.concatenate((feats, new_feats))
      if hist['take'] == True:
         new_feats = self.get_hist(img, hist)
         feats = np.concatenate((feats, new_feats))
      if grad['take'] == True:
         new_feats = self.get_grad(img, grad)
         feats = np.concatenate((feats, new_feats))
      if chain['take'] == True:
         new_feats = self.get_chain(img, chain)
         feats = np.concatenate((feats, new_feats))
      self.feats = feats

   def get_plain(self, img):
      feats = img.reshape(-1,)
      feats = (feats + 1) / 2
      return feats

   def get_pool(self, img, pool):
      pooled = np.array([])
      if pool['class'] == 'max':
         pooled = skimage.measure.block_reduce(img, (2,2), np.max)
      if pool['class'] == 'mean':
         pooled = skimage.measure.block_reduce(img, (2,2), np.mean)
      feats = pooled.reshape(-1,)
      feats = (feats + 1) / 2
      return feats

   def get_hist(self, img, hist):
      feats = []
      for k in range(len(hist['h'])):
         for i in range(0, 16-hist['h'][k]+1, 4):
            for j in range(0, 16-hist['w'][k]+1, 4):
               patch = img[i:i+hist['h'][k], j:j+hist['w'][k]].reshape(-1,)
               hist1 = sum(patch < 0)
               hist2 = hist['h'][k]*hist['w'][k] - hist1
               feats.append(hist1/hist['h'][k])
               feats.append(hist2/hist['h'][k])
      feats = np.array(feats)
      return feats

   def get_grad(self, img, grad):
      gx, gy = np.gradient(img)
      # print(gx)
      # print(max(gy), min())
      if grad['class'] == 'hist':
         hist_param = {'h': [4], 'w': [4]}
         feats_x = self.get_hist(gx, hist_param)
         feats_y = self.get_hist(gy, hist_param)
         feats = np.concatenate((feats_x, feats_y))
         return feats

      if grad['class'] == 'plain':
         feats_x = gx.reshape(-1,)
         feats_y = gy.reshape(-1,)
         feats = np.concatenate((feats_x, feats_y))
         return feats

      return

   def get_chain(self, img, chain):
      # process image
      epsilon = 0.00001
      import cv2
      ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
      img = img.astype(dtype=np.uint8)
      
      # obtain contours
      image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

      new_contours = contours[0].reshape(-1, 2)
      if len(contours) > 1:
         for contour in contours[1:]:
            contour = contour.reshape(-1, 2)
            new_contours = np.concatenate((new_contours, contour))
      contours = new_contours
      directions = [1, 2, 3, 8, 4, 7, 6, 5]
      i_change = [-1, 0, 1, -1, 1, -1, 0, 1]
      j_change = [-1, -1, -1, 0, 0, 1, 1, 1]
      # 1 2 3 #get direction (chain code)
      # 8   4
      # 7 6 5

      cur_c = contours[0]
      feats = np.zeros((9,8))
      for k, contour in enumerate(contours[1:]):
         for i, d in enumerate(directions):
            if (contour[0] == cur_c[0] + i_change[i]) and (contour[1] == cur_c[1] + j_change[i]):
               p = get_pose(cur_c)
               feats[p][d-1] = feats[p][d-1] + 1
         cur_c = contour
      contour = contours[0]
      for i, d in enumerate(directions):
         if (contour[0] == cur_c[0] + i_change[i]) and (contour[1] == cur_c[1] + j_change[i]):
            p = get_pose(cur_c)
            feats[p][d-1] = feats[p][d-1] + 1
      
      # get histogram

      feats = np.reshape(feats, (-1,))
      num = sum(feats)
      feats = feats / num * 8

      return feats





