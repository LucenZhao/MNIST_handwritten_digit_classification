import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def extract_data(path):
   data_file = open(path).readlines()
   num_img = len(data_file)
   width = 16
   height = 16
   labels = np.zeros(num_img, dtype=np.int8)
   imgs = np.zeros((num_img, height, width), dtype=np.float32)
   for i, v in enumerate(data_file):
      data = np.fromstring(v, dtype=np.float32, sep=' ')
      labels[i] = int(data[0])
      img = data[1:].reshape(height, width)
      imgs[i] = img

   return labels, imgs


def visualize(labels, all_imgs):
   plt.figure(figsize=(10,3))
   gs = gridspec.GridSpec(3, 10, wspace=0.025, hspace=0.05) 
   
   for i in range(10):
      imgs = all_imgs[(labels == i),]
      np.random.seed(100)
      idx = np.random.permutation(len(imgs))
      for j in range(3):
         img = imgs[idx[j]]
         ax = plt.subplot(gs[j,i])
         plt.axis('off')
         ax.imshow(img, cmap='gray')
         ax.set_xticklabels([])
         ax.set_yticklabels([])
         ax.set_aspect('equal')
   plt.savefig("err_images.png", pad_inches = 0)
   plt.show()


if __name__ == '__main__':
   train_labels, train_imgs = extract_data('data/zip.train')

   # chain code
   
   epsilon = 0.00001
   import cv2
   directions = [1, 2, 3, 8, 4, 7, 6, 5]
   i_change = [-1, 0, 1, -1, 1, -1, 0, 1]
   j_change = [-1, -1, -1, 0, 0, 1, 1, 1]
   # 1 2 3
   # 8   4
   # 7 6 5
   def calc_chain(img):
      ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
      img = img.astype(dtype=np.uint8)

      _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      contours = np.array(contours)
   
      new_contours = contours[0].reshape(-1, 2)
      if len(contours) > 1:
         for contour in contours[1:]:
            contour = contour.reshape(-1, 2)
            new_contours = np.concatenate((new_contours, contour))
      contours = new_contours

      dirs = []
      cur_c = contours[0]
      for k, contour in enumerate(contours[1:]):
         for i, d in enumerate(directions):
            if (contour[0] == cur_c[0] + i_change[i]) and (contour[1] == cur_c[1] + j_change[i]):
               dirs.append(d)
         cur_c = contour
      contour = contours[0]
      for i, d in enumerate(directions):
         if (contour[0] == cur_c[0] + i_change[i]) and (contour[1] == cur_c[1] + j_change[i]):
            dirs.append(d)

      x = []
      y = []

      for val in contours:
         x.append(val[0])
         y.append(val[1])
      #plt.plot(contours[0][0], contours[0][1], 'bo')


      return x, y
   
   # visualize
   
   plt.figure(figsize=(10,5))
   gs = gridspec.GridSpec(5, 10, wspace=0.025, hspace=0.05) 
   
   for i in range(10):
      imgs = train_imgs[(train_labels == i),]
      np.random.seed(100)
      idx = np.random.permutation(len(imgs))
      for j in range(5):
         img = imgs[idx[j]]
         ax = plt.subplot(gs[j,i])
         plt.axis('off')
         ax.imshow(img, cmap='gray')
         x, y = calc_chain(img)
         ax.plot(x,y)
         ax.set_xticklabels([])
         ax.set_yticklabels([])
         ax.set_aspect('equal')
   plt.savefig("mnist_chain.png", pad_inches = 0)
   plt.show()
   
