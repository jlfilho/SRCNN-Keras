import cv2
import math
import imageio
import numpy as np


from PIL import Image
from subprocess import Popen, PIPE
from timeit import default_timer as timer


class VideoRestore():
    #def __init__(self):

    @staticmethod
    def scale_lr_imgs(imgs):
        """Scale low-res images prior to passing to SRGAN"""
        return imgs / 255.
        
    @staticmethod
    def unscale_hr_imgs(imgs):
        """Un-Scale high-res images"""
        #return (imgs + 1.) * 127.5
        return imgs * 255.
    
    def count_frames_manual(self,cap):
        count=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret):
                count +=1
            else:
                break
        return count
    
    def count_frames(self,cap):
        '''Count total frames in video'''
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total = self.count_frames_manual(cap)
        return total
    
    def sr_genarator(self,model,img_lr):
        """Predict sr frame given a LR frame"""
        # Predict high-resolution version (add batch dimension to image)
        img_sr = np.squeeze(
                    model.generator.predict(img_lr,
                        batch_size=1
                    ),
                    axis=0
                )
        # Remove batch dimension
        img_sr = self.unscale_hr_imgs(img_sr)
        return img_sr
       
    def write_srvideo(self, model=None,lr_videopath=None,sr_videopath=None,scale=None,print_frequency=30,crf=15):
        """Predict SR video given LR video """
        cap = cv2.VideoCapture(lr_videopath) 
        if cap.isOpened():
            fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
            # ffmpeg setup '-qscale', '5',
            p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(fps), '-i', '-', '-vcodec', 'libx264','-preset', 'veryslow', '-crf',str(crf), '-r', str(fps), sr_videopath], stdin=PIPE)
        else:
            print("Error to open low resolution video")
            return -1
        
        # Get video total frames
        t_frames = self.count_frames(cap)    
        #cria arquivo video hr if hr video is open
        count = 0
        time_elapsed = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start = timer()
                img_sr = self.sr_genarator(model,frame,scale=scale)
                end = timer()
                time_elapsed.append(end - start)
                im = Image.fromarray(img_sr.astype(np.uint8))
                im.save(p.stdin, 'JPEG')
                count +=1
            else:
                break
            if(count % print_frequency == 0):
                print('Time per Frame: '+str(np.mean(time_elapsed))+'s')
                print('Estimated time: '+str(np.mean(time_elapsed)*(t_frames-count)/60.)+'min')
        p.stdin.close()
        p.wait()
        cap.release()
        return time_elapsed
   


    def write_sr_images(self,model, input_images, output_images,scale):
      
        # Load the images to perform test on images
        imgs_lr, imgs_hr = self.load_batch(idx=0,img_paths=input_images, training=False)
        # Scale color values
        imgs_hr = self.unscale_hr_imgs(np.array(imgs_hr))
        imgs_lr = self.unscale_lr_imgs(np.array(imgs_lr)) 

        # Create super resolution images
        imgs_sr = []
        time_elapsed = []
        for img_lr,img_hr in zip(imgs_lr,imgs_hr):
            start = timer()
            img_sr = self.sr_genarator(model,img_lr)    
            end = timer()
            time_elapsed.append(end - start)   
            
            img_sr = Image.fromarray(img_sr.astype(np.uint8))
            img_sr.save(output_images.split(".")[0]+"SR.png")
            #imageio.imwrite(output_images.split(".")[0]+"SR.png", img_sr)
            
            img_hr = Image.fromarray(img_hr.astype(np.uint8))
            img_hr.save(output_images.split(".")[0]+"HR.png")
            #imageio.imwrite(output_images.split(".")[0]+"HR.png", img_hr)
            
            img_lr = Image.fromarray(img_lr.astype(np.uint8))
            img_lr.save(output_images.split(".")[0]+"LR.png")
            #imageio.imwrite(output_images.split(".")[0]+"LR.png", img_lr)
        return time_elapsed

    def sr_genarator(self,model,img_lr,scale):
        """Predict sr frame given a LR frame"""
        # Predict high-resolution version (add batch dimension to image)
        img_lr = cv2.resize(img_lr,(img_lr.shape[1]*scale,img_lr.shape[0]*scale), interpolation = cv2.INTER_CUBIC)
        img_lr=self.scale_lr_imgs(img_lr)
        img_sr = model.predict(np.expand_dims(img_lr, 0))
        # Remove batch dimension
        img_sr = img_sr.reshape(img_sr.shape[1], img_sr.shape[2], img_sr.shape[3])
        img_sr = self.unscale_hr_imgs(img_sr)
        return img_sr