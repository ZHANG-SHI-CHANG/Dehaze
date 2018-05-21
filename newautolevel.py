# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import glob

class AutoLevel():
    def __init__(self):
        
        self.WindowName = 'dehaze'
        
    def linearmap(self,rlow,rhigh,glow,ghigh,blow,bhigh):
        
        map_list = np.zeros((1,256,3),np.uint8)
        for i in range(256):
            if i < blow:
                map_list[:,i,0] = 0
            elif i > bhigh:
                map_list[:,i,0] = 255
            else:
                map_list[:,i,0] = int(255*(((i-blow)*1.0)/((bhigh-blow)*1.0)))

            if i < glow:
                map_list[:,i,1] = 0
            elif i > ghigh:
                map_list[:,i,1] = 255
            else:
                map_list[:,i,1] = int(255*(((i-glow)*1.0)/((ghigh-glow)*1.0)))

            if i < rlow:
                map_list[:,i,2] = 0
            elif i > rhigh:
                map_list[:,i,2] = 255
            else:
                map_list[:,i,2] = int(255*(((i-rlow)*1.0)/((rhigh-rlow)*1.0)))
        
        return map_list
    
    def autolevel(self,img,lowcut = 0.005,highcut = 0.005):
        
        RedHist = cv2.calcHist([img[:,:,2]],[0],None,[256],[0.0,255.0])
        GreenHist = cv2.calcHist([img[:,:,1]],[0],None,[256],[0.0,255.0])
        BlueHist = cv2.calcHist([img[:,:,0]],[0],None,[256],[0.0,255.0])
        
        PixelAmount = min(RedHist.sum(),GreenHist.sum(),BlueHist.sum())
        
        CumRed = RedHist.cumsum()
        CumGreen = GreenHist.cumsum()
        CumBlue = BlueHist.cumsum()
        
        minR = np.nonzero(CumRed>=PixelAmount*lowcut)[0][0]+1
        minG = np.nonzero(CumGreen>=PixelAmount*lowcut)[0][0]+1
        minB = np.nonzero(CumBlue>=PixelAmount*lowcut)[0][0]+1
        maxR = np.nonzero(CumRed>=PixelAmount*(1-highcut))[0][0]+1
        maxG = np.nonzero(CumGreen>=PixelAmount*(1-highcut))[0][0]+1
        maxB = np.nonzero(CumBlue>=PixelAmount*(1-highcut))[0][0]+1
        
        allmap = self.linearmap(minR,maxR,minG,maxG,minB,maxB)
        
        im_result = cv2.LUT(img,allmap)
        
        return im_result
    
    def nop(self):
        pass
    
    def ReadImage(self,ImageName):
        return cv2.imread(ImageName)
    
    def StartFrame(self,Image):
        
        cv2.namedWindow(self.WindowName,cv2.WINDOW_NORMAL)
        cv2.createTrackbar('lowcut',self.WindowName,0,100,self.nop)
        cv2.createTrackbar('highcut',self.WindowName,0,250,self.nop)
        
        while True:
            lowcut = cv2.getTrackbarPos('lowcut',self.WindowName)/1000.0
            highcut = cv2.getTrackbarPos('highcut',self.WindowName)/1000.0
            
            img_result = self.autolevel(Image,lowcut,highcut)
            
            cv2.imshow(self.WindowName,img_result)
            
            k = cv2.waitKey(1)&0xff
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite('Result.jpg',img_result)
        
        cv2.destroyAllWindows()
    
    def StartFrameList(self,FrameListPath):
        cv2.namedWindow(self.WindowName,cv2.WINDOW_NORMAL)
        cv2.createTrackbar('lowcut',self.WindowName,0,100,self.nop)
        cv2.createTrackbar('highcut',self.WindowName,0,250,self.nop)
        
        Frames = glob.glob(FrameListPath)
        
        for i,fname in enumerate(Frames):
            lowcut = cv2.getTrackbarPos('lowcut',self.WindowName)/1000.0
            highcut = cv2.getTrackbarPos('highcut',self.WindowName)/1000.0
            
            Frame = self.ReadImage(fname)
#            Result = self.autolevel(Frame,lowcut,highcut)
            Result = self.autolevel(Frame,0.000005,0.013)
            cv2.imwrite('fishresult/%07d.jpg' % (i),Result)
            
            cv2.imshow(self.WindowName,Result)
            
            if cv2.waitKey(30)&0xff == ord('q'):
                break
            else:
                pass
        
        cv2.destroyAllWindows()
    
    def StartCapture(self,VideoName):
        
        self.Capture = cv2.VideoCapture(VideoName)
        
        cv2.namedWindow(self.WindowName,cv2.WINDOW_NORMAL)
        cv2.createTrackbar('lowcut',self.WindowName,0,100,self.nop)
        cv2.createTrackbar('highcut',self.WindowName,0,250,self.nop)
        
        success,frame = self.Capture.read()
        
        while not success:
            print('连接摄像头失败,尝试继续连接')
            success,frame = self.Capture.read()
            
        while success:
            lowcut = cv2.getTrackbarPos('lowcut',self.WindowName)/1000.0
            highcut = cv2.getTrackbarPos('highcut',self.WindowName)/1000.0
            
            StartTime = time.time()
            img_result = self.autolevel(frame,lowcut,highcut)
            print('总花费 %s ms.' % (time.time() - StartTime))
            
            cv2.imshow(self.WindowName,img_result)
            
            if cv2.waitKey(1)&0xff == ord('q'):
                break
            
            success,frame = self.Capture.read()
            
        self.Capture.release()
        
        cv2.destroyAllWindows()
    

def main_Frame():
    test = AutoLevel()
    test.StartFrame(test.ReadImage('5.jpg'))

def main_Video():
    test = AutoLevel()
    test.StartCapture('test.avi')

def main_FrameList():
    Path = 'fish/*.jpg'
    test = AutoLevel()
    test.StartFrameList(Path)

if __name__ == '__main__':
    main_Frame()
    #main_Video()
#    main_FrameList()















