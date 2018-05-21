import tensorflow as tf
import numpy as np
import cv2

x = tf.placeholder(tf.float32,[None,None,None,3],name='x')
lowcut = tf.Variable(tf.constant(0.005),name='lowcut')
highcut = tf.Variable(tf.constant(0.001),name='highcut')

RedHist = tf.histogram_fixed_width(x[:,:,:,0],[0.0,255.0],nbins=255)
GreenHist = tf.histogram_fixed_width(x[:,:,:,1],[0.0,255.0],nbins=255)
BlueHist = tf.histogram_fixed_width(x[:,:,:,2],[0.0,255.0],nbins=255)

PixelAmount = tf.cast(tf.reduce_min([tf.reduce_sum(RedHist),tf.reduce_sum(GreenHist),
                            tf.reduce_sum(BlueHist)]),tf.float32)

CumRed = tf.cast(tf.cumsum(RedHist,axis=0),tf.float32)
CumGreen = tf.cast(tf.cumsum(GreenHist,axis=0),tf.float32)
CumBlue = tf.cast(tf.cumsum(BlueHist,axis=0),tf.float32)

minR = tf.where(tf.cast(tf.add(tf.subtract(CumRed,tf.multiply(PixelAmount,lowcut)),
         tf.abs(tf.subtract(CumRed,tf.multiply(PixelAmount,lowcut)))),tf.bool))[0][0]
minG = tf.where(tf.cast(tf.add(tf.subtract(CumGreen,tf.multiply(PixelAmount,lowcut)),
         tf.abs(tf.subtract(CumGreen,tf.multiply(PixelAmount,lowcut)))),tf.bool))[0][0]
minB = tf.where(tf.cast(tf.add(tf.subtract(CumBlue,tf.multiply(PixelAmount,lowcut)),
         tf.abs(tf.subtract(CumBlue,tf.multiply(PixelAmount,lowcut)))),tf.bool))[0][0]
maxR = tf.where(tf.cast(tf.add(tf.subtract(CumRed,tf.multiply(PixelAmount,tf.subtract(1.0,highcut))),
         tf.abs(tf.subtract(CumRed,tf.multiply(PixelAmount,tf.subtract(1.0,highcut))))),tf.bool))[0][0]
maxG = tf.where(tf.cast(tf.add(tf.subtract(CumGreen,tf.multiply(PixelAmount,tf.subtract(1.0,highcut))),
         tf.abs(tf.subtract(CumGreen,tf.multiply(PixelAmount,tf.subtract(1.0,highcut))))),tf.bool))[0][0]
maxB = tf.where(tf.cast(tf.add(tf.subtract(CumBlue,tf.multiply(PixelAmount,tf.subtract(1.0,highcut))),
         tf.abs(tf.subtract(CumBlue,tf.multiply(PixelAmount,tf.subtract(1.0,highcut))))),tf.bool))[0][0]

RedRange = tf.concat([tf.zeros((tf.cast(tf.add(minR,1),tf.int32),)),
    tf.linspace(start=0.0,stop=255.0,num=tf.cast(tf.subtract(maxR,minR),tf.int32)),
    tf.cast(tf.multiply(255,tf.cast(tf.ones((tf.subtract(255,tf.cast(maxR,tf.int32)),)),tf.int32)),tf.float32)],0)
RedGather = tf.gather(RedRange,tf.cast(x,tf.int32)[:,:,:,0])

GreenRange = tf.concat([tf.zeros((tf.cast(tf.add(minG,1),tf.int32),)),
    tf.linspace(start=0.0,stop=255.0,num=tf.cast(tf.subtract(maxG,minG),tf.int32)),
    tf.cast(tf.multiply(255,tf.cast(tf.ones((tf.subtract(255,tf.cast(maxG,tf.int32)),)),tf.int32)),tf.float32)],0)
GreenGather = tf.gather(GreenRange,tf.cast(x,tf.int32)[:,:,:,1])

BlueRange = tf.concat([tf.zeros((tf.cast(tf.add(minB,1),tf.int32),)),
    tf.linspace(start=0.0,stop=255.0,num=tf.cast(tf.subtract(maxB,minB),tf.int32)),
    tf.cast(tf.multiply(255,tf.cast(tf.ones((tf.subtract(255,tf.cast(maxB,tf.int32)),)),tf.int32)),tf.float32)],0)
BlueGather = tf.gather(BlueRange,tf.cast(x,tf.int32)[:,:,:,2])


new_x = tf.stack([RedGather,GreenGather,BlueGather],axis=3)
new_x = tf.cast(new_x,tf.uint8)


with tf.Session() as sess:
    image = cv2.imread('1.jpg')
    image = image[np.newaxis,:,:,::-1]
    
    sess.run(tf.global_variables_initializer())
    
    result = sess.run(new_x,feed_dict={x:image})
    print(result)
    
    cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    cv2.imshow('test',result[0,:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
