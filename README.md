# HawkEye
This is the repository of our HawkEye project.

Weekly Progress 6/28 – 7/5 

这个星期主要架建了一个CNN去把inputted images localize 跟classify  成五个classes： drone， airplane， bird， helicopter， other. 这里面training Data 为 有airplane 的图片 以及 五个 outputs: 位置跟大小x,y,w,h, 以及  class number. 

 Image Resizing: 一开始要把1920 x1080 的input image resize 成1024 x 1024 的正方形， 一开始有考虑512x512 的正方形 但是图片里面的airplane 太小， 所以还是需要保持相对的大小。 

Data Source: 里面的data 取用了用multi-frame subtraction 找出的airplane location 跟size x, y, w, h 

CNN Architecture： 7 层CNN + maxpooling layers， 既然input image 比较大 可能需要加入更多 的Conv layers 

Training: 这次train 了150 个iteration 来overfit 2 张图片，用了CPU 来train 非常慢，因为input image 的关系 可能需要烧GPU~~~  

Next week improvement: 
1. 加入更多Conv 层数  以及FC layer
2. 加入不同class的training images 来ovefit

- Jiafu Wu 7/6
