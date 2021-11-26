# Military Vehicles Tracking

___
## Introduction
> This is the code for my project ___"Multiple Object Tracking for Military Vehicles"___, which is a part of my BSc in electrical engineering at Technion Israel. 
The project was done under the supervision of Gabi Davidov PhD - Thanks for his guidance and support durring the whole process.

The project can be divided into roughly 3 parts:

1. __Create a Custom Dataset__:
After looking online, one can see that a dataset for military vehicles detection can't be found.
Therefore, a custom dataset containing around 4000 images was collected and labeled for the project.

2. __Train an Object Detection Model__:
This project uses YOLOv5[[3]](#ref3) for object detection. As popular datasets used for training 
(such as COCO, ImageNet, etc.) have limited amount of military vehicles images, training an object on a custom dataset
is necessary.

3. __Combine with an Object Tracking Model__:
After obtaining the object detections (bounding box and class for the objects in each frame), the purpose of the tracking phase is to understand the relation between the objects over different frames. For this purpose, the DeepSort[[1]](#ref1) algorithm was chosen, with a pre-trained Pytorch implementation[[2]](#ref2).

