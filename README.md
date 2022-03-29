###
Capture文件夹为包含所有的数据

Capture/* 表示一次仿真的全部结果，包含红外 bmp 图像和一个 json 文件

json文件：image_width 与 image_height 代表图像的大小
         agents_labels 表示物体的详细信息
         color为label成像中物体的rgb值（此处用不到）
         position_in_image为物体在图像中的位置：x与y分别代表xmin与xmax，z与w分别代表ymin与ymax
         mask_file表示图像在该文件夹下的具体命名
         