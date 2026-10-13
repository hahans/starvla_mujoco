#真机轨迹三条
轨迹路径都在 /Users/neow/Desktop/starvla_mujoco/real_raw2


#物理参数
1.红色方块到机械臂底座的距离为50cm，如图/Users/neow/Desktop/starvla_mujoco/starvla_mujoco/real_robot_image/1.jpg，/Users/neow/Desktop/starvla_mujoco/starvla_mujoco/real_robot_image/3.jpg
2.方块的边长都是5cm
3.黑色桌面的空洞之间为2.5cm
4.episode 0 中的红色方块最右边缘到桌面左边缘有16个空洞有42cm（以相机的视角的左边），绿色方块都是与红色方块平行，与桌面左边缘11个空洞，也就是红色和绿色方块中间会有有两个空洞，如图/Users/neow/Desktop/starvla_mujoco/starvla_mujoco/real_robot_image/2.jpg。三条轨迹的红色与绿色方块的起始位置都不同，其中ep0 ：9x16 ep1:8x16 ep2:8x12，16、16、12都是指红色方块到桌面左边缘（以相机的视角的左边），9、8、8表示到靠近机器人底座的黑色桌面边缘的空洞数量。
5.相机到黑色桌面的距离23cm左右
6.在ep0中相机到绿色方块边缘最短为97cm左右，如图/Users/neow/Desktop/starvla_mujoco/starvla_mujoco/real_robot_image/4.png
7.整体黑色桌面离地98cm
8.整体相机视角可以具体看数据集：/Users/neow/Desktop/starvla_mujoco/real_raw2/episode_0000/rgb/000000.png，用的是相机视角下左边的机械臂，然后机械臂的底座有一个45度斜角

