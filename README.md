# SAR-Sidelobe-Suppression
SAR Sidelobe Suppression
旁瓣抑制中有一种邪门的方法是谱变形方法。通过将图像转化为二维频谱，然后对其进行切割，可以改变频谱方向。
![image](https://github.com/user-attachments/assets/e75adb09-a4f8-4cd9-b983-5100d1620c29)
如上图所示，只会改变旁瓣走向而不改变物体大致的形状。虽然纹理可能变了。
同时由于切割频谱导致主瓣展宽了。
为了减少切割带来的影响，故设计了如下网络
![image](https://github.com/user-attachments/assets/b957f4c1-77c7-42eb-89d3-931b569de918)
具体而言，就是通过变化检测检测出原图中变化的区域。即原图中旁瓣的区域。
然后因为谱变形后旁瓣被移走，所以原图中旁瓣的位置就实现了去除旁瓣的效果。
使用后处理方法将旁瓣所在位置的谱变形数据替换掉原图的数据，就实现了在原图上的旁瓣抑制。
这样就针对性的去处了SAR旁瓣。
在传统方法中改变了整张图片，而我这个方法只改变旁瓣。此乃本人纯原创，可以联系我挂个二作三作都行哈哈哈。
导师非要我弄去噪网络，说我这个不行。就不研究了。
代码我剔除了训练数据和谱变形代码，可以去看看王怀军的论文。很简单在GPT的帮助下你可以的。
