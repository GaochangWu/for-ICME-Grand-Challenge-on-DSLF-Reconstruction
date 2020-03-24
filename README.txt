0. Category or categories I am competing
	Category 1, Category 2 and Category 3


1. Run the code
	python mainx8.py
	python mainx16.py
	python mainx32.py
	
	*NOTE: (1) Since we use FROZEN model (.pd) in the tensorflow lib (python), the placeholder has a fixed shape 
	for each model, therefore, we have to use different .pd file for each iteration;
	(2) The shapes of the placeholders are fixed at:
	[None, 25, 1280, 1] and [None, 73, 1280, 1] for Cat1
	[None, 13, 1280, 1] and [None, 49, 1280, 1] for Cat2
	[None,  7, 1280, 1] ,   [None, 25, 1280, 1] and [None, 97, 1280, 1] for Cat3,
	If the angular resolution is not 193 and the image width is not 1280, please contact me.
	(3) You can set the parameter "EVALUATION" to "0", if you want evaluation the result on your own code.

2. Results for validation
	Cat 1:
	----------------------------------------------------------
	DD2
	Time(s):                                          792.36
	PSNR (min, RGB channel):                          36.976
	PSNR (averaged on synthesized view, RGB channel): 38.892
	----------------------------------------------------------
	DD3
	Time(s):                                          795.48
	PSNR (min, RGB channel):                          34.911
	PSNR (averaged on synthesized view, RGB channel): 36.883
	----------------------------------------------------------
	
	Cat 2:
	----------------------------------------------------------
	DD2
	Time(s):                                          441.62
	PSNR (min, RGB channel):                          35.777
	PSNR (averaged on synthesized view, RGB channel): 37.659
	----------------------------------------------------------
	DD3
	Time(s):                                          438.52
	PSNR (min, RGB channel):                          31.047
	PSNR (averaged on synthesized view, RGB channel): 34.628
	----------------------------------------------------------

	Cat 3:
	----------------------------------------------------------
	DD2
	Time(s):                                          1240.37
	PSNR (min, RGB channel):                          31.737
	PSNR (averaged on synthesized view, RGB channel): 34.114
	----------------------------------------------------------
	DD3
	Time(s):                                          1231.34
	PSNR (min, RGB channel):                          26.849
	PSNR (averaged on synthesized view, RGB channel): 30.903
	----------------------------------------------------------

3. Info about computer configuration I used
	. Memory 125GB, Unbuntu(64-bit), CPU Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, GPU TITAN XP 12GB


4. My contact info
	Gaochang Wu,                Yebin Liu                                  Fang Lu
	ahwgc2009@163.com           liuyebin@mail.tsinghua.edu.cn              fanglu@sz.tsinghua.edu.cn


5. Prerequisites for Deployment
	. Python 3.5.X
	. Required libraries: numpy, tensorflow 1.9.0 (or higher), glob, matplotlib


