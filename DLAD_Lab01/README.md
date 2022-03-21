# DLAD Lab 01
!["Exaples"](https://github.com/YanxiangDing/DLAD2021_Lab/blob/main/DLAD_Lab01/figures/Fig01.png)

## Instruction
In the whole data, there are 13 types of pictures, and each type of figure shows the number in chinese.

## Label the data!!!
* 0: Arabic number 0 written in chinese.
* 1: Arabic number 1 written in chinese.
* 2: Arabic number 2 written in chinese.
* 3: Arabic number 3 written in chinese.
* 4: Arabic number 4 written in chinese.
* 5: Arabic number 5 written in chinese.
* 6: Arabic number 6 written in chinese.
* 7: Arabic number 7 written in chinese.
* 8: Arabic number 8 written in chinese.
* 9: Arabic number 9 written in chinese.
* 10: Arabic number 10 written in chinese.
* 11: Arabic number 100 written in chinese.
* 12: Arabic number 1000 written in chinese.

## Talk about the model
In this work, I implemented a classifier with four fully-connected layers. The output layer applied softmax as activation function, and used CrossEntropyLoss as the cost function.
