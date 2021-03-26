# crf-image-labeling

Thanks to [snknitin's](https://github.com/snknitin) code of conditional random field model used in ocr, I wrote an api myself based on his code and extended it. So it is easily used for sequence labeling as shown in my code. </br>
</br>
The code is implemted in numpy and I divided it into functions and a running example. In crf_main.py, I used mini-batch sgd with nesterov momentum update and regularization, achieving the accuracy of 99% and 97% on training and testing sets, respectively, after 100 iterations. </br>
