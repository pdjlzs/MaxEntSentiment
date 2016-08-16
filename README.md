MaxEntropySentiment
======
MaxEntropySentiment is a package for sentiment analysis using neural networks based on package [LibN3L](https://github.com/SUTDNLP/LibN3L).  
Sentiment analysis aims to detect the sentiment polarity of a given sentence. 

Compile
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and compile it. 
* Open [CMakeLists.txt](CMakeLists.txt) and change "../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.  

`cmake .`  
`make`  

Input data format 
======
Each line contains one sentence and its labels, and they are seperated by space .  
