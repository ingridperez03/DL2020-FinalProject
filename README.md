# From Sentiment Analysis to Emotion Recognition

Students:

	- Ingrid Pérez Aguilera				NIA: 205536 
  	  Email: ingrid.perez03@estudiant.upf.edu

	- Clara Reolid Sánchez				NIA: 207531 
  	  Email: clara.reolid01@estudiant.upf.edu 

	- Gerard Planell Bosch				NIA: 207533
  	  Email: gerard.planell01@estudiant.upf.edu 
	  
Practice group: 102

Group: 1

## Problem Description 

Sentiment analysis is the problem of analysing text data in order to identify and categorise opinions in order to determine whether the writer’s attitude towards a topic is positive, neutral and negative. Similar to Sentiment analysis, there is Emotional analysis which aims to detect and recognise feelings through the expressions of texts, such as anger, disgust, fear, happiness, sadness, and surprise.

The aim is to create a recurrent neural network to solve the sentiment analysis problem using a dataset that contains tweets labelled as either positive, neutral, or negative. Once the model has been trained, the aim is to adapt the architecture of the recurrent neural network so that it performs accurately the task of emotion analysis on another dataset of tweets.

## Data

The data can be obtained through the following link: https://drive.google.com/drive/folders/1VO0fUJd0Cm716MC8utI4Pvj72rYX0lgm?usp=sharing

Four datasets can be found in such file:

- Dataset 1: Sentiment Dataset 
[[1]: (http://alt.qcri.org/semeval2017/task4/index.php?id=results)]
	- Size: 50,337 tweets
	- Labels: 
		- Negative → 0
		- Neutral → 1
		- Positive → 2
	- Data Distribution: 
	
		![Image of Dataset1](Results/SA_BadData/SA_BadData_Dist.png)
	
- Dataset 2: Sentiment Tweets 
[[2]: (http://www.t4sa.it/)]

	- Size: 1,179,957 tweets
	- Labels: 
		- Negative → 0
		- Neutral → 1
		- Positive → 2
	- Data Distribution: ![Image of Dataset2]()

- Dataset 3: Emotion Dataset 
[[3]: (https://competitions.codalab.org/competitions/17751#learn_the_details-datasets)]
	- Size: 7,102 tweets
	- Labels: 
		- Sadness → 0
		- Fear → 1
		- Joy → 2
		- Anger → 3
	- Data Distribution: 
	
		![Image of Dataset3](Results/ER_BadData/ER_BadData_Dist.png)

- Dataset 4: Emotion Tweets 
[[4](https://github.com/omarsar/nlp_pytorch_tensorflow_notebooks)]

	- Size: 416,809 tweets
	- Labels: 
		- Sadness → 0
		- Joy → 1
		- Love → 2
		- Anger → 3
		- Fear → 4
		- Surprise → 5
	- Data Distribution: 
	
		![Image of Dataset4](Results/EmotionRecognition/ER_GoodData_Dist.png)
	
## Architectures



## Results
- Dataset 1: Sentiment Dataset 

	- Parameters:
		- Epochs → 10
		- Hidden Layer → 256
		- Embeddign → 400
		- Optimiser → Standard Gradient Descent (momentum 0.9)
		- Learning rate → 0.1
	- Results:

		|  Results  | Test Loss  | Accuracy |
		|  :-------:  | :----------: | :-------------: |
		| Network 1 |    0.933   | 60.7%  |
		| Network 2  | 0.935  | 60.5%  |
	- Confusion Matrix: 
		- Network 1:
		
			![Image of ConfusionMatrixD1_Network1](Results/SA_BadData/SA_BadData_ConfusionMatrix_Net1.png)
		- Network 2:
		
			![Image of ConfusionMatrixD1_Network2](Results/SA_BadData/SA_BadData_ConfusionMatrix_Net2.png)
		

- Dataset 2: Sentiment Tweets
	- Parameters:
		- Epochs → 5
		- Hidden Layer → 256
		- Embeddign → 400
		- Optimiser → Standard Gradient Descent (momentum 0.9)
		- Learning rate → 0.1
	- Results:

		|  Results  | Test Loss  | Accuracy |
		|  :-------:  | :----------: | :-------------: |
		| Network 1 |    0.596   | 95.4%  |
		| Network 2  | 0.599  | 96.1%  |
	- Confusion Matrix: 
		- Network 1:
		
			![Image of ConfusionMatrixD2_Network1]()
		- Network 2:
		
			![Image of ConfusionMatrixD2_Network2]()

- Dataset 3: Emotion Dataset 

	- Parameters:
		- Epochs → 10
		- Hidden Layer → 256
		- Embeddign → 400
		- Optimiser → Standard Gradient Descent (momentum 0.9)
		- Learning rate → 0.1
	- Results:

		|  Results  | Test Loss  | Accuracy |
		|  :-------:  | :----------: | :-------------: |
		| Network 1 |    1.226   | 44.2%  |
		| Network 2  | 1.097  | 60.6%  |	
	- Confusion Matrix: 
		- Network 1:
		
			![Image of ConfusionMatrixD3_Network1](Results/ER_BadData/ER_BadData_ConfusionMatrix_Net1.png)
		- Network 2:
		
			![Image of ConfusionMatrixD3_Network2](Results/ER_BadData/ER_BadData_ConfusionMatrix_Net2.png)

- Dataset 4: Emotion Tweets 

	- Parameters:
		- Epochs → 5
		- Hidden Layer → 256
		- Embeddign → 400
		- Optimiser → Standard Gradient Descent (momentum 0.9)
		- Learning rate → 0.1
	- Results:

		|  Results  | Test Loss  | Accuracy |
		|  :-------:  | :----------: | :-------------: |
		| Network 1 |    1.228   | 81.4%  |
		| Network 2  | 1.194  | 84.8%  |
	- Confusion Matrix: 
		- Network 1:
		
			![Image of ConfusionMatrixD4_Network1](Results/EmotionRecognition/ER_GoodData_ConfusionMatrix_Net1.png)
		- Network 2:
		
			![Image of ConfusionMatrixD4_Network2](Results/EmotionRecognition/ER_GoodData_ConfusionMatrix_Net2.png)
