#### **Week 1**

**Paper title**: Rojas-Barahona, LM. Deep learning for sentiment analysis, *Lang Linguist Compass* 2016; 10: 701– 719. doi: [10.1111/lnc3.12228](https://doi.org/10.1111/lnc3.12228).

Roles:

* **Scientific Peer Reviewer**:
  * the paper presents an overview of deep learning approaches used for sentiment analysis. The overview of used techniques includes Feed-forward neural networks (FFNN), convolutional neural networks (CNN), and recurrent neural networks (RNN). They also use these neural architectures to encode information using sentence structure or some tree structure, e.g. constituency trees.
  * Task: sentiment analysis is a task where a given model needs to classify whether its overall message is **negative** or **positive.**
  * Model(s): multiple neural approaches that use either CNNs or RNNs and encode the sentences using pre-trained word vectors
* Experiments: Datasets: Movie Reviews & Sentiment TreeBank
  * Sentiment Tree Bank (Sentence: "This is n't a new idea", Sentiment: 😒 )
  * ![image (8).png](text://image?imageFileName=image%20%288%29.png)
  * Movie Reviews ("Not the best plot I have expected", sentiment: 😐)
  * Results
  * ![image (6).png](text://image?imageFileName=image%20%286%29.png)![image (7).png](text://image?imageFileName=image%20%287%29.png)![image (9).png](text://image?imageFileName=image%20%289%29.png)
* Conclusion: 1) review of sentiment analysis architectures that use neural networks, 2) comparison of models using three popular datasets (Sentiment Treebank, )
* **Archaeologist**:
  * **Previous work**:
  * ![image (10).png](text://image?imageFileName=image%20%2810%29.png)Sara Rosenthal, Alan Ritter, Preslav Nakov, and Veselin Stoyanov. 2014. [SemEval-2014 Task 9: Sentiment Analysis in Twitter](https://aclanthology.org/S14-2009). In *Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014)*, pages 73–80, Dublin, Ireland. Association for Computational Linguistics.
    * TLDR: An overview of sentiment analysis methods on the challenge dataset
      * Highlights: preprocessing is an important task for dealing with tweets (noisy text)
  * **Future work**: Dang, N.C., Moreno-García, M.N. and De la Prieta, F., 2020. Sentiment analysis based on deep learning: A comparative study. *Electronics*, *9*(3), p.483.
    * TLDR: An overview of all sentiment analysis methods up to 2020, seven datasets, methods for encoding text data (TF-IDF, word vectors)
* **Academic Researcher**:
  * Domain-specific application
    * It would more interesting to see the effect of these methods on other domains
      * For ex: the model trained on movie reviews applied to -> food reviews
      * 
  * Other languages:
    * Do similar analysis on other languages
    * How can we transfer the learned English models to other languages?
* **Industry Practitioner**:
  * You are a data scientist at a car company and want to analyze which specific models cause more problems and what people are saying about which car and what is problematic
  * Method: get all tweets mentioning your car models, apply sentiment analysis and find out which features are broken (negative) or working amazing (positive)
* **Private Investigator**
* Author#1: Lina Maria Rojas‐Barahona
  * Work at the time of publication: Cambridge University, Research Associate
  * Previous works: Postdoc (Université de Lorraine, CNRS - Centre national de la recherche scientifique), Temporary professor (Université de Lorraine)
  * Current work: Research Scientist at Orange (Media Company from France)
  * GScholar: <https://scholar.google.fr/citations?user=n42dh0cAAAAJ&hl=en>
    * No previous or future papers about sentiment analysis (before or after 2016).
    * A new survey paper about Dialogue State Tracking from 2022
* **Social Impact Assessor**
  * Positive impact on the world:
    * Nice overview for future research
    * Makes it easier for future researchers to start from a certain point and not doing anything redundant
  * Negative impact
    * Bias in the domains: the models are trained only on movie, tweets
      * Application of such models on unseen domains might cause problems
        * Automatic sentiment analysis on user reviews
  * 

#### **Week 2**

Paper title:  Simone Conia, Andrea Bacciu, Roberto Navigli "Unifying Cross-Lingual Semantic Role Labeling with Heterogeneous Linguistic Resources", NAACL 2021, <https://aclanthology.org/2021.naacl-main.31/>

Roles:

* Scientific Peer Reviewer
  * Task: Semantic role labeling - cross-lingual
  * Dataset: CONNL 2009 with 6 langs, PDT-Vallex, English Propbank, AnCora, Chinese Propbank, VerbAtlas
  * Method: Shared represenatation across languages that is based on husing word represetations....
  * Experiments:
  * Recommendation
* Archaeologist
  * Previous work
  * Future work
* Academic Researcher \*
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

#### **Week 4**

Paper title: Abigail See, Christopher Manning "Understanding and predicting user dissatisfaction in a neural generative chatbot" , (See & Manning, SIGDIAL 2021) <https://aclanthology.org/2021.sigdial-1.1>

Roles:

* Scientific Peer Reviewer
  * Task: **Predict the user's dissatisfaction before it occurs** during the conversation between a user and a chatbot. Therefore the likelihood of low-quality utterances by the bot is reduced.![Task.png](text://image?imageFileName=Task.png)
  * Dataset: **NeuralChatTurns** (not publicly released). They use the conversations from their chatbot CHIRPY (2nd place in Alexa Prize) as data.

    data form: **(c, b, u)**

    b: a purely neural-generated bot utterance.

    c: the Neural Chat context that preceded b.

    u: user response to b.
  * Data Annotation:
    1. Detect User Dissatisfaction:

       ![type of dissatisfaction (2).png](text://image?imageFileName=type%20of%20dissatisfaction%20%282%29.png)

       **Regex Classifiers**: High precision but lower recall (means low false positives but high false negatives)

       **Human-labelled set**: develop higher recall dissatisfaction classifiers.

       **Nearest Neighbors Classifiers**: compute **PkNN (D | u)**. If u (a new utterance) has a human label for D's regex (dissatisfaction type); if not we compute the proportion of u's neighbors that are labeled D.
    2. What causes user dissatisfaction:

       ![what causes dissatisfaction (2).png](text://image?imageFileName=what%20causes%20dissatisfaction%20%282%29.png)

       t task for dealing with tweets (noisy text)Annotate 900 examples. One expert annotator viewed each (c, b,) without seeing (u)
    3. 22 % user utterances are unclear.

       in 12% contexts, the user is already dissatisfied.

       ![with vs without users prob.png](text://image?imageFileName=with%20vs%20without%20users%20prob.png)

       Most frequent errors: **redundant questions and bot logic error.** (use history more correctly and do common sense more)
    4. The relationship between bot errors and user dissatisfaction.

       positive Logic Regression coefficient (p < 0.05)

       ![Users vs Bot Error.png](text://image?imageFileName=Users%20vs%20Bot%20Error.png)

       Logic error is the least damaging one. Some users find it funny.

       Privacy and Offensive are not correlated with bot errors: Depends on the user's attitude.
  * Experiments:
    1. DialoGPT-large model (finetuned on CHIRPY conversations & NeuralChatTurns

       Input : (c, b)

       output: **P( Any | c, b )** the probabillity that the next user utterance u will express "Any" dissatisfaction.

       ![Dissatisfaction Predictor.png](text://image?imageFileName=Dissatisfaction%20Predictor.png)
    2. Train the predictor with Mean Squared Error ( the average squared difference between **P(Any | c, d)** the estimated value and and **PkNN(Any | u)** the actual value)
    3. K-nn Score (actual value) vs. dissatisfaction predictor (estimate value)

       ![image.png](text://image?imageFileName=image.png)![image (2).png](text://image?imageFileName=image%20%282%29.png)
  * Recommendation: Dataset only focuses on the demographic of the US Alexa customers who spoke to CHIRPY in 2019-2020.
* Archaeologist:
  * ***previous paper:*** Ashwin Paranjape, Abigail See, Kathleen Kenealy, Haojun Li, Amelia Hardy, Peng Qi, Kaushik Ram Sadagopan, Nguyet Minh Phu, Dilara Soylu, and Christopher D Manning. 2020. Neural generation meets real people: Towards emotionally engaging mixed-initiative conversations. In Alexa Prize Proceedings. (<https://arxiv.org/pdf/2008.12348.pdf>)

    \-> Chirpy Cardinal; same senior, the current first author also part in previous paper
  * ***following paper(s):***
    * See’s dissertation (2021): Neural Generation of Open-Ended Text and Dialogue (<https://www.proquest.com/pagepdf/2600828158?accountid=11531>)
    * Bang (2022): UX Design and Evaluation on Conversational Bot Supporting Multi-Turn and Multi-Domain Dialogues \*\*\*\*(<https://ieeexplore.ieee.org/abstract/document/9932056>)
    * Deng (2022): User Satisfaction Estimation with Sequential Dialogue Act Modeling in Goal-oriented Conversational Systems ([https://dl.acm.org/doi/pdf/10.1145/3485447.3512020?casa_token=rRk2z1pU7RAAAAAA:z7IseWzM_gPTK93IOYcU2TXnEFuLSAX21HExqCHMCfTv-DrnYGzoHExEEvFnzT7Y2H1y5Bs-MiW\\\_](https://dl.acm.org/doi/pdf/10.1145/3485447.3512020?casa_token=rRk2z1pU7RAAAAAA:z7IseWzM_gPTK93IOYcU2TXnEFuLSAX21HExqCHMCfTv-DrnYGzoHExEEvFnzT7Y2H1y5Bs-MiW%5C_))
* Academic Researcher
* **Industry Practitioner**
  * Causes of user dissatisfaction are one of the important considerations for an industry practitioner in this regard.
  * Some effects of unclear utterances and prior dissatisfaction on bot errors that the user’s utterances 22% of control set examples. In these contexts, it’s impossible for the bot to reliably produce a good response.
  * Observing that when the user’s utterance is unclear, the generative model tends to hallucinate.
  * The user has already expressed dissatisfaction about 12% in the Neural Chat context.
  * **Regex Classifiers**: High precision but lower recall (means low false positives but high false negatives) \* Precision refers to the number of true positives divided by the total number of positive predictions (i.e., the number of true positives plus the number of false positives).
  * **Human-labelled set**: This helps to develop higher recall dissatisfaction classifiers.
  * **Privacy boundaries vary:** some users perceive chatbots as lacking trust and social presence, inhibiting user self-disclosure. chatbots are perceived as more anonymous and non-judgmental than humans; this can increase user self-disclosure.
  * This experiment has done on US Alexa customers who spoke to CHIRPY in 2019–2020. We can do the same for any organization to develop user satisfaction.
  * In particular, due to latency and cost constraints, GPT-2-medium generative model is orders of magnitude smaller than the current largest generative models and trained on a fraction of the data
* **Private Investigator**

  __Abigail See__ : currently Research Scientist at Deep Mind in London (belongs to Google)
  * Wrote the paper as part of her PHD in Stanford
  * Several papers on Deep Learning in Linguistics before this one
  * Studied Mathematics in Stanford
* __Chris Manning:__ Professor in Stanford for Linguistics and Computer Science
  * Supervisor of Abigail See
  * Born in Australia did his PHD in Stanford
  * worked as a professor at Carnegie Mellon University and University of Sydney
* **Social Impact Assessor**
  * ***Positive Impacts:***
    * Documentation and analysis of user dissatisfaction in neural generative models
    * The Neural Chat module aims to offer an empathetic experience by showing an interest in the user's feelings and experiences
  * ***Negative Impacts:***
    * Small generative model
    * Limited data set

#### 

#### **Week 5**

**Paper#1**

**Title**: Christian Otto, Matthias Springstein, Avishek Anand, and Ralph Ewerth. 2019. Understanding, Categorizing and Predicting Semantic Image-Text Relations. In 2019 International Conference on Multimedia Retrieval (ICMR ’19), June 10–13, 2019, Ottawa, ON, Canada. ACM, New York, NY, USA, 9 pages. <https://doi.org/10.1145/3323873.3325049>

Roles:

* **Scientific Peer Reviewer:**

  o   **Summary**: The paper presents an approach to an automatic classification (with a deep learning model) of image-text pairs considering 8 classes (Uncorrelated, Complementary, Contrasting, Independent, Anchorage, Bad Anchorage, Illustration and Bad Illustration) and 3 metrics (CMI (cross-modal mutual information), SC (semantic correlation) from previous researches and create a third one, STAT (Status relation)). In addition, because there is not a sufficient large data set to train the deep learning model, authors outline how a comprehensive dataset can be automatically collected and augmented to train the deep learning system. Finally, they present a deep learning system (trained with their new dataset) that automatically classifies the 8 classes and metrics.

  o   **Datasets**:
* Created their own dataset

o   They used **crowdsourcing** to annotate data (to avoid doing it manually) but this approach requires efforts to maintain high quality annotations.

1. **MSCOCO** (2014)

   \- Uncorrelated class.
2. **Visual Storytelling** (VIST) (2016):

   **- Anchorage** (Desc-in-Isolation)

   **- Complementary** (Story-in-Sequence)
3. **Internet Advertisements data set:**

\- **Independent** class. 4. **ImageNET dataset** (2015):

\- **Illustration** class 5. Transform the respective positive counterparts by replacing around 530 keywords (2017) (adjectives, directional words, colors) by antonyms and opposites in the textual description of the positive examples to make them less comprehensible. This is for **Contrasting, Bad Illustration and Bad Anchorage** because these classes rarely occur

o   Total dataset: 224 856 image-text pairs.

![image (31).png](text://image?imageFileName=image%20%2831%29.png)

**Models:**

Two Classifiers:

1. **__Classic Approach:__**

\- Outputs the most likely image-text class.

1. **__Cascade Approach:__**

\- Based on classifiers for the three metrics.

\- 3 networks had to be trained and applied to predict an image-text class.

* **Experiments**:

  \- Split dataset into a training and a test set (was manually labeled to generate high quality labels).

  \- 800 image-text pairs, 100 examples for each class, were manually annotated by a 3 persons of the group.

  \- Remaining 239 207 examples were used to train the 4 different models (3 cascade and 1 classic) for 100 000 iterations each with the TensorFlow framework (2015).

  \- Used Adam Optimizer and Softmax cross entropy loss

  \- Images were rescaled to to size of 299 X 299

  \- Limit the length of textual information to 50 words per sentence and 30 sentences per image-text-pair.

![image (14).png](text://image?imageFileName=image%20%2814%29.png)Table shows the best results for predicting image-text classes using the “classic approach”.

![image (15).png](text://image?imageFileName=image%20%2815%29.png)- [16]: Henning and Ewerth (2017)

**Results**:

* Classic approach outperformed the cascade method by 6% in accuracy.
* Class Uncorrelated achieved the lowest recall in both classifiers which indicates that they often detect a connection even though there was none. Nonetheless, high precision indicates that it was almost always correct, in particular for cascade classifier.
* The classes with positive SC are mainly confused with their negative counterpart because the difference between them is often caused by a few keywords.

**Conclusion**:

* The experimental results show an accuracy of 80.83% which demonstrate the feasibility of the proposed approach.
* **Note**: Part of the work was supported by Leibniz Association, Germany (Leibniz Competition 2018, funding line “Collaborative Excellence”, project SALIENT)
* **Contributions(1-3 sent) - Novelty of paper**

\- **Derive a categorization**- from communication science research- of distinctive semantic image-text classes from multimodal analysis and search.

\- **Training data augmentation-** authors outline how a comprehensive dataset can be automatically collected and augmented to train the deep learning system in predicting the 8 image-text classes.

\- **Automatic prediction of image-text classes with a deep learning system** trained with their own data set- that automatically classifies the 8 classes and metrics.

* **Strengths and Reason to Accept**

\- Improved the previous work done by Henning and Ewerth (2017. They introduced two metrics to describe cross-modal interrelations: cross-modal mutual information (CMI) and semantic correlation (SC).

\-They considered previous research from communications science to establish the categorization classes (McCloud (1993), Nöth (1995), Barthes (1997), Martinec and Salway (2005), and van Leewen (2005))

* **Weaknesses and limitations - what has been missed and what lacks the evaluation**
* The test data was manually labeled by members of the team which can produce bias.

\- Authors state that an accuracy 80.83% for the model demonstrates the feasibility of the approach, however, in my opinion the model needs improvement because the class Uncorrelated obtained a recall of 45% and Bad Anchorage a 74.7%. Additionally, precision was low for Illustration, Bad Illustration and Anchorage (75.7%, 75.8%, 56.5%).

* **Recommended Score** - 8
* Archaeologist
* **Academic Researcher**
  * hand-labelling the whole dataset (or first reducing it and then annotate), because there seems to be disagreement between the generated image-text pairs from the augmentation and the human annotators -> what can a model really learn from the generated data, if it is falsely labelled?

    ![Screenshot 2022-11-14 at 12.03.24.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2012.03.24.png)
  * training a model on the whole dataset and testing it on the test set without labelling -> expecting very high measures
  * rebalancing the dataset, there are very few examples for Interdependent (1,007) but many for Anchorage (62637)

  ![Screenshot 2022-11-14 at 12.03.12 (2).png](text://image?imageFileName=Screenshot%202022-11-14%20at%2012.03.12%20%282%29.png)
  * taking their trained model as a validator for a image-text generator -> nothing should be generated that according to their taxonomy does not make sense
* **Industry Practitioner**
  * Main contribution: automatic tagging of the relationship between a picture and text
  * general assumption: multimodal information captures attention better:
    * sells more of items or improves learning
  * useful for search enginges: give better results
  * filter spam and adds when the picture is uncrorrelated to the text
    * automatic quality check for wiki types and textbooks
* **Private Investigator:**
  * **__Christian Otto__**
    * ***Work:*** Scientific Assistant in the "Visual Analytics" research group at the Leibniz Universität Hannover
    * ***Education:*** Master’s ([M.Sc](http://M.Sc).), Leibniz Universität Hannover
    * ***Interests:*** Cross-modal relationships between visual and textual information
  * **__Matthias Springstein__**
    * ***Work:*** Scientific Assistant in the "Visual Analytics" research group at the Leibniz Universität Hannover
    * ***Education:*** Master’s (M.Eng.), Leibniz Universität Hannover
    * ***Interests:*** Deep learning, concept recognition, and monocular depth estimation
  * **__Avishek Anand__**
    * ***Work:*** Associate professor at the Software Technology department at the Delft University of Technology
    * ***Education:*** Ph.D. in Information Retrieval, 2013, MPI Informatik & Saarland University
    * ***Interests:*** Interpretability of learning systems, deep learning for search, information retrieval
  * **__Ralph Ewerth__**
    * ***Work:*** Professor at the Leibniz Universität Hannover and head of the "Visual Analytics" research group
    * ***Education:*** Ph.D. in “Robust Video Content Analysis via Transductive Learning Methods”, 2008, Philipps University in Marburg
    * ***Interests:*** Multimedia information retrieval, analysis of multimodal information, computer vision
  * Part of this work is financially supported by the Leibniz Association, Germany.
* Social Impact Assessor

  \-Negative impact: Biased dataset

  \-Positive impacts: Introduction of a text-image relation taxonomy from a computer science perspective

  \-The model can improve information retrieval in search

**Paper#2**

**Title**: __Categorizing and Inferring__ the relationship between the Text and the Image of Twitter Posts

Roles:

* Scientific Peer Reviewer
  * Summary
    * **The main research question** - Infer the relationship between the tweets shared as text along with its associated image. The researchers defined the annotation scheme that focuses on two methods separately and look at both their semantic overlap and contribution to the meaning of whole tweet.
    * **Experiments(Scores, Datasets)** -
      * Data collection - Collected annotations from 4,471 tweets
      * All tweets were in English
      * Tweets taken from US(to not complicate with different diversities)
      * Split across original posts, retweets and favourite posts
    * **The proposed methodology** - They categorised the tasks into two main categories(Text Task and Image Task) and then used 5 methods in to analyse the data.
      * Text Task - Determine whether there is semantic overlap between the context of the text and the image
      * ![Screenshot 2022-11-14 at 13.22.31.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2013.22.31.png)
      * Image Task - Focuses on the role of the image to the semantics of a tweet
        * ![Screenshot 2022-11-14 at 13.23.20.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2013.23.20.png)![Screenshot 2022-11-14 at 13.27.28.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2013.27.28.png)
      * **Data Annotation -**
        * Data were annotated using Figure Eight technology
        * Text Task - Redundancy of 3, Krippendorf's Aplha = 0.71, Annotators maintained > 85% accuracy over test set
        * Image Task - Redundancy of 5, Krippendorf's Aplha = 0.46, Annotators maintained > 75% accuracy over test set
      * **Methods**
        * User Demographics - Age, Gender and Education
        * Text-based Methods
          * Surface Features
          * Bag of Words
          * LSTM - Long Short Term Memory
        * Image-based methods
          * ImageNet Classes
          * Tuned InceptionNet
        * Joint Text-Image Methods
          * Ensemble - A simple method for combining the information from both modalities to build an ensemble classifier and it is done with logistic regression model using two features, **BoW and Tuned InceptionNet models to predict class probability**
          * LSTM + InceptionNet - This is a joint approach from the final layers of LSTM and InceptionNet models and passing through a fully connected feed forward neural network with one hidden layer. The final outcome was text-image relationship type and later used ADAM optimizer to fine tune this network.
          * ![Screenshot 2022-11-14 at 13.18.19.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2013.18.19.png)
  * **Predicting Text-Image Relationship**
    * Split data into 80%(Training) and 20% test data set.
    * Parameters using 10-fold cross validation with training set and results are reported on the test.
    * **Majority Baseline**
    * ![Screenshot 2022-11-14 at 13.49.52.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2013.49.52.png)![Screenshot 2022-11-14 at 13.50.44.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2013.50.44.png)![Screenshot 2022-11-14 at 13.51.59.png](text://image?imageFileName=Screenshot%202022-11-14%20at%2013.51.59.png)![Screenshot 2022-11-14 at 00.02.38 (2).png](text://image?imageFileName=Screenshot%202022-11-14%20at%2000.02.38%20%282%29.png)
  * **Contributions** -
    * Relationship type is Predictable from both text and image
    * New classification schema and data set for text-image relationship on Twitter
    * Text-image relationship is useful for downstream relationship
  * Strengths and reason to accept - The work they have done will be definitely useful for future research, the research is thorough in giving results based on US demographics.
  * Weaknesses and limitations - The data is limited, all other than english language tweets were filtered out
  * Recommendation Score - 6-7
* **Archaeologist**
  * **Previous Work**
    * Thoroughly documented in the RELATED WORK section
      * Looking back from 2003 all the way to 2019 (year of publishing the paper)
      * Most previous work is focused on "captioning", specially on old papers. Not exactly the same as Twitter posts. A lot more intent/feelings are trying to be transmitted, lots of not-necessarily-well-thought combinations(lower barrier of entry).
      * Examples:
        * **Older paper** : Radan Martinec and Andrew Salway. 2005. A System for Image–Text Relations in New (and Old) Media. Visual Communication, 4(3):337–371. -> This one has a very nice way of evaluating the task, very similar to current paper. "*Martinec and Salway (2005) aim to categorize text-image relationships in scientific articles from two perspectives: the relative importance of one modality compared to the other and the logico-semantic overlap"*
        * **Newer papers(and most similar)**:
          * **Most similar works**: Tao Chen, Dongyuan Lu, Min-Yen Kan, and Peng Cui. 2013. Understanding and classifying image tweets. MM, pages 781–784. (correlation text\~image)
          * **Most similar works:** Tao Chen, Hany M SalahEldeen, Xiangnan He, MinYen Kan, and Dongyuan Lu. 2015. Velda: Relating an image tweet’s text and images. AAAI, pages 30– 36. (specific tecnique Visual-Emotional LDA [https://www.cs.jhu.edu/\\\~taochen/data/pubs/velda_aaai15.pdf](https://www.cs.jhu.edu/%5C~taochen/data/pubs/velda_aaai15.pdf) )
  * **Future Work**
    * **Dataset statistics have been used:** Sun, Lin et al. “RpBERT: A Text-image Relation Propagation-based BERT Model for Multimodal NER.” *ArXiv* abs/2102.02967 (2021): n. pag. (entity recognition paper, notices how most tweets pictures are irrelevant to the text)
    * **Most relevant paper referencing this one:** Hessel, Jack and Lillian Lee. “Does My Multimodal Model Learn Cross-modal Interactions? It’s Harder to Tell than You Might Think!” *ArXiv* abs/2010.06572 (2020): n. pag. (does not use dataset, but references it) [https://scholar.google.com/scholar?hl=ca&as_sdt=0%2C5&q=Does+my+multimodal+model+learn+cross-modal+interactions%3F+It's+harder+to+tell+than+you+might+think!+Jack+Hessel%2C+Lillian+Lee&btnG=](https://scholar.google.com/scholar?hl=ca&as_sdt=0%2C5&q=Does+my+multimodal+model+learn+cross-modal+interactions%3F+It%27s+harder+to+tell+than+you+might+think%21+Jack+Hessel%2C+Lillian+Lee&btnG=)
    * **Related recent work:**
      * <https://scholar.google.com/scholar?as_ylo=2022&q=cross+modal++twitter&hl=ca&as_sdt=0,5>
      * **Caption generation with coherence** [Malihe Alikhani](https://arxiv.org/search/cs?searchtype=author&query=Alikhani%2C+M), [Piyush Sharma](https://arxiv.org/search/cs?searchtype=author&query=Sharma%2C+P), [Shengjie Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+S), [Radu Soricut](https://arxiv.org/search/cs?searchtype=author&query=Soricut%2C+R), [Matthew Stone](https://arxiv.org/search/cs?searchtype=author&query=Stone%2C+M) "Clue: Cross-modal Coherence Modeling for Caption Generation" <https://arxiv.org/abs/2005.00908> (2021)
      * **Multimodal analysis is still under study, surely it will be useful to look insights:** Xueming Yan, Haiwei Xue, Shengyi Jiang & Ziang Liu "Multimodal Sentiment Analysis Using Multi-tensor Fusion Network with Cross-modal Modeling" (2022)
* **Academic Researcher:**
  * first annotated corpus studying the relationship between text + image in same social media post → use dataset to try out some of the applications they suggested → eg performance comparison when using data/task/classifier as preprocessing step VS. without filtering out cases of no semantic overlap
    * automatic image description generation
    * multi-modal named entity disambiguation
    * multi-modal topic labelling
    * improving recommendations
  * dive more into discovered age difference
    * mutual understandability?
    * age detection based on use?
  * can results be replicated for different individual/multiple cultures?

**Industry Practitioner**:

* I'm working for LinkedIn. Many posts in LinkedIn tend to contain images that do not add any information to the text of the post.

  In order to save screen space and data traffic (*see image below*), it would be a good idea to implement a feature (for example a placeholder in the place of the image) which allow the user to choose if they can see an image that does not add any more information about the text.

  The model described in the paper shows high performance (F1 score = 0.81) expecially in identifying when the image does not add any meaning to the text, and could therefore be ignored.

  The important thing is to minimize the risk that images which bring meaning to the text are hidden. When images that does not bring meaning to the text are shown, is not that big of a problem. This model works well in this sense.

  ![Screenshot (189).png](text://image?imageFileName=Screenshot%20%28189%29.png)On the left there images of graphs that add to the meaning of the text and are therefore not hidden. Images on the right do not provide us with additional information about the text and are therefore hidden to save screen space.
* **Private Investigator**

  __Alakananda Vempala__

  \-Research engineer at Bloomberg since 2019

  \-PhD in computer science from University of North Texas (grad. 2019)

  \-Research focuses:

  Social media analysis: A series of papers regarding the analysis of Twitter activities e.g., event participation

  Semantic information extraction: spatial knowledge

  __Daniel Preotiuc-Pierto__

  \-Senior research scientist and team lead in AI Group Bloomberg

  \-PhD focuses on temporal models for social media at University of Sheffield

  \-Supervise a large number of studies analysing various user behaviours all over different social media platforms

###### Social Impact Assessor

**|+|** Being able to automatically hide images that likely add little or no content is useful as it **saves screen space and web traffic**

**|+|** Researchers in social sciences could use classification tool to **better analyse and describe social media usage**

**|-|** Users might feel compelled to **escape classifier** by designing their image posts in such a way that text and image complete each other (for their content to take up more screen space and thus be more visible)

**|-| Certain groups of people** (e.g. those who cannot read very well or prefer to understand content visually) **might not want images to be hidden** for posts where an image illustrates the textual content

#### **Week 6**

Paper#1 title: Leveraging User Paraphrasing Behavior In Dialog Systems To Automatically Collect Annotations For Long-Tail Utterances

Roles:

**Scientific Peer Reviewer**

* **Short Summary**

*Problem setting*: in commercial dialog systems, users express the same request in a variety of ways:

1. there are a few, more common ways to express a request: "Play blinding lights by the weekend", and 2) a wide range of less-common ways to express the same request: "the weekend blind uh lights please", or "we decided to listen to the weeknd ’s blinding lights".

*Research question*: How can we handle those numerous, different ways to express the same things?

*The proposed methodology*: They propose MARUPA, a *data collection approach* that detects when the message from the users is not understood by the machine, and automatically collects and annotates those sentences to re-train and improve the dialog system on new data.

Marupa has three components:

1. Paraphrase detection (PD). *Paraphrases* are definied here as different utterances with:

   \- the same *intent* (= what the user wants, for example: *PlayMusic*) and

   \- the same *slots* ( = assigning the right tokens to the respective slots, for example: *the weeknd* -> *ARTIST*, *Blinding lights* -> *SONG*)
2. Friction detection (FD). *Friction* occurs when the utterance from the user is not correctly interpreted by the system, which gives a different reaction than expected.
3. Label projection (LP) alligns the tokens on the frictioned sentence to the correct interepreted sentence.

   ![image (11).png](text://image?imageFileName=image%20%2811%29.png)

*Experiments (scores, datasets, comparison to other work)*:

Data: German, Italian, and Hindi utterances. They use dialog system model before applying MARUPA as the baseline and compare it against the model (MARUPA) re-trained with the new data.

To collect training examples, they apply MARUPA to a sample of anonymazed user intereaction.

For paraphrase detection, they derive a corpus of 300k parphrase pairs from the training data of the baseline model and split into train, validation and test spilt (70/10/20). They fine-tune BERT-Base Multilingual Cased\*.\*

They model friction detection as a binary classificaton task, and using a linear SVM for classification.

Then they aggregate the collected examples by keeping only utterances that occur at least 10 times and for which the same annotation was created in at least 80% of the cases

*RESULTS*

They evaluate the model error rate on its main test set, and on a set of held-out MARUPA collected examples (20%). While the overall change on the main test set (total) is negligible, low-frequency utterances leads the error rate reduction of up to 3%. NO negative effects of high-frequency utterances

![Screenshot (191) (2).png](text://image?imageFileName=Screenshot%20%28191%29%20%282%29.png)

* **Contributions** (1-3 sentences): novelty of the paper as bullet points

  \- While models that are trained on prediction that a previous version of the model made on unlabeled data are already widely used, the novelity of this model lays in the use of user feedback in the form of paraphrasing and friction
* **Strengths and reasons to accept** (1-2 sentences): what makes this work worthy of publication

The method is fully automatic (no cost for annotation), and it increases coverage and performance for the noumerous less-frequent utterances

* **Weaknesses and limitations** (1-2 sentences): the reasons to reject, what has been missed in the paper? What lacks in the experiments/evaluation?

There is a paraphrase detection dataset available and no baseline: as the baseline they use the model of the dialog system before re-training it with the model.

\- **Recommendation score**: 1-10 (higher better)

**7**. The main problem here is that it is difficult to determine the performance of the model. But the model has two main advantages: 1) it is fully automatic and 2) it deals with a wide range of uncommon utterances. I also liked the idea that it uses as its own advantage the paraphrases by the users.

* Archaeologist
  * Previous Work:
    * Most similar (according to themselves): Muralidharan et al (2019): positive and negative user feedback to turn coarse-grained annotations into fine-grained ones, domain-specific to music requests, requires coarse-grained annotations, only 5 citations so far
  * 2020: this paper
  * Future Work:

    14 Citations
    * Two citations from 2021 by co-authors or supervisors of this paper, but only mentioning this paper, not exactly building on it
    * Efficient Semi-supervised Consistency Training for Natural Language Understanding (2022) by G. Leung and J. Tan: use the MARUPA dataset for their experiments to leverage large unlabelled data to augment data for dialogue systems. Experiments compared back-translation, dropout and user paraphrasing (MARUPA), user paraphrasing performing worst :/
    * Thesis: Neural conversation models with expert knowledge by A. Tysnes cites the paper 5 times in the text
    * All others only barely touch on the paper’s topic directly, it’s more of a “look what else there is out there”
* Academic Researcher (Robin)

  [Potential follow up project, building on paper's findings]

  \-> Idea: Transport approach to related dialogue system:

  Build a system that creates a similar synthetic set of annotated data from **user-chatbot interactions**
  * Problem: Users make queries to the chatbot, sometimes using utterances from the tail of the distribution --> bot does not understand --> user paraphrases query
  * run
    * **FD**: flag turns of conversation where user is dissatisfied with Intent Classif. or Slot Label.

      \-> here: try to filter out other reasons automatically (e.g. SpeeRec error, EntRes etc...) beforehand using user disatisfaction analysis tools
    * **PD**: find all (u, i, s) - (u', i, s)
    * **LP**: map utterances and slot labels of u (causing friction) to those of u' (not causing friction)
  * Use on different languages, e.g. Turkish where tokenization and lemmatization works differently

  Industry Practitioner
  * Direct human data (better quality, expensive) vs. machine-processed/generated data (poorer quality, cheap).
  * Large enterprises (e.g., Amazon, Google) may not find the proposed method necessary because the wide-spread products have alreay been collecting a large volume of data collected from end users, and they certainly have enough resource for optimising the database.
  * For SMEs (e.g., start-ups focusing on B2C products), the method can be a cost-down opportunity while decision making; that is, choosing between getting human data or machine-generated ones for scaling up database. The will of implementing the method could alter depending on market-specific business strategies (Hindi could be particularly attractive owing to its market potentials, despite the poorer quality in model output).
* Private Investigator (Guillem)
  * Funding: Not mentioned explicitly in the paper. But it is pretty obviously amazon:
    * ![image (4).png](text://image?imageFileName=image%20%284%29.png)
  * **Paper is 2020. Alexa came out on November 6, 2014**
  * Authors
    * Tobias Falke:
      * website: <https://tbsflk.github.io/>
      * linkedin: <https://www.linkedin.com/in/tobiasfalke/>
      * Work: Working at Amazon while this paper was presented(since 2019). Had worked in Google previously(3 months in 2018) while doing PhD.
      * Education: Ph.D. at UKP/AIPHES (NLP,ML,DL), Master in AI, Bachelor in Business Information Technology
        * ![image (6).png](text://image?imageFileName=image%20%286%29.png)
      * Interests (shown by his profile):
        * ![Oldest Papers](text://image?imageFileName=image.png)
        * ![Newest papers](text://image?imageFileName=image%20%282%29.png)
        * 15 papers since 2016, 2 awards(**Industry track**, Best resource paper award)
    * Markus Boese
      * Linkedin: <https://www.linkedin.com/in/markus-boese-a20802109/>
      * Work: Developer since 2009, Researcher in Amazon since 2017.
      * Education: Master and Bachelor in FH Münster(one after another)
      * Interests(<https://aclanthology.org/people/m/markus-boese/>):
        * ![Not that active in academia](text://image?imageFileName=image%20%285%29.png)
    * Daniil Sorokin
      * Linkedin: <https://www.linkedin.com/in/daniilsorokin/>
      * Work: Developer since 2010, Researcher in Amazon since 2019. (many short experiences/jobs). First NLP position in 2014
      * ![image (12).png](text://image?imageFileName=image%20%2812%29.png)
      * Education: PhD at UKP/AIPHES (NLP,ML,DL) . 2 masters(Applied Linguistics, Computational Linguistics) + 1 Bachelor's(Applied linguistics) All with excellent marks.
        * ![image (7).png](text://image?imageFileName=image%20%287%29.png)
      * Interests: Really active in NLP since 2017 <https://aclanthology.org/people/d/daniil-sorokin/>
    * Caglar Tirkaz
      * Linkedin: <https://www.linkedin.com/in/caglar-tirkaz-3a444392>
      * Work: Developer since 2004, Researcher in Amazon since 2016. (many short experiences/jobs until Amazon). First NLP position in 2014
      * Education: PhD at Sabanci University (NLP,ML,DL) . 1 masters(Applied Linguistics, CompEng) + 1 Bachelor's(CompSci)
      * ![image (8).png](text://image?imageFileName=image%20%288%29.png)
      * Interests:
        * Polyvalent job profile
        * Only active in publishing papers since 2020:
      * ![All papers](text://image?imageFileName=image%20%289%29.png)
    * Patrick Lehnen
      * Linkedin: <https://www.linkedin.com/in/patrick-lehnen-b97596209/>
      * Work: In amazon since 2014
      * Education: Doctor in Informatics 2014. Doctor in Physics finished in 2007
      * Interests: First NLP paper in 2008. More active recently, not so much in the past. <https://aclanthology.org/people/p/patrick-lehnen/>
* Social Impact Assessor -
  * (+) Helpful for future researchers.
  * (+) The thought behind this will be really helpful if someone researches further on this path and it can be helpful to people with speech problem or someone who cannot express themselves clearly
  * (-) How they handle data(source and usage) is unknown, since they are using different users request to generate the model. If data is leaked, then there are chances to understand one's personality
  * (-) Test dataset was heavily relied on German and Italian, still a long way to go for many different languages.

**Paper#2 title:**

Roles:

* Scientific Peer Reviewer
* Archaeologist:

  **Influenced by:**

  *Zero-shot learning by providing task descriptions*

  [Language Models are Unsupervised Multitask Learners](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf)

  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (2019)

  *Reducing the amount of compute required for few-shot learning.*

  [Green AI](https://arxiv.org/abs/1907.10597) (2020)

  Roy Schwartz, Jesse Dodge, Noah ASmith, Oren Etzioni

  *PET(Pattern-Exploiting Training), using knowledge distillation and self-training*

  [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)

  Timo Schick, Hinrich Schütze (submitted in Jan, 2020, revised in 2021)

  **Influence:**

  [Calibrate Before Use: Improving Few-shot Performance of Language Models](http://proceedings.mlr.press/v139/zhao21c.html)

  Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh(2021)
* Academic Researcher
  * Can the same be applied to generative tasks using generative models to see how effective it is .
  * As they perform good considering 1 specific task at a particular time . How will they perform in a multiple task setting
  * also as they have not done any pre processing except sorting all examples to the maximum sequence length. what is the performance some preprocessing techniques are involved and also what happens if more data is given.
* Industry Practitioner
* Private Investigator

  Paper was cited 354 times according to Google Maps (considerable number)

  **Main Author: Timo Schick**
  * PhD student, Computer Science, LMU Munich
  * Researcher at Center for Information and Language Processing, LMU Munich
  * worked as Data Scientist in the past, has recently started a job at Meta
  * Research Areas: Natural Language Processing, Representation Learning, Low-Resource Settings
  * has (co-)published a number of scientific papers a variety of topics (Language Model architecture etc...)

    ![grafik.png](text://image?imageFileName=grafik.png)

  **Supervisor: Prof. Dr. Hinrich Schütze**
  * Doctor of Comp. Ling. from Stanford University
  * former president of Assoc. for Comp. Ling.
  * has authored standard works on Information Retrieval and Stat. NLP, published by MIT Press / Cambridge University Press
* **Social Impact Assessor**
  * ***Positive Impacts:***
    * It will help to retrieve the exact information from the search.
    * Quick decision-making on some context.
    * Low financial cost using PET rather than GPT-3. It Gives almost similar performance.
    * Helpful for future researchers and practitioners.
    * Reduces environmental impact and leads to a much smaller carbon footprint.
  * ***Negative Impacts:***
    * Requires finetuning multiple models for each task.
    * PET does not work for generative tasks.
    * Verbalizer used in pattern-exploiting training maps with each output to only a single token.
    * Works on small language models.

#### **Week 7**

Paper#1 GRILLBot: A flexible conversational agent for solving complex real-world tasks <https://assets.amazon.science/7c/99/e7a8d35a43c88cf0e8ad59b92dfc/grillbot-a-flexible-conversational-agent-for-solving-complex-real-world-tasks.pdf>

**Title**:GRILLBot: A flexible conversational agent for solving complex real-world tasks

Roles:

* **Scientific Peer Reviewer**

**Summary** : GRILLBot, is a task bot which was developed for the Alexa TaskBot Challenge in 2022 by students of University of Glasgow Scotland.Which is a multi-modal task-oriented voice assistant. It basically assists the user in cooking and house improvement related domains by effectively guiding the user through long and complex tasks.

**Two UI’s**

(a)   Voice UI – mainly focuses on optimal speech response

(b)   Screen UI – to improve communication and keep the user engaged throughout the process.

**Dialogue Schema**

The User conversation is represented in finite set of phases

·       Domain - introduction and selection

·       Planning - Searching or guiding

·       Validation - Choice confirmation

·       Execution – guided through the steps

·       Farewell – End

**System overview**

![image (21).png](text://image?imageFileName=image%20%2821%29.png)**Task Representation – TaskGraph**

Task graphs are used to enable the complex flow of the execution of steps

1\.Steps (instruction) 2. Requirements (tools and ingredients required) 3. Conditions (external information) 4. Logic(conjunction) 5. Actions (by system) 6. Extra Information (fun facts)

![image (25).png](text://image?imageFileName=image%20%2825%29.png)

**Offline task Graph Curation**

1. The system processes TaskGraphs offline which will cover most of users need.
2. ·  Multiple crawls are used to download the raw HTML for target domains and develop website-specific wrappers to extract semi-structured information about each task.
3. ·  These semi structured informations are used to filter out the poor quality TaskGraphs and create summaries of the task nodes.
4. ·  The task corpus contains 165,677 TaskGraphs after pre-processing and filtering according to their observation over 95% of user tasks are covered by this corpus.

\->**GrillBot also handles different types of questions which is not directly associated with moving the task forward.**

These Questions are divided into 9 categories. Which are dealt with different Q&A Systems

![image (22).png](text://image?imageFileName=image%20%2822%29.png)This is especially important when the language model does not know the answer to a question and needs to both inform the user of this and make them aware of their capabilities.

· out of domain questions and chit-chat - neural language models (Zero-Shot approach) to retain maximal fluency.

· task-specific QA (ingredient, task, and step questions) We use zero-shot UnifiedQA [Khashabi et al., 2020] to autoregressively generate an answer.

· When general questions are asked Amazon Evi API is called.

**Analysis**

![image (23).png](text://image?imageFileName=image%20%2823%29.png)Fig. 7a – The rating given by the users throughout the competion - During week 7 and 9 its increased because of   where the Grillbot if guidance is needed initiates sub-dialog to exact user preference.

Fig. 7b - shows the average ratings by conversation length.-Conversations under 60 seconds usually came from a user that started the bot by mistake For conversations longer than 1 minute, the ratings increase proportionally with the conversation duration.

**Conclusion:**

Most of the necessary things were explained well. The Task Bot is well developed and helpful.

**Strengths and reasons to accept (1-2 sentences)**: what makes this work worthy of publication

·       The TaskBot itself performs exceptionally, and the device's performance is extremely good.

·       It can be implemented in the real world, which could help a lot of people in daily life.

·       This can be used as a blueprint for the introduction of more task bots in many fields.

·       The system functions better by minimizing waiting times when using task corpora that contains task graphs offline.

·       Understands how to bring back the user to their task and handle a variety of questions

**Weaknesses and limitations (1-2 sentences)**: the reasons to reject, what has been missed in the paper? What lacks in the experiments/evaluation?

·       If the improper terms are used, the bot might not function properly and it might stop the flow.

·       A warning before halting the procedure could be beneficial.

·       The paper's structure could have been done more effectively. Some of it was tough to grasp, and the flow wasn't very good.

**Recommendation score**: 8.5 - 9

* Archaeologist
  * Previous work:
    * no paper stands out also only 14 references
    * papers mentioned describe used resources (Reddit dataset by Baumgartner et. al 2020) or definitions of their approach (e.g. Raffel et. al 2020)
  * Future work:
    * one additional paper about Grillbot:
      * **GRILLBot: An Assistant for Real-World Tasks with Neural Semantic Parsing and Graph-Based Representations**
      * additional summary of Grillbot
* Academic Researcher (Yi-Sheng)
  * Improving of the proposed voice assistance in shorter talks -- the model doesn't really perform well in rating (< 3/5 around 1 min) , and collecting data on dissatisfaction could make another research topic![截圖 2022-11-25 22.14.40.png](text://image?imageFileName=%E6%88%AA%E5%9C%96%202022-11-25%2022.14.40.png)
  * Leveraging the methodology to other fields, potentially medical industry of aviation, which also involve complex procedures. These fields of expertise may also help narrowing down the focuses and improving voice assistance performance because conversations on cooking and housing can usually be more random and various at a semantic level.
* Industry Practitioner
* Private Investigator (Lara)
  * GRILL stands for Glasgow Representation and Information Learning Lab
  * All 5 students are members of GRILLLab, Jeff Dalton is "Principal Investigator" (<https://grilllab.ai/people/>)
  * Were selected for participation in the challenge, among 9 other teams, received research grant of $250,000 to support 12 months of work, also four amazon Alexa devices, access to Amazon Web Services and support from Amazon Alexa Team, won the competition and received $500,000
  * (top row: Paul Owoicho, Prof. Jeff Dalton, bottom row: Iain Mackie, Federico Rossetta (Tablet), Carlos Gemmell, and Sophie Fischer)

    ![GRILLBot Team (2).jpg](text://image?imageFileName=GRILLBot%20Team%20%282%29.jpg)
  * **Carlos Gemmell** (https://carlos-gemmell.github.io)
    * Studied computer science at University of Glasgow
    * Ph.D. student at Glasgow uni, supervisor Jeff Dalton
    * “Led the winning team for the 2021-2022 Amazon Alexa Prize TaskBot challenge” - this paper
    * Interests: decoupling memorisation from reasoning, delegating computation and knowledge by using tools
    * Applications interests: Question Answering, Conversational Systems
    * Published 8 papers, 2 in 2020 and 6 in 2022, only generating 17 citations overall, 3 concerning GrillBot
    * School of Computing Science - Lab Assistant (Tutor) at Uni
  * **Jeff Dalton**
    * PhD in 2014, University of Massachusetts Amherst at Center for Intelligent Information Retrieval
    * 2014 - 2017: Joined Google Research, developer in the Google Assistant NLU team
    * Now Senior Lecturer at Uni for Computing Science (AI and Information Retrieval)
    * 1185 citations on Google Scholar, 79 publications (also one book according to Glasgow Uni website)
    * Interests: Knowledge graphs, intersection of NLU and Information Seeking
    * Application interests: health, biomedical, travel, food, science, engineering, history (very broad in my opinion )
    * Leader of GRILL, Turing AI Acceleration Fellow
    * ”I've engineered large-scale systems for web crawling, link analysis, web information extraction, knowledge base construction, and real-time query suggestion.” - so GRILLBot is right up his alley
    * Supervises all 4 PhD students (and their interests):
      * Carlos Gemmell: Deep learning for structured prediction using interative dialogue
      * Iain Mackie: Deep Learning for Information Extraction and Knowledge-Centric Entity and Text Ranking
      * Federico Rosetto: Multi-modal music conversational assistance using reinforcement learning
      * Paul Owoicho: Mixed-initiative and feedback for Conversational Information Seeking
  * **Sophie Fischer (for completion)**
    * Tutor at School of Computing Science
    * Member of GRILLLab as a MSci student
* **Social Impact Assessor** (Guillem Gili i Bueno)
  * Positive Impacts
    * Multi-domain admissible (on so many domains teaching/instructing is a necessity, specially on the ones where revising tech happens fast)
    * You cannot (always) interact with a delicate interactive machine in any domain. Or interacting with a screen is eye-exhausting.
    * Google has become increasingly less useful, nice replacement for search how-to's. Excellent search functionality.
    * Allows for more general human autonomy(people in good infrastructure areas will get things solved faster).4
  * Negative Impacts
    * Does it add that much value to the tutorial?
      * Why do we need an AI model for the equivalent of scrolling(the search functionality is useful)
      * You usually only get stuck at 1 or 2 steps during a tutorial. Can it be helfpul in those situations
    * Internet dependant(strategies for poor country infrastructures?/remote locations, internet issues can happen). More internet traffic.
    * ![image (26).png](text://image?imageFileName=image%20%2826%29.png)
    * ![image (13).png](text://image?imageFileName=image%20%2813%29.png)<https://arstechnica.com/gadgets/2022/11/amazon-alexa-is-a-colossal-failure-on-pace-to-lose-10-billion-this-year/>
    * 

**Paper#2**

**Title:**

Yen Ting Lin, Alexandros Papangelis, Seokhwan Kim, and Dilek Hakkani-Tur. 2022. [Knowledge-Grounded Conversational Data Augmentation with Generative Conversational Networks](https://aclanthology.org/2022.sigdial-1.3). In *Proceedings of the 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue*, pages 26–38, Edinburgh, UK. Association for Computational Linguistics. <https://arxiv.org/pdf/2207.11363v1.pdf>

**Roles:**

* **Scientific Peer Reviewer: Md Delwar Hossain**
  * **Summary:**
    * They presented a data augmentation method based on GCN(Generative Conversational Networks) to generate conversational data on unstructured textual knowledge. This method can generate high-quality data given small seed data. In the part of human evaluation and analysis, the GCN method can generate diverse data beneficial to task learning. while rich open-domain textual data are generally available and may include interesting phenomena (humor, sarcasm, empathy, etc.) most are designed for language processing tasks.
  * **Research Question:** How Generative Conversational Networks can produce a meaningful full response from the non-conversational text format?
  * **Dataset:** Topical Chat
    * The experiments have been done in conversations with and without knowledge on the **Topical Chat dataset** using automatic metrics and human evaluators.
    * It's a set of human-human conversations without explicitly defining the role of each participant.
    * It's collected over Amazon Mechanical Turk.
    * Each participant had access to a set of factors or articles.
  * **Task:**
    * Open Domain Response Generator
      * Open-domain conversation:
        * Given dialogue context C = U1, U2, U3.....U(n-1)
        * Generate the response: Un
    * Knowledge Grounded Response Generation
      * [Knowledge-Grounded Conversation::](https://drive.google.com/file/d/17S09B0yEsap8LGS6uDHO2rUl1ZlPR3n8/view?usp=sharing)
      * Given dialogue context C = U1, U2, U3.....U(n-1)
      * Retrieve relevant knowledge pieces Kc =argmax kϵK = { cos(tC, tk)}
        * TF-IDF retriever works best in pilot experiments
      * Generate the response: Un

        ![Knowledge Grounded Response.PNG](text://image?imageFileName=Knowledge%20Grounded%20Response.PNG)
  * **Proposed Methodology/Model:**
    * Generative Conversational Network(GCN): Data Generator and Learner
    * It is a meta-learning approach that can be used to generate labeled, diverse and targeted data.
    * Experiments in low resource settings.
    * Seed data (Training):
      * For open domain conversation, 10% of the original data is randomly selected.
      * For Knowledge grounded conversation, 1%, 5%, and 10% of the original data are randomly selected.
    * Data Generator: DialoGPT-small as the initial model for the generator
    * BART, and BlenderBot for Learner as initial models.
    * [Better Image Quality](https://drive.google.com/file/d/1FUTwvXpVMmgKLosoYeXbblCbwr8QCsSC/view?usp=sharing)

      ![Capture.PNG](text://image?imageFileName=Capture.PNG)
    * Reward Metrics Calculation:
      * Open Domain: BLEU + Rouge-L + BERTScore
      * Knowledge Grounded: BLEU + Knowledge F1
    * Generator Update: Proximal Policy Optimization(PPO) with reward metric r(C, U).

      where C represents the context including the knowledge, and U represents the model’s response.
    * Final Learner: Picked the best performance generator checkpoint measured by the learner's performance on the validation set. and create a final augmentation set that is usually much larger than the training sets.
  * **Results/Evaluatio**n:
    * Open-domain:
      * Conversations without knowledge grounding, GCN can generalize from the seed data, producing novel conversations that are less relevant but more engaging: [Image](https://drive.google.com/file/d/1Qi-A92btbPxygSil59i8_nl7JzjNtDWb/view?usp=sharing)

        ![1.PNG](text://image?imageFileName=1.PNG)
      * Generated a thousand examples for each condition using the same context and 3 ratings per example and per condition.
    * Knowledge Grounded:
      * Used samples 1%, 5%, and 10% of Topical Chat as seed data.
      * DialoGPT-small and BlenderBot as initial models for the generator and the learner respectively.
      * Conversations with knowledge-grounded can produce more knowledge-focused, fluent, and engaging conversations. [Image](https://drive.google.com/file/d/1JAPE91P7znnSzicWTZWyDtQ5UAncmEAS/view?usp=sharing)
        * ![2.PNG](text://image?imageFileName=2.PNG)
    * Human Evaluation:
      * The performance of both data generators in the GCN and the final Learners
      * The learner GCN with reinforcement learning produces more engaging, fluent, and relevant conversation and overall outperforms both baselines well. [image](https://drive.google.com/file/d/1FCvMfy0_CtJwumDQlgBGwL0LJs5WECtv/view?usp=sharing)

        ![3.PNG](text://image?imageFileName=3.PNG)
      * Amazon Mechanical Turk setup for human evaluation: [Image](https://drive.google.com/file/d/1nIK_bLpATpNhCXLRpQrMArNKKk1F8lw4/view?usp=sharing)

        ![Amazon Mechanical Turk setup for human evaluation.PNG](text://image?imageFileName=Amazon%20Mechanical%20Turk%20setup%20for%20human%20evaluation.PNG)
  * **Contributions** :
    * In previous work, They showed that the GCN can be used to generate data for intent detection tests and Slot filling tests.
    * Generate knowledge-grounded conversational data from unstructured textual knowledge.
    * Improved response generation quality over a baseline that uses fine-tuning on seed data, eliminating the need for additional human-human data collection.
    * Improved performance on knowledge-grounded response generation on Topical Chat.
  * **Reasons to accept:**
    * The methods used are applied rigorously and explain why and how the data support the conclusions.
    * Connections to prior work in the same fields are made the article's arguments clear.
    * The article tells a good story.
  * **Weaknesses and limitations:**
    * GCN relies on Reinforcement Learning, so it may be difficult to tune for more complex applications.
    * Due to its meta-learning nature, it can be computationally expensive.
  * **Recommendation score:** 8
* **Archaeologist**
  * ***Previous work:***
    * 49 references
    * Shah, P., Hakkani-Tür, D.Z., Tür, G., Rastogi, A., Bapna, A., Kennard, N.N., & Heck, L. (2018). Building a Conversational Agent Overnight with Dialogue Self-Play. ArXiv, abs/1801.04871.
      * Common author: Dilek Hakkani-Tür
      * Summary: They propose a framework using both automation and crowdsourcing to generate goal-oriented dialogues and train dialogue agents.
      * ![photo_2022-11-28_14-30-42.jpg](text://image?imageFileName=photo_2022-11-28_14-30-42.jpg)
    * Lin, C. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. Annual Meeting of the Association for Computational Linguistics.
      * More than 7800 citations
  * ***Future work:*** NA
* **Academic Researcher**
  * I would try to replicate the experiment expanding the seed (using more than the 86 dialogues of the paper), and see how much the model improves.
  * Maybe experimenting with more model architectures in addiction to Generative Conversational Networks (GCN)?
* Industry Practitioner
* Private Investigator(Heena) :
  * **Yen-Ting Lee** -
    * Completed Bachelors in Information Management and PhD from National Taiwan University
    * Currently not working (No further information on socials)
    * Worked as an Machine learning and Deep learning Intern at various companies including Amazon
  * **Alexandros Papengelis** - He has been very active as a research contributor alongside his job and interests lies in NLP, understanding conversational network, user sentiments etc.
    * ![Screenshot 2022-11-28 at 12.22.38.png](text://image?imageFileName=Screenshot%202022-11-28%20at%2012.22.38.png)![Screenshot 2022-11-28 at 12.24.03.png](text://image?imageFileName=Screenshot%202022-11-28%20at%2012.24.03.png)
  * **Seokwan Kim**
    * He finished his bachelors and PhD from Pohand university of science and technology in South Korea.
    * Currently, working at Amazon Alexa and research lies in understanding Conversational Networks, understanding Dialog System with Natural Language.
    * Ver much active as a researcher and contributor
    * <https://seokhwankim.github.io>
  * **Dilek Hakkani-Tur**
    * ![Screenshot 2022-11-28 at 12.46.39.png](text://image?imageFileName=Screenshot%202022-11-28%20at%2012.46.39.png)![Screenshot 2022-11-28 at 12.46.58.png](text://image?imageFileName=Screenshot%202022-11-28%20at%2012.46.58.png)
* Social Impact Assessor

  **Positive Impact:**

  \-  GCN could be used to develop bots that can carry out more specialized conversations.

  **Negative Impact:**

  \-  We could face negative scenarios if these types of bots are developed because they would produce more convincing conversations and users may tend to trust the bot and would not question if the information being shared is false.

  \- Develop bots that could replace people who work in customer service (chats).

  \- The GCN is trained with data information from web sources which as we already know has a bias towards majoritarian groups, which can foster discrimination towards minority groups.

#### **Week 8**

Paper#1 title: Integrating Text and Image: Determining Multimodal Document Intent in Instagram Posts <https://aclanthology.org/D19-1469/>

Roles:

* Scientific Peer Reviewer 

  MIRO Board : <https://miro.com/app/board/uXjVP9aqQXo=/>
* Archaeologist (Yi-Sheng)
  * 39 references
    * Emily E Marsh and Marilyn Domas White. 2003. A taxonomy of relationships between images and text. Journal of Documentation, 59(6):647–672.
      * Contribute to the taxonomy methodology of the current paper, which nevertheless includes modification to adapt to modern medium and scenarios (e.g., Instagram)
      * Research question: how does an illustration relate to the text with which it is associated, or, what are the functions of illustration?
  * ![截圖 2022-12-04 00.36.31.png](text://image?imageFileName=%E6%88%AA%E5%9C%96%202022-12-04%2000.36.31.png)
    * Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In CVPR, pages 770–778.
      * Modeling: image encoder
    * Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Cor- rado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In *NIPS*, pages 3111–3119.

      Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word rep- resentations. *arXiv preprint arXiv:1802.05365*.
      * Modeling: text encoder
  * Cited 61 times
  * The following papers mostly focus on cross-modal (corresponding to the focused representation) and intention (owing to the semiotic aspect)
    * KIELA, Douwe, et al. The hateful memes challenge: Detecting hate speech in multimodal memes. Advances in Neural Information Processing Systems, 2020, 33: 2611-2624.
      * Published by Facebook AI Team; inspired by the fact that either image or caption could completely alter or even reverse the meaning of a post in meme culture
  * 
* Academic Researcher
  * two approaches: either look at the model or the dataset
  * model and dataset are build for intent classification in a social media context
    * evaluate model performance on datasets of other domains: **news texts** ( might be difficult because for factual than personal)
    * additions to the taxonomie to caputre more subtle intent in specialised areas
  * the word embeddings were done using word2vec -poorer performance than character based system: use of pretrained transfomer models
* Industry Practitioner (Guillem Gili i Bueno)
  * Dataset
    * Very nicely-balanced dataset, good taxonomy. 
    * Decently varied too (entertainment may overlap a lot, but that is just modern media). 
    * Biased towards bigger influencers (which will have more clear intent, but may be worse for detecting edge cases). Would be interesting to know user variety in posts. 
    * <https://github.com/karansikka1/documentIntent_emnlp19> (no license so public domain)
  * Use cases:
    * First-impression reactions can often help predict stock movements (being able to be the fastest to react properly when someone tweets/publishes has value, also analyzing the immediate responses of the rest)
    * Market segmenting(we can %)
      * General analysis to news responses
        * Ucraine war, post sentiment? (not necessarily good, but propaganda is as old as news)
      * E.g. we throw a new product. How is the intent of the people that talk about it segmented?
        * Provocative/Expressive: We know how our clients feel
        * Expressive/Entertainment: Get an idea of how clients feel about its artistic/aesthetic value
        * Promotive/Informative: How collaborative is the community around our product(important for RPi)
* Private Investigator
* Social Impact Assessor

  Possible applications that this study can contribute to (among others):
* **Detection and study of propaganda**
* **Detection and study of advertisement**

  Positive and negative aspects:
* (+) **Deeper understanding on how persuasive text work**. News media are analyzed not only based on what they say (the text), but also on what they show (the images)
* (-) **Problem: restrictions on releasing image data** collected from Instagram could be an obstacle for researchers that want to conduct additionaly experiments based on this paper

---

### Paper#2 title: Visually Grounded Reasoning across Languages and Cultures - Liu, Bugliarello et al

Roles:

#### Scientific Peer Reviewer (Robin)

Paper: <https://aclanthology.org/2021.emnlp-main.818/>

Video: <https://aclanthology.org/2021.emnlp-main.818.mp4>

Datasets accessible under: <https://marvl-challenge.github.io/download>

![grafik (4).png](text://image?imageFileName=grafik%20%284%29.png)![grafik (6).png](text://image?imageFileName=grafik%20%286%29.png)![grafik (17).png](text://image?imageFileName=grafik%20%2817%29.png)

##### TASK:

Visual Language Tasks (V/L)

\- - - Image Captioning

\- - - - - Classif. True or False | given an image caption, MLM, MRM, Word Region Alignment, etc...

##### PURPOSE OF PRESENTED PAPER:

1. Identify **biases** in existing V/L databases
2. Introduce a new protocol for creating datasets that better represent the **global diversity of languages & cultures**

##### A FEW IMPORTANT TERMS / DEFINITIONS:

::: info
**Concept**: Mental representation of a category (e.g. BIRD), where instances ob objects and events with similar properties are grouped together

:::

::: info
**NLVR2:** dataset containing NL sentences grounded in images

**ImageNet:** dataset that depicts WordNet hierarchy in images (**WordNet**: Lexical database of English langugae

:::

::: info
**MaRVL**: Multicultural Reasoning over Vision & Language - dataset that authors have created using their proposed protocol

:::

##### PROBLEM DESCRIPTION:

The most used database ImageNet (<https://image-net.org/>)

\--> is derived from English lexical lists and image queries in English language

\--> provides a source with European / North American bias

##### IDENTIFYING BIASES:

::: warn
**__Concepts in ImageNet are not universal__**

:::

Covers concepts familiar to people in the Anglosphere but exotic to rest of the world (-> tailgating) [Link](https://en.wikipedia.org/wiki/Tailgate_party#/media/File:Benstailgate.JPG)

\+ Vice versa

How universal are they? Ideas:

* For concepts in ImageNet, check in how many languages a Wikipedia entry is available (Fig. 2)

  ![grafik (11).png](text://image?imageFileName=grafik%20%2811%29.png)--> many only available in 30 or less languages, most of them from same language family (\*)

::: warn
**__Concepts  are too fine-grained, overly specific (to English?) (s. 2.2)__**

:::

Images where obtained through search engine queries

o   queries only in ENGLISH Spanish Dutch Italian etc.. + Chinese

o   search engines don’t represent real distributions, they customize results to user

o   cleanup via Amazon Mechanical Turk, users probably not representative of global diversity

##### PROPOSED SOLUTION:

::: success
New Protocol to derive a more balanced ImageNet alternative that is representative of more languages and cultures.

:::

Ideas:

Let native speakers decide what concepts and visual representations are important to them

Feature a diverse set of languages

> **__Select Languages__ |** Universal concepts | Lang.-spec. concepts | Image selection | Caption annotation

* ID, ZH, SW, TA, TR *(Indonesian, Mandarin Chinese, Swahili, Tamil, Turkish)*
* Includes low-resource languages (SW, TA)
* covers different language families
* different writing systems

> Select Languages **| \__Universal concepts \_\_**| Lang.-spec. concepts | Image selection | Caption annotation

* Find human universals based on Ethnographic studies / Comparative Linguistics most important semantic field

  \-> Intercontinental Dictionary Series
* retaining 18/22 semantic fields that are the most concrete (e.g. MUSICAL INSTRUMENT)

> Select Languages | Universal concepts | **\__Lang.-spec. concepts \_\_**| Image selection | Caption annotation

* 5 native speakers provide Wikipedia links for 5-10 specific concepts in their culture (e.g. Koto
* retain 5 top picks

⇒  18 \* 5 = 96 concepts per language, representative of the annotators

> Select Languages **|** Universal concepts | Lang.-spec. concepts | **\__Image selection \_\_**| Caption annotation

* 2 (\*) native annotators select images (guidelines s. 3.4) to find complex images *(to require compositional reasoning not just object detection)*

> Select Languages **|** Universal concepts | Lang.-spec. concepts | Image selection | **__Caption annotation__**

* make 4 image pairs and write captions (2 True, 2 False)
* repeat (here: x4)
* native translators from [proZ.com](http://proZ.com) translate the captions into other languages
* Validators resolve conflicting cases of potentially wrong labels

##### PROPERTIES OF NEW DATASET (MaRVL):

::: info
Accuracy and inter-annotator agreement of native speakers in labelling task are **very high**

:::

![grafik (12).png](text://image?imageFileName=grafik%20%2812%29.png) .... extract features of MaRVL vs NLVR2 using a deep NN called ResNet50 ...

::: info
image selection appears to be more evenly distributed in MaRVL than in ImageNet

:::

(less fine-grained concepts)

![grafik (14).png](text://image?imageFileName=grafik%20%2814%29.png)

::: info
Many concepts can’t be found in English based WordNet

:::

::: error
Limitations (acc. to authors) :

:::

* too few annotators for low-resource languages ⇒ individual bias
* incompleteness of Wikipedia (esp. for certain languages)
* only 5 concepts per semantic field

##### BASELINES:

Testing zero shot performance on MaRVL database using a customized version of state-of-the-art (at the time of publication)

![grafik (15).png](text://image?imageFileName=grafik%20%2815%29.png)<https://production-media.paperswithcode.com/methods/6bf9dc19-6dfb-4c31-a71a-450dd32d9850.jpg>

##### INTERPRETING THE RESULTS:

![grafik (16).png](text://image?imageFileName=grafik%20%2816%29.png)

**__Testing state-of-the-art models (trained on traditional databases) shows:__**

1. zero-shot (no fine-tuning at all)
2. translate test (translating the data from MaRVL into ENglish before test)

Findings:

::: warn
Models perform well on English data but struggle with MaRVL

\--> out-of-distribution nature of MaRVL data and cross-lingual transfer

:::

**Notes**:

Trying various neural architectures shows

best performances are pretty evenly distributed

* neural architecture itself can be ruled out as the decisive factor

Translating ZH manually into English in translation test shows:

only little improvement in performance

* noise created from automatic translation can be ruled out as decisive factor

##### CONCLUSION (simplified):

**Concepts and images** documented in visiolinguistic datasets are often **neither salient nor prototypical for speakers of many languages** (other than English)

**MaRVL protocol mitigates these biases** by letting native speakers of diverse languages drive selection steps

Baselines show that current models perform poorly on dataset (compared to English-based datasets)

\--> New dataset is a better estimate to assess models capabilities in real-life applications

##### *Poor formatting & other problems:*

*“Introduction” - pCcnIIw*

*missing source for Tzeltal Mayans*

*term "synset" not explained*

*“anecdotally” (Koto instrument)*

*“International” (instead of Intercontinental) Dictionary Series*

---

Archaeologist(Md Delwar Hossain):

* Cited 30 times
  * Alane Suhr, Mike Lewis, James Yeh, and Yoav Artzi. 2017. [A Corpus of Natural Language for Visual Reasoning](https://aclanthology.org/P17-2034). In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 217–223, Vancouver, Canada. Association for Computational Linguistics.
    * Integration of information across modalities. 
  * Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. 2019. [__A Corpus for Reasoning about Natural Language Grounded in Photographs__](https://aclanthology.org/P19-1644). In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 6418–6428, Florence, Italy. Association for Computational Linguistics.
    * Deep linguistics Understanding. 

Academic Researcher

* They have developed a new annotation protocol, we can use that to **extend the dataset to include other language families**.
  * Specifically, with these 5 languages, take each one individually, including languages in its family through this new process, and then evaluate their performance.
* We can use this dataset to **perform other tasks** for these languages, generating captions, object recognition, improve translation amongst them.
* Industry Practitioner:
  * I am working at Amazon on a project with the department that works with the online catalog. They work with customers and have received several complaints that product descriptions and images, especially in the China marketplace, do not match.

    At this time, to reduce the amount of misdescribed products in the department, there are people doing manual searches and correcting the descriptions or images. However, this **manual search requires a lot of time** and people who speak the language of the marketplace (Mandarin Chinese). Therefore, I am leading a project which goal is to develop a tool that allows an automated review of these listings on the platform and that identifies if the descriptions and images are for the same product.

    For this, the dataset developed by the authors in the paper, MaRVL, could be implemented because it is a multilingual and multicultural dataset this would let us develop an effective tool for  specific marketplaces (as China).

    By developing the tool with this dataset, the company would be able to reduce costs because they will have and **automatic tool that identifying if the product description and the images match** and they would not have to hire people to do the manual searches. Additionally, they would be able to cover non-English speaking marketplaces.
* Private Investigator

  **2021 best long paper ward by emnlp**

  **Paper cited 30 times**

  **Fangyu LIU**

  ·      PhD student in NLP at University of Cambridge

  ·      He has published 45 papers

  ·      [__(3DCNN-DQN-RNN: A Deep Reinforcement Learning Framework for Semantic Parsing of Large-scale 3D Point Clouds__](https://scholar.google.ca/citations?view_op=view_citation&hl=en&user=d19PiS0AAAAJ&citation_for_view=d19PiS0AAAAJ:d1gkVwhDpl0C) )has the most citation of 108 released 2017

  ·      He worked as an applied scientist intern at amazon (published a paper while working there (unsupervised sentence representation at ICLR 2022)and also working as a research intern at google

  .    Interest - applying machine learning methods to (multi-modal) language representation learning

  ·    Linked in profile <https://www.linkedin.com/in/fangyu-liu-48a003b0/>

  **Emanuele Bugliarello**

  ·      PHD student in Computer science at University of Copenhagen

  ·      2 bachelors degree

  ·      Telecommunications engineering - politecnico di torino Italy

  ·      bachelors degree electronics and information engineering - Tongji university China 29.34/30

  ·      Masters in communications systems -  EPFL (École polytechnique fédérale de Lausanne) Grade: 5.53/6

  ·      He has interned in many places some notable ones are Reseach scientist intern at google and spotify

  ·      github  [__https://e-bug.github.io/projects/__](https://e-bug.github.io/projects/)

  **Edardo Maria Ponti**

  ·      Currently working as a Lecturer in Natural Language Processing, University of Edinburgh

  ·      47 total publications

  ·      Phd computational linguistics University of Cambridge

  ·      Masters in computational linguistics at Università di Pavia Italy

  ·      [__https://www.linkedin.com/in/edoardo-maria-ponti/?originalSubdomain=uk__](https://www.linkedin.com/in/edoardo-maria-ponti/?originalSubdomain=uk)

  **Siva reddy**

  ·      Currently working as Assistant Professor  McGill University

  ·      Total 70 publications

  ·      Postdoctoral Researcher - Computer Science t, Stanford University

  ·      PhD, Informatics - University of Edinburgh, UK

  ·      He has 2 masters

  ·      M.S. by Research, Computer Science - University of York, UK

  ·      M.S. by Research, Computer Science - IIIT Hyderabad, India

  ·      Research Intern at  Google Research

  .      Many areas of interests - Probing Deep Learning Models (Bias, Interpretability)Compositionality and Reasoning , Language in Grounded Environments (text worlds, vision, robotics) etc.

  ·      Active contributor in github

  ###### Nigel Collier

  .       Professor of Natural Language Processing at the University of Cambridge

  ·       He was supervisor for this paper

  .       PHD computational linguistics -  University of Manchester

  .       MSC in machine transalation

  .       Authored and co-authored 293 publications

  .       Paper ([__Sentiment analysis using support vector machines with diverse information sources__](https://scholar.google.de/citations?view_op=view_citation&hl=en&user=ZMelBa0AAAAJ&citation_for_view=ZMelBa0AAAAJ:d1gkVwhDpl0C)  )   published in 2004 – has total 1013 citations

  .        Research interests are creating more human-like models for natural language understanding and applications with the potential for tangible social impact

  ###### Desmond Elliot

  ·      Currently working as Assistant Professor at the University of Copenhagen

  ·      Has done 3 postdoctoral research

  ·      Postdoctoral Researcher - School of Informatics, University of Edinburgh (new methods in multi- modal machine translation)

  ·      Postdoctoral Researcher - Institute for Logic, Language and Computation (ILLC) (multilingual multimodal modelling.)

  ·      Postdoctoral researcher - Centrum Wiskunde & Informatica (vision and language modelling )

  ·      56 total publications

  .      Reviewer for EMNLP, AAAI, NAACL, ACL, EMNLP, COLING, TACL, ICLR

  ·      Phd computer science - University of Edinburgh
* Social Impact Assessor
* +
* -