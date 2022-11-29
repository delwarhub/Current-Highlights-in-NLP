#### 

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
* Archaeologist
* Academic Researcher
* Industry Practitioner (Guillem Gili i Bueno)
  * Very nicely-balanced dataset, good taxonomy. Decently varied too (entertainment may overlap a lot, but that is just modern media). Biased towards bigger influencers (which will have more clear intent, but may be worse for detecting edge cases). Would be interesting to know user variety in posts. -> <https://github.com/karansikka1/documentIntent_emnlp19> (no license so public domain)
  * ![image (15).png](text://image?imageFileName=image%20%2815%29.png)
  * Accuracy is a weird pick for a metric, rather than F1.
  * First impression reactions can often help predict stock movements(being able to be the fastest to react properly when someone tweets/publishes has value, also analyzing the inmediate responses of the rest)
  * Market segmenting(we can %
    * E.g. we throw a new product. How is the intent of the people that talk about it segmented?
      * Provocative/Expressive: We know how our clients feel
      * Expressive:
  * 
* Private Investigator
* Social Impact Assessor

Paper#2 title:

Roles:

* Scientific Peer Reviewer
* Archaeologist
* Academic Researcher
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

#### **Week 9**

#### 

Paper#1 title:

Roles:

* Scientific Peer Reviewer
* Archaeologist
* Academic Researcher
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

Paper#2 title:

Roles:

* Scientific Peer Reviewer
* Archaeologist
* Academic Researcher
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

#### **Week 10**

**NO PAPER READING on January 2, 2023**

#### **Week 11**

#### 

Paper#1 title:

Roles:

* Scientific Peer Reviewer
* Archaeologist
* Academic Researcher
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

Paper#2 title:

Roles:

* Scientific Peer Reviewer
* Archaeologist
* Academic Researcher
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

#### **Week 12**

#### 

Paper#1 title:

Roles:

* Scientific Peer Reviewer
* Archaeologist
* Academic Researcher
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

Paper#2 title:

Roles:

* Scientific Peer Reviewer
* Archaeologist
* Academic Researcher
* Industry Practitioner
* Private Investigator
* Social Impact Assessor

#### 