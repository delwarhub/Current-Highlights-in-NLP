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
