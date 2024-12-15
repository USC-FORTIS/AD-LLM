# GPT o1-preview version
# the prompt for selecting an unsupervised model for anomaly detection a specific text dataset
# note that o1-preview only accept user prompt
# if you wish to use this prompt in other GPT model like GPT-4o
# you can adjust the prompt accordingly
# uncomment system_message and assistant_message, and modify the user_message
def generate_model_selection_prompt(name, size, original_task, normal_label_list, anomaly_label_list,
                                    avg_len, max_len, min_len, std_len, normal_text, anomaly_text):
    # prepare label categories
    normal_categories = ", ".join(
        [f'{category}' for category in normal_label_list])
    anomaly_categories = ", ".join(
        [f'{category}' for category in anomaly_label_list])

    # system_message = "You are an expert in model selection for anomaly detection on text datasets."

    user_message = f"""
You are an expert in model selection for anomaly detection on text datasets.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Dataset Size: {size}
- Background: This dataset is originally for {original_task}.
- Data Structure: Textual data with multiple categories. One category is considered anomalous, while the others are normal.
    - Normal Category(ies): {normal_categories}
        - An Example: {normal_text}
    - Anomaly Category: {anomaly_categories}
        - An Example: {anomaly_text}
-  Text Length Statistics:
    - Average Length: {avg_len}
    - Maximum Length: {max_len}
    - Minimum Length: {min_len}
    - Standard Deviation: {std_len}

## Model Information:
- Models utilize language models to generate embeddings and feed the embeddings into the models.
- We provide the abstracts of the papers that introduce the models for your reference.
### Model Options:
- AutoEncoder (AE): Autoencoders are a natural choice for outlier detection because they are commonly used for dimensionality reduction of multidimensional data sets as an alternative to PCA or matrix factorization. In fact, one can divide the autoencoder on two sides of the middle hidden layer into two portions corresponding to an encoder and a decoder, which provides a point of view very similar to that in matrix factorization and PCA.
- Deep Support Vector Data Description (DeepSVDD): Despite the great advances made by deep learning in many machine learning problems, there is a relative dearth of deep learning approaches for anomaly detection. Those approaches which do exist involve networks trained to perform a task other than anomaly detection, namely generative models or compression, which are in turn adapted for use in anomaly detection; they are not trained on an anomaly detection based objective. In this paper we introduce a new anomaly detection method—Deep Support Vector Data Description—, which is trained on an anomaly detection based objective. The adaptation to the deep regime necessitates that our neural network and training procedure satisfy certain properties, which we demonstrate theoretically. We show the effectiveness of our method on MNIST and CIFAR-10 image benchmark datasets as well as on the detection of adversarial examples of GTSRB stop signs.
- Empirical-Cumulative-Distribution-Based Outlier Detection (ECOD): Outlier detection refers to the identification of data points that deviate from a general data distribution. Existing unsupervised approaches often suffer from high computational cost, complex hyperparameter tuning, and limited interpretability, especially when working with large, high-dimensional datasets. To address these issues, we present a simple yet effective algorithm called ECOD (Empirical-Cumulative-distribution-based Outlier Detection), which is inspired by the fact that outliers are often the "rare events" that appear in the tails of a distribution. In a nutshell, ECOD first estimates the underlying distribution of the input data in a nonparametric fashion by computing the empirical cumulative distribution per dimension of the data. ECOD then uses these empirical distributions to estimate tail probabilities per dimension for each data point. Finally, ECOD computes an outlier score of each data point by aggregating estimated tail probabilities across dimensions. Our contributions are as follows: (1) we propose a novel outlier detection method called ECOD, which is both parameter-free and easy to interpret; (2) we perform extensive experiments on 30 benchmark datasets, where we find that ECOD outperforms 11 state-of-the-art baselines in terms of accuracy, efficiency, and scalability; and (3) we release an easy-to-use and scalable (with distributed support) Python implementation for accessibility and reproducibility.
- Isolation Forest (IForest): Anomalies are data points that are few and different. As a result of these properties, we show that, anomalies are susceptible to a mechanism called isolation. This paper proposes a method called Isolation Forest (iForest) which detects anomalies purely based on the concept of isolation without employing any distance or density measure—fundamentally different from all existing methods. As a result, iForest is able to exploit subsampling (i) to achieve a low linear time-complexity and a small memory-requirement, and (ii) to deal with the effects of swamping and masking effectively. Our empirical evaluation shows that iForest outperforms ORCA, one-class SVM, LOF and Random Forests in terms of AUC, processing time, and it is robust against masking and swamping effects. iForest also works well in high dimensional problems containing a large number of irrelevant attributes, and when anomalies are not available in training sample.
- Local Outlier Factor (LOF): For many KDD applications, such as detecting criminal activities in E-commerce, finding the rare instances or the outliers, can be more interesting than finding the common patterns. Existing work in outlier detection regards being an outlier as a binary property. In this paper, we contend that for many scenarios, it is more meaningful to assign to each object a degree of being an outlier. This degree is called the local outlier factor (LOF) of an object. It is local in that the degree depends on how isolated the object is with respect to the surrounding neighborhood. We give a detailed formal analysis showing that LOF enjoys many desirable properties. Using real-world datasets, we demonstrate that LOF can be used to find outliers which appear to be meaningful, but can otherwise not be identified with existing approaches. Finally, a careful performance evaluation of our algorithm confirms we show that our approach of finding local outliers can be practical.
- Unifying Local Outlier Detection Methods via Graph Neural Networks (LUNAR): Many well-established anomaly detection methods use the distance of a sample to those in its local neighbourhood: so-called `local outlier methods', such as LOF and DBSCAN. They are popular for their simple principles and strong performance on unstructured, feature-based data that is commonplace in many practical applications. However, they cannot learn to adapt for a particular set of data due to their lack of trainable parameters. In this paper, we begin by unifying local outlier methods by showing that they are particular cases of the more general message passing framework used in graph neural networks. This allows us to introduce learnability into local outlier methods, in the form of a neural network, for greater flexibility and expressivity: specifically, we propose LUNAR, a novel, graph neural network-based anomaly detection method. LUNAR learns to use information from the nearest neighbours of each node in a trainable way to find anomalies. We show that our method performs significantly better than existing local outlier methods, as well as state-of-the-art deep baselines. We also show that the performance of our method is much more robust to different settings of the local neighbourhood size.
- Single-Objective Generative Adversarial Active Learning (SO-GAAL): Outlier detection is an important topic in machine learning and has been used in a wide range of applications. In this paper, we approach outlier detection as a binary-classification issue by sampling potential outliers from a uniform reference distribution. However, due to the sparsity of data in high-dimensional space, a limited number of potential outliers may fail to provide sufficient information to assist the classifier in describing a boundary that can separate outliers from normal data effectively. To address this, we propose a novel Single-Objective Generative Adversarial Active Learning (SO-GAAL) method for outlier detection, which can directly generate informative potential outliers based on the mini-max game between a generator and a discriminator. Moreover, to prevent the generator from falling into the mode collapsing problem, the stop node of training should be determined when SO-GAAL is able to provide sufficient information. But without any prior information, it is extremely difficult for SO-GAAL. Therefore, we expand the network structure of SO-GAAL from a single generator to multiple generators with different objectives (MO-GAAL), which can generate a reasonable reference distribution for the whole dataset. We empirically compare the proposed approach with several state-of-the-art outlier detection methods on both synthetic and real-world datasets. The results show that MO-GAAL outperforms its competitors in the majority of cases, especially for datasets with various cluster types or high irrelevant variable ratio.
- Variational AutoEncoder (VAE): How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions are two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.
### Embedding Options:
- Bidirectional Encoder Representations from Transformers (BERT): We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
- "text-embedding-3-large" from OpenAI (referred to as OpenAI): Text embeddings are useful features in many applications such as semantic search and computing text similarity. Previous work typically trains models customized for different use cases, varying in dataset choice, training objective and model architecture. In this work, we show that contrastive pre-training on unsupervised data at scale leads to high quality vector representations of text and code. The same unsupervised text embeddings that achieve new state-of-the-art results in linear-probe classification also display impressive semantic search capabilities and sometimes even perform competitively with fine-tuned models. On linear-probe classification accuracy averaging over 7 tasks, our best unsupervised model achieves a relative improvement of 4% and 1.8% over previous best unsupervised and supervised text embedding models respectively. The same text embeddings when evaluated on large-scale semantic search attains a relative improvement of 23.4%, 14.7%, and 10.6% over previous best unsupervised methods on MSMARCO, Natural Questions and TriviaQA benchmarks, respectively. Similarly to text embeddings, we train code embedding models on (text, code) pairs, obtaining a 20.8% relative improvement over prior best work on code search.

## Rules:
1. Availabel options include "BERT+AE", "BERT+DeepSVDD", "BERT+ECOD", "BERT+iForest", "BERT+LOF", "BERT+LUNAR", "BERT+SO-GAAL", "BERT+VAE", "OpenAI+AE", "OpenAI+DeepSVDD", "OpenAI+ECOD", "OpenAI+iForest", "OpenAI+LOF", "OpenAI+LUNAR", "OpenAI+SO-GAAL", "OpenAI+VAE."
2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
3. Response Format:
    - Provide responses in a strict **JSON** format with the keys "reason" and "choice."
        - "reason": Your explanation of the reasoning.
        - "choice": The model you have selected for anomaly detection in this dataset.

Response in JSON format:
"""

    # assistant_message = f"Response in JSON format:\n"

    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        # {"role": "assistant", "content": assistant_message}
    ]

    return messages
