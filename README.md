<div align="center">
    <img src="title image.jpg" alt="Logo" width="" height="200">
<h1 align="center">Speech Recognition</h1>
</div>

## 1. Problem Statement
Speech recognition, also known as automatic speech recognition (ASR), is a technology that enables the conversion of spoken language into written text. It is a complex task that involves understanding and processing human speech, which varies widely in terms of accents, pronunciations, speed, and context. Deep learning has revolutionized speech recognition by providing powerful models that can learn from vast amounts of audio data, improving the accuracy and robustness of ASR systems. 

The primary objective of speech recognition is to develop a model that can accurately transcribe spoken words into written text. This involves several key challenges:

- **Acoustic Variability:**  Handling different accents, intonations, and speaking speeds.
- **Background Noise:** Distinguishing speech from various background noises.
- **Context Understanding:** Recognizing and interpreting words within the context of a sentence.
- **Vocabulary Size:** Managing a large vocabulary and differentiating between similar-sounding words.

### Applications

- **Voice Assistants:** Powering virtual assistants like Siri, Alexa, and Google Assistant.
- **Transcription Services:** Automating the transcription of meetings, interviews, and lectures.
- **Accessibility:** Assisting individuals with hearing impairments by providing real-time text transcriptions.
- **Call Centers:** Automating customer service interactions and analyzing call center data.


## 2. Related Works
This section explores existing research and solutions related to speech recognition using deep learning models from 2017 to 2023, including the exact models used, descriptions, and links to their papers or GitHub repositories:

| Date | Title                                  | Description                                                                                                 | Links                                                                                              |
|------|----------------------------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| 2017 | Deep Speech 2                          | Utilizes Recurrent Neural Networks (RNN) with Connectionist Temporal Classification (CTC) loss.             | [Paper](https://arxiv.org/abs/1512.02595) [GitHub](https://github.com/baidu-research/warp-ctc)     |
| 2018 | Transformer ASR                        | Uses Transformer architecture for speech recognition, emphasizing self-attention mechanisms.                 | [Paper](https://arxiv.org/abs/1809.08895) [GitHub](https://github.com/espnet/espnet)               |
| 2019 | QuartzNet                              | A convolutional neural network (CNN) model using 1D time-channel separable convolutions for ASR.             | [Paper](https://arxiv.org/abs/1910.10261) [GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) |
| 2020 | Wav2Vec 2.0                            | Combines CNN and Transformer models for self-supervised learning of speech representations.                  | [Paper](https://arxiv.org/abs/2006.11477) [GitHub](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) |
| 2020 | Conformer-CTC                          | Integrates convolutional layers with Transformer layers for enhanced ASR performance.                        | [Paper](https://arxiv.org/abs/2006.11477) [GitHub](https://github.com/espnet/espnet)               |
| 2021 | HuBERT                                 | Utilizes a masked prediction task to learn hidden representations of speech.                                 | [Paper](https://arxiv.org/abs/2106.07447) [GitHub](https://github.com/pytorch/fairseq/tree/main/examples/hubert) |
| 2022 | Whisper                                | OpenAI's ASR model designed for robustness across different languages and accents.                           | [Paper](https://openai.com/research/whisper) [GitHub](https://github.com/openai/whisper)           |
| 2023 | Efficient Conformer                    | A variant of Conformer with efficient computation tailored for low-latency ASR applications.                 | [Paper](https://arxiv.org/abs/2305.00359) [GitHub](https://github.com/espnet/espnet)               |
| 2023 | E-Branchformer                         | A hybrid model combining Branchformer architecture with enhancements for ASR tasks.                          | [Paper](https://arxiv.org/abs/2305.00120) [GitHub](https://github.com/espnet/espnet)               |



## 3. The Proposed Method
Transformers, originally introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, have revolutionized natural language processing (NLP) by relying solely on self-attention mechanisms, foregoing recurrent architectures entirely. This innovation has since been adapted for Automatic Speech Recognition (ASR), resulting in substantial improvements.
Transformers have significantly advanced the field of speech recognition, offering powerful models that improve accuracy, efficiency, and scalability. 

Transformers are used in end-to-end models where the raw audio or its features are directly mapped to text without intermediate steps like phoneme recognition. Also, Transformers are combined with other architectures, such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs), to leverage the strengths of multiple approaches. In this task, we used a CNN as featuer embedding method combined with transformer layer.

<div align="center">
    <img src="title method.jpg" alt="Logo" width="" height="300">
<h4 align="center">Proposed method</h4>
</div>

## 4. Implementation
This section delves into the practical aspects of the project's implementation.

### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for the medical image segmentation task. It includes details about the dataset source, size, composition, preprocessing, and loading applied to it.
[Dataset](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data)

### 4.2. Model
In this subsection, the architecture and specifics of the deep learning model employed for the segmentation task are presented. It describes the model's layers, components, libraries, and any modifications made to it.

### 4.3. Configurations
This part outlines the configuration settings used for training and evaluation. It includes information on hyperparameters, optimization algorithms, loss function, metric, and any other settings that are crucial to the model's performance.

### 4.4. Train
Here, you'll find instructions and code related to the training of the segmentation model. This section covers the process of training the model on the provided dataset.

### 4.5. Evaluate
In the evaluation section, the methods and metrics used to assess the model's performance are detailed. It explains how the model's segmentation results are quantified and provides insights into the model's effectiveness.

