# Notes on Machine Learning and Medical Research Papers

A collection of research paper summaries, on machine learning and medical (brain computer interface and vision). ML papers are mainly on solving computer vision or sequential problems, and medical papers are focusing on vision problems. 

Papers are sorted by topics and tags. Go to the [Issues](https://github.com/jinglescode/papers/issues) tab to browse, search and filter research papers. 

## Table of Contents

- **[Machine Learning](#machine-learning)**
    - [Computer Vision](#computer-vision)
    - [Sequential](#sequential)
    - [Sequential: Transformer](#sequential-transformer)
    - [Representation learning](#representation-learning)

- **[Medical](#medical)**
    - [Brain computer interface](#brain-computer-interface)
    - [Vision](#vision)

---

# Machine Learning

## Computer vision

[Interpretable and Fine-Grained Visual Explanations for Convolutional Neural Networks](https://github.com/jinglescode/ml-papers/issues/1)
- produce mask to focus on interpretability
- smallest region of image must be retained to preserve (or deleted to change) model output
- fine grain visual explanation, no smoothing and regularisations

[Stand-Alone Self-Attention in Vision Models](https://github.com/jinglescode/ml-papers/issues/21)
- self-attention can be an effective stand-alone layer

[On the relationship between self-attention and convolutional layers](https://github.com/jinglescode/papers/issues/22)
- attention layers can perform convolution, they learn to behave similar to convolutional layers
- multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer

[Dynamic Convolution: Attention over Convolution Kernels](https://github.com/jinglescode/papers/issues/27)
- increases model complexity without increasing the network depth or width
- single convolution kernel per layer, dynamic convolution aggregates multiple parallel convolution kernels dynamically based upon their attentions, which are input dependent
- can be easily integrated into existing CNN architectures

[Dynamic Group Convolution for Accelerating Convolutional Neural Networks](https://github.com/jinglescode/papers/issues/29)
- propose dynamic group convolution (DGC) that adaptively selects which part of input channels to be connected within each group for individual samples on the fly
- introduce a tiny auxiliary feature selector for each group to dynamically decide which part of input channels to be connected based on the activations of all of input channels
- Multiple groups can adaptively capture abundant and complementary visual/semantic features for each input image
- similar computational efficiency as the conventional group convolution simultaneously

[An image is worth 16x16 words: Transformers for image recognition at scale](https://github.com/jinglescode/papers/issues/50)
- global image attention by patches
- learn to attend to patches further away at the lower layers which convnet cannot

[End-to-End Video Instance Segmentation with Transformers](https://github.com/jinglescode/papers/issues/60)
- end-to-end instance segmentation on video frames, tracking the object across frames
- achieves the highest speed among all existing video instance segmentation models, and achieves the best result

[Deep learning-enabled medical computer vision](https://github.com/jinglescode/papers/issues/63)
- four key considerations when applying ML technologies in healthcare:
    1. assessment of data,
    2. planning for model limitations,
    3. community participation, and
    4. trust building

[Bottleneck Transformers for Visual Recognition](https://github.com/jinglescode/papers/issues/64)
- incorporates self-attention in ResNet's bottleneck blocks, improves instance segmentation and object detection while reducing the parameters.
- convolution and self-attention can beat ImageNet benchmark, pure attention ViT models struggle in small data regime, but shine in large data regime.


## Sequential

[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://github.com/jinglescode/ml-papers/issues/14)
- Authors proposed and results show that TCN outperforms RNN, LSTM, and GRU; across a broad range of sequence modeling tasks.
- TCNs exhibit substantially longer memory, and are thus more suitable for domains where a long history is required.

[Machine translation of cortical activity to text with an encoder–decoder framework](https://github.com/jinglescode/ml-papers/issues/15)
- RNN decoder encoder sequence-to-sequence network that act like a language translation machine, from ECoG to words. Building a representation to map between the 2 different sources.

[Speech synthesis from neural decoding of spoken sentences](https://github.com/jinglescode/ml-papers/issues/16)
- translates neural activity into speech

[Wavenet: A generative model for raw audio](https://github.com/jinglescode/ml-papers/issues/17)
- a deep generative model of audio data that operates directly at the waveform level. WaveNets are autoregressive and combine causal filters with dilated convolutions to allow their receptive fields to grow exponentially with depth, which is important to model the long-range temporal dependencies in audio signals.
- promising results when applied to music audio modeling and speech recognition

[Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation](https://github.com/jinglescode/ml-papers/issues/18)
- fully-convolutional time-domain audio separation network consists of three processing stages: encoder, separation, and decoder

[Convolutional Sequence to Sequence Learning](https://github.com/jinglescode/ml-papers/issues/19)
- fully convolutional model for sequence to sequence learning
- use of gated linear units eases gradient propagation
- separate attention mechanism for each decoder layer
- outperforms strong recurrent models on very large benchmark datasets

[Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://github.com/jinglescode/ml-papers/issues/20)
- fully convolutional encoder and a simple decoder can give superior results to a strong RNN baseline while being an order of magnitude more efficient. Key to the success of the convolutional encoder is a time-depth separable block structure which allows the model to retain a large receptive field

[Parallel wavenet: Fast high-fidelity speech synthesis](https://github.com/jinglescode/papers/issues/23)
- high-fidelity speech synthesis based on WaveNet using Probability Density Distillation

[Tacotron: Towards End-to-End Speech Synthesis](https://github.com/jinglescode/papers/issues/24)
- end-to-end generative text-to-speech model that synthesizes speech directly from characters
- train from <text, audio> pairs, model takes characters as input and outputs raw spectrogram

[Wave-Tacotron: Spectrogram-free end-to-end text-to-speech synthesis](https://github.com/jinglescode/papers/issues/25)
- text encoding is passed to a block-autoregressive decoder using attention, producing conditioning features
- use location-sensitive attention, which was more stable than the non-content-based GMM attention

[Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis](https://github.com/jinglescode/papers/issues/26)
- using simple location-relative attention mechanisms to do away with content-based query/key comparisons, to handle out-of-domain text
- introduce a new location-relative attention mechanism to the additive energy-based family, called Dynamic Convolution Attention (DCA)

[Pay Less Attention with Lightweight and Dynamic Convolutions](https://github.com/jinglescode/papers/issues/28)
- introduce dynamic convolutions which are simpler and more efficient than self-attention
- very lightweight convolution can perform competitively to the best reported self-attention results
- number of operations required by this approach scales linearly in the input length, whereas self-attention is quadratic

[Learning representations from EEG with deep recurrent-convolutional neural networks](https://github.com/jinglescode/papers/issues/30)
- designed to preserve the spatial, spectral, and temporal structure of EEG which leads to finding features that are less sensitive to variations and distortions within each dimension
- robust to inter- and intra-subject differences, as well as to measurementrelated noise

[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://github.com/jinglescode/papers/issues/32)
- a framework for self-supervised learning of representations from raw audio data, wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations

[Improved Noisy Student Training for Automatic Speech Recognition](https://github.com/jinglescode/papers/issues/33)
- adapt and improve noisy student training for automatic speech
- recognition (noisy student training is an iterative self-training method that leverages augmentation to improve network performance)

[Visual to Sound: Generating Natural Sound for Videos in the Wild](https://github.com/jinglescode/papers/issues/36)
- from video frames to audio

[SampleRNN: An unconditional end-to-end neural audio generation model](https://github.com/jinglescode/papers/issues/37)
- able to capture underlying sources of variations in the temporal sequences over very long time spans
- using a hierarchy of modules, each operating at a different temporal resolution. lowest module processes individual samples, and each higher module operates on an increasingly longer timescale and a lower temporal resolution

[Generating Visually Aligned Sound from Videos](https://github.com/jinglescode/papers/issues/38)
- RegNet - video sound generation, visually aligned sound, audio forwarding regularizer
- using GAN, learn a correct mapping between video frames and visually relevant sound

[WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis](https://github.com/jinglescode/papers/issues/73)
- text-to-speech synthesis, synthesizes the waveform directly without using hand-designed intermediate features (e.g., spectrograms)

[Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://github.com/jinglescode/papers/issues/74)
- text to speech synthesis, sequence-to-sequence feature prediction network that maps character embeddings to mel-scale spectrograms, followed by a modified WaveNet model acting as a vocoder to synthesize time-domain waveforms from those spectrograms

## Sequential: Transformer

[Transformer-xl: Attentive language models beyond a fixed-length context](https://github.com/jinglescode/papers/issues/39)
- enables learning dependency beyond a fixed length without disrupting temporal coherence
- resolves the context fragmentation problem

[Compressive transformers for long-range sequence modelling](https://github.com/jinglescode/papers/issues/40)
- compress memory mechanism to compress past memories for long-range sequence learning

[Reformer: The efficient transformer](https://github.com/jinglescode/papers/issues/41)
- more memory efficient and faster on long sequences

[Music transformer: Generating music with long-term structure](https://github.com/jinglescode/papers/issues/42)
- relative attention is very well-suited for generative modeling of symbolic music
- relative attention to much longer sequences such as long texts or even audio waveforms

[Conformer: Convolution-augmented Transformer for Speech Recognition](https://github.com/jinglescode/papers/issues/43)
- combine convolutions with self-attention in ASR models
- self-attention learns the global interaction whilst the convolutions efficiently capture the relative-offset-based local correlations

[Transformer transducer: A streamable speech recognition model with transformer encoders and rnn-t loss](https://github.com/jinglescode/papers/issues/44)
- use the attention in Transformer-XL and apply to speech recognition
- end-to-end speech recognition model with Transformer encoders that can be used in a streaming speech recognition system

[Rethinking Attention with Performers](https://github.com/jinglescode/papers/issues/45)
- proposes a set of techniques called Fast Attention Via positive Orthogonal Random features (FAVOR+) to approximate softmax self attention in Transformers and achieve better space and time complexity when the sequence length is much higher than feature dimensions
- Performers, is provably and practically accurate in estimating regular full-rank attention without relying on any priors such as sparsity or low-rankness. It can also be applied to efficiently model other kernalizable attention mechanisms beyond softmax, achieving better empirical results than regular Transformers on some datasets with such strong representation power
- tested on a rich set of tasks including pixel-prediction, language modeling and protein sequence modeling, and demonstrated competitive results with other examined efficient sparse and dense attention models

[Linformer: Self-Attention with Linear Complexity](https://github.com/jinglescode/papers/issues/46)
- self-attention mechanism can be approximated by a low-rank matrix, reduces the overall self-attention complexity from O(n^2) to O(n) in both time and space.

[Transformers are rnns: Fast autoregressive transformers with linear attention](https://github.com/jinglescode/papers/issues/49)
- reformulates the attention mechanism in terms of kernel functions and obtains a linear formulation, which reduces these requirements. Surprisingly, this formulation also surfaces an interesting connection between autoregressive transformers and RNNs

[An image is worth 16x16 words: Transformers for image recognition at scale](https://github.com/jinglescode/papers/issues/50)
- global image attention by patches
- learn to attend to patches further away at the lower layers which convnet cannot

[Big bird: Transformers for longer sequences](https://github.com/jinglescode/papers/issues/51)
- scaling up transformers to long sequences, by replacing full quadratic attention mechanism by a combination of random attention, window attention, and global attention
- allow the processing of longer sequences, translating to state-of-the-art experimental results
- theoretical guarantees of universal approximation and turing completeness

[Long Range Arena : A Benchmark for Efficient Transformers](https://github.com/jinglescode/papers/issues/53)
- a benchmark for transformer tasks
- compare performance and speed across xformer models

[Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking](https://github.com/jinglescode/papers/issues/54)
- attention at each multi-task to focus on the useful parts of the waveform for each task

[O(n) Connections are Expressive Enough: Universal Approximability of Sparse Transformers](https://github.com/jinglescode/papers/issues/55)
- this paper proofs that sparse transformers can approximate the same as the dense counterpart for any sequence to sequence function

[Are Transformers universal approximators of sequence-to-sequence functions?](https://github.com/jinglescode/papers/issues/56)
- multi-head self-attention layers can indeed compute contextual mappings of the input sequences
- Transformers can represent any sequence-to-sequence functions, Transformers are universal approximators of continuous and permutation equivariant sequence-to-sequence functions with compact support

[Fast Transformers with Clustered Attention](https://github.com/jinglescode/papers/issues/58)
- improve the computational complexity of self-attention. by cluster the queries into non-overlapping clusters

[Transformers with convolutional context for ASR](https://github.com/jinglescode/papers/issues/59)
- replacing the sinusoidal positional embedding for transformers with convolutionally learned input representations
- fixed learning rate of 1.0 and no warmup steps

[Exploring Transformers for Large-Scale Speech Recognition](https://github.com/jinglescode/papers/issues/61)
- perform ASL, with streaming approach base on the Transformer-XL network
- compare BLSTM to Transformer and Transformer-XL

[Transformers without Tears: Improving the Normalization of Self-Attention](https://github.com/jinglescode/papers/issues/62)
- ScaleNorm: normalization with a single scale parameter for faster training and better performance

[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://github.com/jinglescode/papers/issues/66)
- reduce space complexity: query sparsity measurement
- reduce time complexity: ProbSparse
- predict sequence in one batch: generative style decoder (decoder generates long sequences with 1 forward pass)

## Representation learning

[Deep Canonical Correlation Analysis](https://github.com/jinglescode/papers/issues/99)
- learn complex nonlinear transformations of two views of data such that the resulting representations are highly linearly correlated
- significantly higher correlation than those learned by CCA and KCCA
- introduce a novel non-saturating sigmoid function based on the cube root

# Medical

## Brain computer interface

[Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs](https://github.com/jinglescode/ml-papers/issues/6)
- covers a variety of CCAs
- estimate reliable spatial filters and SSVEP templates given small calibration data

[Deep Learning-based Classification for Brain-Computer Interfaces](https://github.com/jinglescode/ml-papers/issues/12)
- comparing CNN and LSTM for 5 classes SSVEP classification

[Learning representations from EEG with deep recurrent-convolutional neural networks](https://github.com/jinglescode/papers/issues/30)
- designed to preserve the spatial, spectral, and temporal structure of EEG which leads to finding features that are less sensitive to variations and distortions within each dimension
- robust to inter- and intra-subject differences, as well as to measurementrelated noise

[Retinotopic and topographic analyses with gaze restriction for steady-state visual evoked potentials](https://github.com/jinglescode/papers/issues/31)
- findings provide a basis for determining stimulus parameters for neural engineering studies
- proposed experimental paradigm could also provide a precise framework for future SSVEP-related studies

[Steady-state visually evoked potentials: Focus on essential paradigms and future perspectives](https://github.com/jinglescode/papers/issues/34)
- Provide details on SSVEP, essential to understand and build SSVEP based experiments

[Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface](https://github.com/jinglescode/papers/issues/48)
- FBCCA

[Methods of EEG Signal Features Extraction Using Linear Analysis in Frequency and Time-Frequency Domains](https://github.com/jinglescode/papers/issues/52)
- different ways to extract features from EEG

[MI-EEGNET: A novel Convolutional Neural Network for motor imagery classification](https://github.com/jinglescode/papers/issues/57)
- outperformed existing methods on BCI Competition IV dataset IIa

[A Radial Zoom Motion-Based Paradigm for Steady State Motion Visual Evoked Potentials](https://github.com/jinglescode/papers/issues/69))
- radial zoom motion-based SSMVEP paradigm achieves slightly lower accuracy than flicker, but its comfort score and fatigue score is much better than flicker stimulus.

[Selective attention to stimulus location modulates the steady-state visual evoked potential](https://github.com/jinglescode/papers/issues/67)
- a 1996 paper, presented user 2 stimuli, with attention, SSVEP extraction is pausable

[Four Novel Motion Paradigms Based on Steady-state Motion Visual Evoked Potential](https://github.com/jinglescode/papers/issues/13)
- four stimulus paradigms based on basic motion modes: swing, rotation, spiral, and radial contraction-expansion

[Highly Interactive Brain–Computer Interface Based on Flicker-Free Steady-State Motion Visual Evoked Potential](https://github.com/jinglescode/papers/issues/70)
- motion checkerboard stimulation method would keep uniform brightness at all local areas that delivered pure motion stimuli and that motion blur would be further reduced with a high-refresh-rate display to elicit SSMVEPs with a single frequency

[Comparison of Modern Highly Interactive Flicker-Free Steady State Motion Visual Evoked Potentials for Practical Brain–Computer Interfaces](https://github.com/jinglescode/papers/issues/71)
- tested 5 motion based stimuli, motion evoked potentials is more comfortable alternative to flickering visual stimulation
- comparable performance as flickering visual stimulation 

[A new dual-frequency stimulation method to increase the number of visual stimuli for multi-class SSVEP-based brain–computer interface](https://github.com/jinglescode/papers/issues/72)
- as SSVEP frequencies are generally limited by the monitor, this method allow us to increase the number of visual stimuli when necessary
- ITR 33.26 bits/min, accuracy of 87.23%

[Electrophysiological correlates of gist perception: a steady-state visually evoked potentials study](https://github.com/jinglescode/papers/issues/77)
- multi-stimulus paradigms is suitable to measure brain activity related specifically to each stimulus separately
- two neighboring stimuli were flickered at different frequencies, SSVEPs enabled us to separate the responses to the two distinct stimuli by extracting oscillatory brain responses
- succeeded in eliciting oscillatory brain responses at the driving stimuli’s frequencies, their harmonics, and the intermodulation frequency, that is, f1 + f2 = 20.57 Hz
- brain’s response at a linear combination of two frequencies
- demonstrates that SSVEPs are an excellent method to unravel mechanisms underlying the processing within multi-stimulus displays in the context of gist perception
- multiple stimulus displays in combination with the analyses of intermodulation frequencies makes this an ideal approach to investigate gist perception in multi-stimulus processing

[Perception of illusory contours forms intermodulation responses of steady state visual evoked potentials as a neural signature of spatial integration](https://github.com/jinglescode/papers/issues/78)
- spectral decomposition of the measured EEG can show additional peaks at frequencies that are linear combinations of the driving frequencies
- show that the perception of an illusory rectangle resulted in a significant increase of amplitudes in two intermodulation frequencies

[From intermodulation components to visual perception and cognition-a review](https://github.com/jinglescode/papers/issues/79)
- explore different uses of intermodulation, and review a range of recent studies exploiting intermodulation in visual perception research

[Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs](https://github.com/jinglescode/papers/issues/86)
- important / popular paper on CCA for SSVEP detection

[Computational modeling and application of steady-state visual evoked potentials in brain-computer interfaces](https://github.com/jinglescode/papers/issues/87)
- shows how SSVEP works
- notes on SSVEP

[Spatial Filtering in SSVEP-Based BCIs: Unified Framework and New Improvements](https://github.com/jinglescode/papers/issues/88)
- described popular CCAs techniques (itCCA, eCCA, Transfer Template CCA, Filter Bank CCA, Task-Related Component Analysis)
- a unified framework under which the spatial filtering algorithms can be formulated as generalized eigenvalue problems (GEPs) with four different elements: data, temporal filter, orthogonal projection and spatial filter
- design new spatial filtering algorithms

[Spatial Filtering Based on Canonical Correlation Analysis for Classification of Evoked or Event-Related Potentials in EEG Data](https://github.com/jinglescode/papers/issues/89)
- to increase classification accuracy, spatial filters are used to improve the signal-to-noise ratio of the brain signals and thereby facilitate the detection and classification of SSVEP and other VEPs

[SSVEP enhancement based on Canonical Correlation Analysis to improve BCI performances](https://github.com/jinglescode/papers/issues/90)
- this investigate CCA as a signal enhancement method and not as a feature extraction method
- make use of the ability of CCA to handle multichannel EEG and find the space in which EEG samples correlate the most with the stimuli
- CCA yields effective weights (spacial filters) with relatively small training sets

[Multiway Canonical Correlation Analysis of Brain Signals](https://github.com/jinglescode/papers/issues/91)
- CCA does not address the issue of comparing or merging responses across more than two subjects
- Multiway CCA can be applied effectively to multi-subject datasets of EEG, to denoise the data prior to further analyses, and to summarize the data and reveal traits common across the population of subjects
- MCCA-based denoising yields significantly better scores in an auditory stimulus-response classification task, and MCCA-based joint analysis of fMRI data reveals detailed subject-specific activation topographies

[Spatial smoothing of canonical correlation analysis for steady state visual evoked potential based brain computer interfaces](https://github.com/jinglescode/papers/issues/92)
- CCA spatial filter becomes spatially smooth to give robustness in short signal length condition

[Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs](https://github.com/jinglescode/papers/issues/94)
- to utilize the training data corresponding to not only the target stimulus but also the neighboring stimuli for learning and consequently better performance in learning

[An amplitude-modulated visual stimulation for reducing eye fatigue in SSVEP-based brain-computer interfaces](https://github.com/jinglescode/papers/issues/95)
- reduction of eye fatigue for SSVEP
- Four targets were used in combinations of three different modulating frequencies and two different carrier frequencies in the offline experiment, and two additional targets were added with one additional modulating and one carrier frequency in online experiments
- results: caused lower eye fatigue and less sensing of flickering than a low-frequency stimulus, in a manner similar to a high-frequency stimulus

[Visual evoked potential and psychophysical contrast thresholds in glaucoma](https://github.com/jinglescode/papers/issues/96)
- VEP and Psyc thresholds can be quite uncorrelated with glaucoma, the two types of thresholds contain independent information about glaucoma that could be usefully combined

[Contrast sensitivity and visual disability in chronic simple glaucoma](https://github.com/jinglescode/papers/issues/97)
- battery of vision tests was used to quantify visual defect
- static contrast sensitivity function appears to be the most sensitive method of measuring visual defect in glaucoma patients
- vertical sinewave gratings

[Insights for mfVEPs from perimetry using large spatial frequency-doubling and near frequency-doubling stimuli in glaucoma](https://github.com/jinglescode/papers/issues/98)
- lower and higher temporal frequency tests probed the same neural mechanism
- no advantage of spatial frequency-doubling stimuli for mfVEPs

[Multifocal frequency-doubling pattern visual evoked responses to dichoptic stimulation](https://github.com/jinglescode/papers/issues/100)
- results indicated that dichoptic evoked potentials using multifocal frequency-doubling illusion stimuli are practical. The use of crossed orientation, or differing spatial frequencies, in the two eyes reduced binocular interactions.


## Vision

[A comparison of covert and overt attention as a control option in a steady-state visual evoked potential-based brain computer interface](https://github.com/jinglescode/ml-papers/issues/2)
- The average accuracy is found to be reduced by ~20% in the switch from overt to covert attention
- SSVEPs resulting from stimuli located in foveal vision are known to be of large amplitude and very robust
- stimuli located in peripheral vision generate SSVEPs of much smaller amplitude

[Neural Differences between Covert and Overt Attention Studied using EEG with Simultaneous Remote Eye Tracking](https://github.com/jinglescode/ml-papers/issues/3)
- EEG analysis of the period preceding the saccade latency showed similar occipital response amplitudes for overt and covert shifts, although response latencies differed.
- combined EEG and eye tracking can be successfully used to study natural overt shifts of attention
- There were no striking differences in early response components between overt and covert shifts in fronto-central areas
- Most studies of covert vs. overt attention involve instructing the participant to attend to a particular region of the field via a centrally presented cue, and so can be considered as an endogenous direction of attention. In contrast, our experiment provided an exogenous trigger for attention, by the appearance of a target in a peripheral field location. Thus it is possible that a different pattern of activation would be seen in the covert direction of attention by an endogenous cue

[Visual field testing for glaucoma – a practical guide](https://github.com/jinglescode/ml-papers/issues/4)
- This gives a good idea of what glaucoma patients see, useful when considering development test cases.

[Walking enhances peripheral visual processing in humans](https://github.com/jinglescode/ml-papers/issues/5)
- walking leads to an increased processing of peripheral input
- increased contrast sensitivity for peripheral compared to central stimuli when subjects were walking

[The steady-state visual evoked potential in vision research: A review](https://github.com/jinglescode/ml-papers/issues/7)
- provided details about SSVEP
- applications of SSVEP

[Multifocal Visual Evoked Potential (mfVEP) and Pattern-Reversal Visual Evoked Potential Changes in Patients with Visual Pathway Disorders: A Case Series](https://github.com/jinglescode/ml-papers/issues/8)
- mfVEP may provide a more accurate assessment of visual defects when compared with PVEP
- demonstrates that mfVEP, as an objective test for visual fields, is potentially more sensitive than PVEP in detecting focal visual pathway pathology

[Study for Analysis of the Multifocal Visual Evoked Potential](https://github.com/jinglescode/ml-papers/issues/9)
- To introduce the clinical utility of the absolute value of the reconstructed waveform method in the analysis of multifocal visual evoked potential (mfVEP).

[Multifocal visual evoked potentials for quantifying optic nerve dysfunction in patients with optic disc drusen](https://github.com/jinglescode/ml-papers/issues/10)
- To explore the applicability of multifocal visual evoked potentials (mfVEPs) for research and clinical diagnosis in patients with optic disc drusen (ODD). This is the first assessment of mfVEP amplitude in patients with ODD.

[Steady-state multifocal visual evoked potential (ssmfVEP) using dartboard stimulation as a possible tool for objective visual field assessment](https://github.com/jinglescode/ml-papers/issues/11)
- To investigate whether a conventional, monitor-based multifocal visual evoked potential (mfVEP) system can be used to record steady-state mfVEP (ssmfVEP) in healthy subjects and to study the effects of temporal frequency, electrode configuration and alpha waves.

[A Review of Deep Learning for Screening, Diagnosis, and Detection of Glaucoma Progression](https://github.com/jinglescode/papers/issues/35)
- review on deep learning and ophthalmology

[Objective visual field determination in forensic ophthalmology with an optimized 4-channel multifocal VEP perimetry system: a case report of a patient with retinitis pigmentosa](https://github.com/jinglescode/papers/issues/65)
- objective technique evaluating cortical activity, mfVEP was able to proof the concentric reduction of the visual field in this patient with late-stage retinitis pigmentosa

[An oblique effect in parafoveal motion perception](https://github.com/jinglescode/papers/issues/76)
- discriminate angular direction of moving grating in different frequency and in parafoveea

[Choice of Grating Orientation for Evaluation of Peripheral Vision](https://github.com/jinglescode/papers/issues/80)
- evaluate peripheral resolution and detection for different orientations in different visual field meridians

[Motion Perception in the Peripheral Visual Field](https://github.com/jinglescode/papers/issues/81)
- peripheral perception for motion stimulus

[Development of Grating Acuity and Contrast Sensitivity in the Central and Peripheral Visual Field of the Human Infant](https://github.com/jinglescode/papers/issues/82)
- analyze EEG of stimulus at peripheral (8 to 16 degree), to determine the separate responses for central and peripheral fields
- no infant in this study had higher acuity in the peripheral field than in the central field
- well, the peripheral field is relatively more mature at birth

[Motion perception in the peripheral visual field](https://github.com/jinglescode/papers/issues/83)
- performance in the temporal hemified was slightly superior to that in the nasal hemifield and depended on the orientation as well as on the direction of the motion. 
- perception of horizontal motion was better than that of vertical motion. 
- in spite of large variations, centrifugal motion was significantly more readily perceived than centripetal motion.

[Speed of visual processing increases with eccentricity](https://github.com/jinglescode/papers/issues/84)
- the fovea has the resolution required to process fine spatial information, but the periphery is more sensitive to temporal properties
- speed of information processing varies with eccentricity: processing was faster when same-size stimuli appeared at 9° than 4° eccentricity
- at the same eccentricity, larger stimuli are processed more slowly

[Stimulus dependencies of an illusory motion: Investigations of the Motion Bridging Effect](https://github.com/jinglescode/papers/issues/85)
- using ring of points, check retinal eccentricity with various configurations

[Ehud Kaplan on Receptive fields](https://github.com/jinglescode/papers/issues/93)
- about M-cells and the P-cells
