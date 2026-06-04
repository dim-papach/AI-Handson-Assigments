# HECATE Astrophysics Conversational AI Agent

## Tasks
A quick overview of the tasks.

### 1. Knowledge Base & RAG System

The project aims to undestand how galaxies are classified and what are their properties. For this reason we used the HECATE catalog, which contains information about 100,000 galaxies. For our model to be able to answer questions about the HECATE catalog, we used a RAG system to retrieve relevant information from the catalog. The knowledge base we provided to the model contains articles, papers and books about galaxies, all of them in pdf format.

The RAG system is implemented using LangChain and ChromaDB. We split the documents into chunks of 1000 characters with an overlap of 200 characters. Then we used the HuggingFaceEmbeddings model to embed the chunks and store them in the ChromaDB as a persistent vector store. The vector store is stored in the data/vector_store directory and the embeddings are created using the all-MiniLM-L6-v2 model. The RAG system is able to answer questions about the galaxies and the data of the HECATE catalog by retrieving the most relevant chunks for a given query and concatenating them into a single string to be passed to the LLM.

#### RAG retrieval example

```zsh
# We use a custom test script to test the RAG system without the LLM agent
AI-Handson-Assigments on  dev [⇡!?] 
⚡poetry run python hw2/src/test_rag.py

Testing RAG System (Retrieval Only)...

Query: 'How is logSFR_HEC calculated?'

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 9419.85it/s]
=== Retrieved Context ===
logSFR 22u – Decimal logarithm of the W4-based SFR estimate (M ⊙ yr−1).
logSFR HEC – Homogenized log SFR (M ⊙ yr−1). Rescaling of SFR indicators is performed only here (Section 4.2.2).
SFR HEC flag – Flag indicating photometry source and SFR indicator used for logSFR HEC (Table 2).
logM HEC – Decimal logarithm of the M⋆ (M⊙).
logSFR GSW G Decimal logarithm of the SFR in GSWLC-2 (M⊙ yr−1).
logM GSW G Decimal logarithm of the M⋆ in GSWLC-2 (M⊙).
min snr – Minimum signal-to-noise ratio of the emission lines used for the activity classiﬁcation ( class sp).
metal, flag metal – Metallicity [12 + log (O/H)] and its quality ﬂag (Section 4.2.3).
class sp – Nuclear activity classiﬁcation (Section 4.2.4): 0 =star forming, 1=Seyfert, 2=LINER, 3=composite, −1=unknown.
agn s17 E AGN classiﬁcation in She et al. ( 2017): Y=AGN, N=non-AGN, ?=unknown.
agn hec – Combination of SDSS and She et al. (2017) classiﬁcations (Section 4.2.4): Y=AGN, N=non-AGN, ?=unknown.

---

logL TIR – Decimal logarithm of the TIR luminosity ( L⊙=3.83 × 1033 erg s−1).
logL FIR – Decimal logarithm of the FIR luminosity (L ⊙).
logL 60u – Decimal logarithm of the 60 μm-band luminosity (L⊙).
logL 12u – Decimal logarithm of the 12 μm-band luminosity (L⊙).
logL 22u – Decimal logarithm of the 22 μm-band luminosity (L⊙).
logL K – Decimal logarithm of the Ks-band luminosity (L⊙).
ML ratio – Mass-to-light ratio (Section 4.2.1).
logSFR TIR – Decimal logarithm of the TIR-based SFR estimate (M ⊙ yr−1).
logSFR FIR – Decimal logarithm of the FIR-based SFR estimate (M ⊙ yr−1).
logSFR 60u – Decimal logarithm of the 60 μm-based SFR estimate (M⊙ yr−1).
logSFR 12u – Decimal logarithm of the W3-based SFR estimate (M ⊙ yr−1).
logSFR 22u – Decimal logarithm of the W4-based SFR estimate (M ⊙ yr−1).
logSFR HEC – Homogenized log SFR (M ⊙ yr−1). Rescaling of SFR indicators is performed only here (Section 4.2.2).

---

1532 P. Popesso et al. 
MNRAS 519, 1526–1544 (2023) 
Figure 1. Left-hand panel: logSFR versus the Universe age in several bins of stellar masses. The data points indicate the SFR based on the MS estimates 
collected in this work. The solid lines show the best linear ﬁt as in equation ( 9 ). Data points and lines are colour-coded as a function of the stellar mass bin as 
indicated in the ﬁgure. For clarity, the relations are artiﬁcially displaced by 0.4 dex from one another. Right-hand panel: α( logM ⋆ ) (upper panel) and β( logM ⋆ ) 
(lower panel) as a function of logM ⋆ . The red solid lines in both panels indicate the best-ﬁtting relations of equation ( 9 ). The dashed line in the bottom panel 
shows the liner ﬁt approximation as proposed in S14 . 
Table 2. The table lists the best-ﬁtting parameters of equations ( 10 ) and ( 14 ) 
in the ﬁrst two columns. The last column lists the best-ﬁtting parameters of

=========================
```

### 2. Model as a Tool

On HW1 the best model we obtanied was XGBoost with a ROC-AUC of 0.9930. Also it was the best model in terms of correctly classifying the different classes. We saved it using pickle and used it as a tool in our RAG system to predict the class of a galaxy. We also saved the preprocessor used in hw1 to preprocess the data before passing it to the model. The saved pickle files are copied in the models directory. 

For the prediction tool, we apply the same preprocessing steps as in HW1 to the input data:
- `apply_iqr_capping`: Capping outliers using the 1.5 IQR rule.
- `apply_imputation`: Filling missing values using IterativeImputer.
- `filter_error_ratios`: Dropping rows with bad SNR.
- `compute_colors`: Engineering features like U-R, R-I.
- `apply_scaling`: Scaling the data using StandardScaler.
After preprocessing, the data is passed to the XGBoost model which predicts the class of the galaxy.

#### Example Prediction

```zsh
# We use a custom test script to test the prediction tool without the LLM agent
AI-Handson-Assigments on  dev [⇡!?] 
⚡poetry run python hw2/src/test_prediction.py

Testing Prediction Tool (HW1 Model Wrapper)...

Input features provided to tool:
{'T': 6.0, 'WF1': 12.5, 'UT': 14.2, 'U': 15.1, 'R': 14.3, 'G': 14.8, 'I': 14.0, 'Z': 13.9, 'logM_HEC': 10.5, 'logSFR_HEC': 0.5, 'METAL': 0.02, 'AGN_HEC': 'Y'}

IQR Capping Applied: 1 rows modified
Error Ratios Filtering Removed: 0 rows
=== Tool Output ===
Prediction: Seyfert (probability: 98.2%)
===================
```

### 3. Conversational AI Agent

The agent is composed of 2 main components:

- The Retrieval Tool, which is used to retrieve relevant information from the knowledge base from Task 1.
- The Prediction Tool, which is used to predict the class of a galaxy from HW1's best model.

We have also added a conversation memory to the agent, so that it can remember the previous turns in the conversation.

### Example Conversations

```zsh
⚡poetry run python hw2/src/agent.py

Testing Agent with groq (type 'quit' to exit)...

You: What are the main indicators that a galaxy has the "Seyfert" classification?
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████| 103/103 [00:00<00:00, 12138.27it/s]

Agent: The main indicators that a galaxy has the "Seyfert" classification include:

* Optical properties, such as emission lines and continuum emission
* IR indicators, such as IRAS and WISE photometry
* Spectroscopic data, which can be used to estimate the metallicity of the galaxy and characterize its nuclear activity
* Clustering properties, which are similar to those of QSOs but with lower luminosities
* Membership in groups and clusters, with a high probability of being in these environments
* Central galaxy fraction in the MS region, which can be identified using IR-selected galaxy catalogs and SFR estimates.

It's worth noting that the classification of a galaxy as a Seyfert is based on a combination of these indicators, and different studies may use different criteria to identify Seyfert galaxies.

You: what other classes do we have on the HECATE catolog beside this one?
Loading weights: 100%|██████████████████████████████████████████████████████| 103/103 [00:00<00:00, 10689.43it/s]

Agent: The HECATE catalog includes several classes of galaxies, including:

* Seyfert galaxies
* Star-forming galaxies
* LINER (Low-Ionization Nuclear Emission-Line Region) galaxies
* Composite galaxies (which exhibit both star-forming and AGN activity)
* Non-AGN galaxies

These classes are based on various indicators, such as optical and IR properties, spectroscopic data, and clustering properties. The HECATE catalog also provides additional information, such as distances, SFRs, and stellar masses, for each galaxy.

You: what was the first class I asked you about?

Agent: The first class you asked me about was the "Seyfert" galaxy classification. You wanted to know the main indicators that a galaxy has this classification.

```


### 4. FastAPI Endpoint

To start the FastAPI server, run the following command:

```zsh
poetry run uvicorn hw2.src.api:app --reload
```

### 5. Additional Tools

Since the dataset we are working with is a quite large (~10,000 galaxies), we need to provide the user with ways to easily find the data they are looking for. This is why we added the CSV lookup tool.

We also added the Dataset Statistics tool, which returns the statistics of the dataset.


Additionaly, since the dataset is numerical and each column has specific units of measurement, we added the Calculator tool, which can be used to perform calculations on the dataset.

#### Example Interaction

```zsh
⚡poetry run python hw2/src/agent.py

Testing Agent with gemini (type 'quit' to exit)...

You: Find the closest Star Forming galaxy in our dataset (D column)

Agent: The closest Star Forming galaxy in our dataset is PGC 44491 (UGC08091) with a distance of 2.0998.

You: give me the the sfr/mass ratio of this galaxy in log scale and in linear scale

Agent: It looks like the `logSFR_HEC` and `logM_HEC` values are not available for PGC 44491 in the dataset. Therefore, I cannot calculate the SFR/mass ratio for this galaxy.

You: give me the closest SF galaxy with both sfr and mass and calculate the sfr/mass in log and linear

Agent: The closest Star Forming galaxy with both SFR and stellar mass data is NGC 5238 (PGC 47853), at a distance of 4.4981.

For this galaxy:
*   log(SFR/Mass) = -9.0735
*   SFR/Mass (linear scale) = 8.443 x 10^-10

You: what is the average SFR of the dataset?

Agent: The average log(SFR) of the dataset is -0.0680.
```

## 6. Streaming Output

We have also implemented streaming output for the agent. This means that the agent will output the response in chunks as it is generated, rather than waiting for the entire response to be generated before outputting it. This is useful for long responses, as it allows the user to see the response as it is being generated.


