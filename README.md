# Lat_Arm-transliteration 🇦🇲

**Lat_Arm-transliteration** is an AI-based tool for converting Latin-scripted Armenian text into Armenian letters.  
The project is developed as a **diploma project** and focuses on building a fast, accurate, and lightweight transliteration system for Armenian language processing.

---

## 📌 Project Overview

Many Armenian users write Armenian text using Latin characters (e.g. *barev inchpes es*).  
This project aims to automatically transliterate such text into proper Armenian script:

**Input (Latin Armenian):**

**Output (Armenian):**


The system is designed to handle:
- Common Latin-to-Armenian spelling variations  
- Ambiguous character mappings  
- Real-world informal text  

---

## 🎯 Objectives

- Build an efficient transliteration model for Latin-scripted Armenian  
- Achieve high accuracy without excessive model complexity  
- Create a foundation for future Armenian NLP tools  

---

## 🧠 Methodology

The transliteration system is based on:

- **Seq2Seq architecture**
- **LSTM-based encoder–decoder**
- Character-level modeling
- Trained on paired Latin–Armenian text data

This approach allows the model to learn contextual character mappings instead of relying on rigid rule-based conversion.

---

## 🛠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Jupyter Notebook** (for experiments and training)

---


---

## 🚀 How It Works

1. Input text is preprocessed and tokenized at character level  
2. The encoder processes the Latin input sequence  
3. The decoder generates Armenian characters sequentially  
4. The final output is reconstructed into Armenian text  

---

## 📊 Results

- High transliteration accuracy on common Armenian phrases  
- Robust handling of informal Latin spellings  
- Efficient inference suitable for web-based usage  

*(Exact metrics can be added after final evaluation.)*

---

## 🔮 Future Improvements

- Expand and diversify the training dataset  
- Improve handling of rare or ambiguous transliterations  
- Add bad-word detection and filtering  
- Integrate the model into a web or mobile application  
- Support dialectal or phonetic variations  

---

## 🎓 Academic Context

This project is developed as part of a **Bachelor Diploma** in the field of **Information Systems**, with a focus on Armenian language technologies.

---

## 👤 Author

**Lusine Atshemyan**  
National Polytechnic University of Armenia  
Faculty of Information Systems  

---

## 📄 License

This project is intended for academic and research purposes.


