# 🩺 MEDI-BOT

**Medibot** is a Flask-based health assistant that predicts diseases from user-input symptoms and provides detailed health guidance such as precautions, medications, diets, and workouts. It also simplifies complex medical terms using NLP and GPT-powered responses.

---

## 🎯 Project Objective

The main aim of this project is to **leverage the power of Artificial Intelligence in the field of medicine**—making healthcare more accessible, understandable, and user-friendly.

---

## 🔄 Project Flow

### 1. `new.html` – The Main Interface

This is the **core user interface** of the project. It features a navigation bar that lets users move between pages. The main functionality allows users to:

- Enter symptoms in a text field.
- Receive a **predicted disease** based on those symptoms.
- View associated **precautions**, **medications**, **diet plans**, and **workout suggestions**.

---

### 2. Contact Page (`contact_copy.html`)

This page provides the **contact information** of the developer.

---

### 3. About Page (`about.html`)

The About page gives users **detailed information** about the project, its purpose, and how it can be beneficial in real-world healthcare scenarios.

---

### 4. Chat Page (`chat.html`)

This is the **second major feature** of the application. It allows users to:

- Interact with a chatbot whose main responsibility is to **simplify complex medical jargon**.
- Understand medical prescriptions or reports **without always needing a doctor**.
- Ask questions in plain language and get responses in an easy-to-understand format.

---

### 5. Test QA Page (`test_qa.html`)

This feature is currently **under testing**. It aims to integrate various AI models, including **OpenAI's GPT-3.5-turbo** (via the OLlama interface). To use this:

- Go to the [OpenAI API Key page](https://platform.openai.com/account/api-keys).
- Create an API key.
- Paste it into the `mainapp.py` file where indicated.

Once this setup is complete, the **AI-powered medical assistant** will be good to go.

---

## 📓 Jupyter Notebook (`.ipynb`) File

This file provides a **command-line interface (CLI)** version of the disease prediction tool:

- It's divided into **code chunks** so that users can manually run each part.
- Useful for testing and understanding the backend logic **without a GUI**.
- Acts as a **summary notebook** for the core disease prediction pipeline.

---

## ✅ Summary

Medibot is a comprehensive healthcare assistant that combines:
- Machine Learning for disease prediction,
- NLP for simplifying medical terms,
- OpenAI's GPT for intelligent medical Q&A,
- And a user-friendly web interface for seamless interaction.

This project demonstrates how **AI can assist in patient education and health decision-making**.

---

