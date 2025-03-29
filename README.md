# Mental Health Web App README



## WEBSITE LINK
[https://my-mental-health-app-af8769922eba.herokuapp.com/]


## DEMO VIDEO LINK
[https://drive.google.com/file/d/1S-k24QQeR0rFMi1dGqxEUcs8uMUVrmMT/view?usp=drive_link]

## Project Overview
The Mental Health Web App is designed to assess and support users' mental health using machine learning algorithms and intelligent chatbot interactions. The application includes features such as stress level assessment, real-time conversation capabilities, and a selection of meditation music to enhance user experience.

## Features
* Stress, Depression, Anxiety Level Assessment: Uses Support Vector Machine (SVM) algorithms to evaluate users' stress, anxiety, depression levels based on various input factors.
* Intelligent Chatbot: Provides real-time interaction with users to engage them in conversations related to their mental health.
* Meditation Music: Offers a curated selection of meditation music to help users relax and improve their mental well-being.

## Technologies Used
* Programming Languages: Python
* Machine Learning: SVM algorithms, Large Language Models (LLMs)
* Web Development: Flask (or any other web framework used)
* Additional Libraries: scikit-learn, pandas, numpy,openapi
* Frontend: HTML, CSS, JavaScript
* Backend: Flask


## How to Run  

### 1. Clone the Repository  
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Set up Virtual environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a .env file and add necessary API keys or configurations:

```sh
OPENAI_API_KEY=your-api-key
```


### 5.Run the application 

```sh
python app_main.py
```