from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
import pandas as pd 
import pickle
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = Flask(__name__)
app.secret_key = 'vanhdz'
@app.route("/")
def index():
    return render_template('index.html')



chat_messages = []
assistant_messages = [{'role': 'system', 'content': 'You are a friend who listen actively. You may ask a couple of questions until you fully understand your friend story, then you do an indepth analysis of the story and give advice. Do not give general advice but rather focus on the situation.'}]    

@app.route("/chatbot", methods=["GET", "POST"])

def chatbot():
    global chat_messages
    if request.method == "GET":
        chat_messages = []
        return render_template('chatbot.html')
    elif request.method == "POST":
        #print(chat_messages, assistant_messages)
        msg = request.form["msg"]
        input_msg = msg
        user_role = {'role': 'user', 'content': input_msg}

        # chat history 
        chat_messages.append(assistant_messages[-1])
        chat_messages.append(user_role)

        # append to assistant_messages
        response = get_chat_response(chat_messages)
        assistant_messages.append({'role': 'system', 'content': response})
        print(chat_messages)
        return response


def get_chat_response(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )   
    return response.choices[0].message.content

@app.route("/sentiment", methods=["POST"])
def sentiment():
# Define the user inputs
    user_inputs = [chat_messages[i]['content'] for i in range(len(chat_messages)) if i%2 == 1]

    # Perform sentiment analysis on each user input
    sentiments = []
    for input_text in user_inputs:
        blob = TextBlob(input_text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            sentiments.append('Positive')
        elif sentiment < 0:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    sentiment_counts = {
    'Positive': sentiments.count('Positive'),
    'Negative': sentiments.count('Negative'),
    'Neutral': sentiments.count('Neutral')
    }

    colors = {
        'Positive': 'lightgreen',  # Green
        'Negative': 'red',    # Red
        'Neutral': 'lightgray'     # Gray
    }

    trace5 = go.Bar(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()),marker=dict(color=[colors[sentiment] for sentiment in sentiment_counts.keys()])
)
    layout5 = go.Layout(title='Sentiment Analysis Results', xaxis=dict(title='Sentiment'), yaxis=dict(title='Message Count'))
    fig5 = go.Figure(data=[trace5], layout=layout5)
    graphJSON5 = fig5.to_json()
    return jsonify(graphJSON5)

@app.route("/wordcloud", methods=["GET"])
def generate_wordcloud():
    global chat_messages
    # Extract text from chat messages
    all_messages =  [chat_messages[i]['content'] for i in range(len(chat_messages)) if i%2 == 1]


    # Join all messages into a single string
    text = ' '.join(all_messages)

    # Generate word frequencies
    word_freq = Counter(text.split())

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Plot the word cloud
    # Save word cloud to a file
    wordcloud_path = "wordcloud.png"
    wordcloud.to_file('./static/' + wordcloud_path)

    return wordcloud_path

# Add a route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)





@app.route("/survey_main", methods = ["GET"])
def survey_main():
    return render_template('survey_main.html')

@app.route("/aboutus", methods = ["GET"])
def aboutus():
    return render_template('about_us.html')


# Use pickle to load in the pre-trained model.
with open('static/model/model1.pkl', 'rb') as f:
    model_depression = pickle.load(f)

@app.route('/Depression_survey', methods=['GET', 'POST'])
def Depression_survey():

    if request.method == 'POST':  #when we will click the submit button
        q1 = request.form['q1']
        q2 = request.form['q2']
        q3 = request.form['q3']
        q4 = request.form['q4']
        q5 = request.form['q5']
        q6 = request.form['q6']
        q7 = request.form['q7']
        q8 = request.form['q8']
        q9 = request.form['q9']
        q10 = request.form['q10']
        q11 = request.form['q11']
        q12 = request.form['q12']
        q13 = request.form['q13']
        q14 = request.form['q14']
        q15 = request.form['q15']
        q16 = request.form['q16']
        q17 = request.form['q17']
        q18 = request.form['q18']
        q19 = request.form['q19']
        q20 = request.form['q20']
        q21 = request.form['q21']
        q22 = request.form['q22']
        q23 = request.form['q23']
        q24 = request.form['q24']
        edu = request.form['edu']
        q25 = request.form['urban']
        q26 = request.form['gender']
        q27 = request.form['eng']
        q28 = request.form['age']
        reli = request.form['religion']
        q29 = request.form['hand']
        q30 = request.form['orientation']
        q31 = request.form['race']
        q32 = request.form['voted']
        q33 = request.form['married']
        q34 = request.form['familysize']
        
        def condition(x):
            if x<=10:
                return 'Under 10'
            if  10<=x<=16:
                return ' Primary Children'
            if 17<=x<=21:
                return 'Secondary Children'
            if 21<=x<=35:
                return 'Adults'
            if 36<=x<=48:
                return 'Elder Adults'
            if x>=49:
                return 'Older People'

        q28 = condition(int(q28))
        def change_var(x):
            if x=='Primary Children':
                return 0
            elif x=='Secondary Children':
                return 1
            elif x=='Adults':
                return 2
            elif x=='Elder Adults':
                return 3
            elif x=='Older People':
                return 4
        q28 = change_var(q28)

        # max-min scaling 
        df = pd.read_csv('static/model/max_min_values.csv', index_col=0)


        # Make DataFrame for model
        input_variables = pd.DataFrame([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13, q14, q15, q16,q17,q18,q19,q20,q21,q22,q23,q24,edu,q25,q26,q27,q28,reli, q29,q30,q31,q32,q33,q34]],
                                      columns=['Q3A', 'Q5A', 'Q10A', 'Q13A', 'Q16A', 'Q17A', 'Q21A', 'Q24A', 'Q26A',
       'Q31A', 'Q34A', 'Q37A', 'Q38A', 'Q42A',
       'Extraverted-enthusiastic', 'Critical-quarrelsome',
       'Dependable-self_disciplined', 'Anxious-easily upset',
        'Open to new experiences-complex', 'Reserved-quiet', 'Sympathetic-warm',
        'Disorganized-careless', 'Calm-emotionally_stable',
        'Conventional-uncreative', 'education', 'urban', 'gender', 'engnat',
         'hand', 'religion',
        'orientation', 'race', 'voted', 'married', 'familysize',
        'Age_Groups'],
                                      dtype=float,
                                      index=['input'])  

        # min-max scaling 
        
        input_variables = (input_variables - df['min'])/(df['max'] - df['min']) 


        prediction = model_depression.predict(input_variables)[0]
        message = 'I have just taken my mental survey, and the prediction is:' + prediction
        chat_messages = [{'role': 'system', 'content': 'The user has just taken a mental survey test, give them an advice, maybe use some quote at the beginning and speak less than 75 words, use some icon'}, {'role': 'user', 'content': message}]
        response = get_chat_response(chat_messages)
        return jsonify({'prediction': prediction, 'response': response})
    #if flask.request.method == 'GET':
    return render_template('Depression_survey.html', prediction = None) #homepage is loaded




@app.route("/music", methods = ["GET"])
def music():
    return render_template('music.html')

with open('static/model/Stress_model.pkl', 'rb') as f:
    model_stress = pickle.load(f)

@app.route('/Stress_survey', methods=['GET', 'POST'])
def Stress_survey():

    if request.method == 'POST':  #when we will click the submit button
        q1 = request.form['q1']
        q2 = request.form['q2']
        q3 = request.form['q3']
        q4 = request.form['q4']
        q5 = request.form['q5']
        q6 = request.form['q6']
        q7 = request.form['q7']
        q8 = request.form['q8']
        q9 = request.form['q9']
        q10 = request.form['q10']
        q11 = request.form['q11']
        q12 = request.form['q12']
        q13 = request.form['q13']
        q14 = request.form['q14']
        q15 = request.form['q15']
        q16 = request.form['q16']
        q17 = request.form['q17']
        q18 = request.form['q18']
        q19 = request.form['q19']
        q20 = request.form['q20']
        q21 = request.form['q21']
        q22 = request.form['q22']
        q23 = request.form['q23']
        q24 = request.form['q24']
        edu = request.form['edu']
        q25 = request.form['urban']
        q26 = request.form['gender']
        q27 = request.form['eng']
        q28 = request.form['age']
        reli = request.form['religion']
        q29 = request.form['hand']
        q30 = request.form['orientation']
        q31 = request.form['race']
        q32 = request.form['voted']
        q33 = request.form['married']
        q34 = request.form['familysize']
        
        def condition(x):
            if x<=10:
                return 'Under 10'
            if  10<=x<=16:
                return ' Primary Children'
            if 17<=x<=21:
                return 'Secondary Children'
            if 21<=x<=35:
                return 'Adults'
            if 36<=x<=48:
                return 'Elder Adults'
            if x>=49:
                return 'Older People'

        q28 = condition(int(q28))
        def change_var(x):
            if x=='Primary Children':
                return 0
            elif x=='Secondary Children':
                return 1
            elif x=='Adults':
                return 2
            elif x=='Elder Adults':
                return 3
            elif x=='Older People':
                return 4
        q28 = change_var(q28)

        # max-min scaling 
        df = pd.read_csv('static/model/max_min_values.csv', index_col=0)
        new_names = ['Q1A', 'Q6A', 'Q8A', 'Q11A', 'Q12A', 'Q14A', 'Q18A', 'Q22A', 'Q27A',
       'Q29A', 'Q32A', 'Q33A', 'Q35A', 'Q39A']
        old_names = [
            'Q3A',
            'Q5A',
            'Q10A',
            'Q13A',
            'Q16A',
            'Q17A',
            'Q21A',
            'Q24A',
            'Q26A',
            'Q31A',
            'Q34A',
            'Q37A',
            'Q38A',
            'Q42A']
        stress_d = dict(zip(old_names, new_names))
        df.rename(index = stress_d, inplace= True)

        # Make DataFrame for model
        input_variables = pd.DataFrame([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13, q14, q15, q16,q17,q18,q19,q20,q21,q22,q23,q24,edu,q25,q26,q27,q28,reli, q29,q30,q31,q32,q33,q34]],
                                      columns=['Q1A', 'Q6A', 'Q8A', 'Q11A', 'Q12A', 'Q14A', 'Q18A', 'Q22A', 'Q27A',
       'Q29A', 'Q32A', 'Q33A', 'Q35A', 'Q39A',
       'Extraverted-enthusiastic', 'Critical-quarrelsome',
       'Dependable-self_disciplined', 'Anxious-easily upset',
        'Open to new experiences-complex', 'Reserved-quiet', 'Sympathetic-warm',
        'Disorganized-careless', 'Calm-emotionally_stable',
        'Conventional-uncreative', 'education', 'urban', 'gender', 'engnat',
        'hand', 'religion',
        'orientation', 'race', 'voted', 'married', 'familysize',
        'Age_Groups'],

                                      dtype=float,
                                      index=['input'])  

        # min-max scaling   
        
        input_variables = (input_variables - df['min'])/(df['max'] - df['min']) 
        print(input_variables.isnull().sum())
        
        prediction = model_stress.predict(input_variables)[0]
        message = 'I have just taken my mental survey, and the prediction is:' + prediction
        chat_messages = [{'role': 'system', 'content': 'The user has just taken a mental survey test, give them an advice, maybe use some quote at the beginning and speak less than 75 words, use some icon'}, {'role': 'user', 'content': message}]
        response = get_chat_response(chat_messages)
        return jsonify({'prediction': prediction, 'response': response})
    #if flask.request.method == 'GET':
    return render_template('Stress_survey.html', prediction = None) #homepage is loaded





####################################################################################################
with open('static/model/Anxiety_model.pkl', 'rb') as f:
    model_anxiety = pickle.load(f)

@app.route('/Anxiety_survey', methods=['GET', 'POST'])
def Anxiety_survey():
    if request.method == 'POST':  #when we will click the submit button
        q1 = request.form['q1']
        q2 = request.form['q2']
        q3 = request.form['q3']
        q4 = request.form['q4']
        q5 = request.form['q5']
        q6 = request.form['q6']
        q7 = request.form['q7']
        q8 = request.form['q8']
        q9 = request.form['q9']
        q10 = request.form['q10']
        q11 = request.form['q11']
        q12 = request.form['q12']
        q13 = request.form['q13']
        q14 = request.form['q14']
        q15 = request.form['q15']
        q16 = request.form['q16']
        q17 = request.form['q17']
        q18 = request.form['q18']
        q19 = request.form['q19']
        q20 = request.form['q20']
        q21 = request.form['q21']
        q22 = request.form['q22']
        q23 = request.form['q23']
        q24 = request.form['q24']
        edu = request.form['edu']
        q25 = request.form['urban']
        q26 = request.form['gender']
        q27 = request.form['eng']
        q28 = request.form['age']
        reli = request.form['religion']
        q29 = request.form['hand']
        q30 = request.form['orientation']
        q31 = request.form['race']
        q32 = request.form['voted']
        q33 = request.form['married']
        q34 = request.form['familysize']
        
        def condition(x):
            if x<=10:
                return 'Under 10'
            if  10<=x<=16:
                return ' Primary Children'
            if 17<=x<=21:
                return 'Secondary Children'
            if 21<=x<=35:
                return 'Adults'
            if 36<=x<=48:
                return 'Elder Adults'
            if x>=49:
                return 'Older People'

        q28 = condition(int(q28))
        def change_var(x):
            if x=='Primary Children':
                return 0
            elif x=='Secondary Children':
                return 1
            elif x=='Adults':
                return 2
            elif x=='Elder Adults':
                return 3
            elif x=='Older People':
                return 4
        q28 = change_var(q28)

        # max-min scaling 
        df = pd.read_csv('static/model/max_min_values.csv', index_col= 0)
        new_names = ['Q2A', 'Q4A', 'Q7A', 'Q9A', 'Q15A', 'Q19A', 'Q20A', 'Q23A', 'Q25A',
       'Q28A', 'Q30A', 'Q36A', 'Q40A', 'Q41A']
        old_names = [
            'Q3A',
            'Q5A',
            'Q10A',
            'Q13A',
            'Q16A',
            'Q17A',
            'Q21A',
            'Q24A',
            'Q26A',
            'Q31A',
            'Q34A',
            'Q37A',
            'Q38A',
            'Q42A']
        anx_d = dict(zip(old_names, new_names))
        df.rename(index = anx_d, inplace= True)
        
        # Make DataFrame for model
        input_variables = pd.DataFrame([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13, q14, q15, q16,q17,q18,q19,q20,q21,q22,q23,q24,edu,q25,q26,q27,q28,reli, q29,q30,q31,q32,q33,q34]],
                                      columns=['Q2A', 'Q4A', 'Q7A', 'Q9A', 'Q15A', 'Q19A', 'Q20A', 'Q23A', 'Q25A',
       'Q28A', 'Q30A', 'Q36A', 'Q40A', 'Q41A',
       'Extraverted-enthusiastic', 'Critical-quarrelsome',
       'Dependable-self_disciplined', 'Anxious-easily upset',
        'Open to new experiences-complex', 'Reserved-quiet', 'Sympathetic-warm',
        'Disorganized-careless', 'Calm-emotionally_stable',
        'Conventional-uncreative', 'education', 'urban', 'gender', 'engnat',
        'hand', 'religion',
        'orientation', 'race', 'voted', 'married', 'familysize',
        'Age_Groups'],
                                      dtype=float,
                                      index=['input'])  
        #input_variables.to_csv('an1.csv')




        # min-max scaling   
        input_variables = (input_variables - df['min'])/(df['max'] - df['min']) 
        
        #input_variables.to_csv('an2.csv')

        prediction = model_anxiety.predict(input_variables)[0]
        message = 'I have just taken my mental survey, and the prediction is:' + prediction
        chat_messages = [{'role': 'system', 'content': 'The user has just taken a mental survey test, give them an advice, maybe use some quote at the beginning and speak less than 75 words, use some icon'}, {'role': 'user', 'content': message}]
        response = get_chat_response(chat_messages)
        return jsonify({'prediction': prediction, 'response': response})
    #if flask.request.method == 'GET':
    return render_template('Anxiety_survey.html', prediction = None) #homepage is loaded


@app.route('/survey_intro', methods=['GET', 'POST'])
def survey_intro():
# Create the choropleth map
    plotly1 = pd.read_csv('static/model/first_plot.csv')

    fig1 = px.choropleth(plotly1, 
                        locations='country',
                        locationmode='country names',
                        color='% people answered yes',  # Color based on 'As Important' column
                        hover_name='country',  # Hover information
                        color_continuous_scale=px.colors.sequential.Sunset_r,  # Color scale
                        title='<b style = "font-size: 25px; font-family: "Merriweather">About 1 in 5 people said they had experience of anxiety or depression</b><br><br>'
                              '<span>Over <b>119,000 </b> people worldwide were asked:</span><b style=" font-style: italic; color: Crimson;"><br>"Have you ever been anxious or depressed that you could not continue your regular daily activities<br>for two weeks or longer?"</b><br>''<span style="font-size: 16px; font-style: italic;">Percentage of people who answered yes: </span> ',
                        labels={'Percentage of people who answered yes': 'Percentage'},
                        width= 1200,
                        height = 800)

    # Update layout for better appearance (optional)
    fig1.update_layout(geo=dict(showcoastlines=True))


    
    # Specify that you want to display custom data (percentage) when hovering    
    # Convert the Plotly figure to JSON
    graphJSON1 = fig1.to_json()
##############################################################

    plotly2 = pd.read_csv('static/model/second_plot.csv')

# Create a horizontal bar plot
    color_scale = px.colors.qualitative.Plotly
    fig2 = go.Figure(go.Pie(
        labels=plotly2.data,  # Labels: Response categories
        values=plotly2.value,  # Values: Percentage
        textinfo='label+percent',  # Text to display on hover
        hoverinfo='label+percent',  # Information to display on hover
        marker_colors=px.colors.qualitative.Plotly,  # Color scale for the pie slices
        textfont=dict(
        family="Arial, sans-serif",  # Change font family
        size=14,  # Change font size
        color="black"  # Change font color
    )
    ))


    # Update layout
    fig2.update_layout(
        title='<b style = "font-size: 25px;">Around 9 in 10 people worldwide believe mental health is as important or<br>more important than physical health</b>',        
        xaxis=dict(title='Percentage'), 
        yaxis=dict(title='Response'),
        width = 1200,
        height= 600

    )
    graphJSON2 = fig2.to_json()

    plotly3 = pd.read_csv('static/model/third_plot.csv')
    # Plot the bar chart using Plotly
    fig3 = px.bar(plotly3, x='Gender', y='Percentage', color='Answer', barmode='group',
                labels={'Count': 'Count of Responses', 'Gender': 'Gender'})
    fig3.update_layout(title="<b style = 'font-size: 25px'>Depression and Anxiety are not determined by Gender<\b><br>Percentage of Anxiety or Depression Experience by Gender")
    #fig.update_yaxes(range=[0, 50])  # Adjust the range as needed

   
    #fig.update_yaxes(range=[0, 50])  # Adjust the range as needed

    # Show the plot
    graphJSON3 =fig3.to_json()



    plotly4= pd.read_csv('static/model/fourth_plot.csv')
    # Plot the bar chart using Plotly
    fig4 = px.bar(plotly4, x='Percentage', y='age_group', color='age_group', orientation='h',
                labels={'Percentage': 'Percentage', 'age_group': 'Age Group'})
    fig4.update_layout(title='<b style = "font-size: 25px">People tend to be depressed or anxious at an early and mid adulthood period </b><br>Percentage of people who said they experience depression/anxiety by Age Group')
    graphJSON4 = fig4.to_json()

    # print("graphJSON1:", graphJSON1)
    # print("graphJSON2:", graphJSON2)
    # print("graphJSON3:", graphJSON3)
    # print("graphJSON4:", graphJSON4)
 
    return render_template('survey_intro.html', ls=[graphJSON1, graphJSON2, graphJSON3, graphJSON4])




if __name__ == '__main__':
    app.run(debug= True)

