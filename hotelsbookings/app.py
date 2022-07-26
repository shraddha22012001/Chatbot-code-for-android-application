# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
from flask import Flask, render_template, request
#import pymysql
import json
#import geocoder
import speech_recognition as sr
import pickle
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import random
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)



def speechrecognition():
    r = sr.Recognizer()
# Reading Audio file as source
# listening the audio file and store in audio_text variable
    with sr.Microphone() as source:
         audio_text = r.listen(source)
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        # using google speech recognition
        text = r.recognize_google(audio_text)
        print('Converting audio transcripts into text ...')
        print(text)
    except:
        text=str(0)
        print('Sorry.. run again...')
    return text

with open("qasystem_updated.json", encoding="utf8") as file:
    data = json.load(file)

m=0
userinfo=''
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            # print(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model11 = tflearn.DNN(net)

try:
    model11.load("model.tflearn")
except:
    model11.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model11.save("model.tflearn")

app.config["SECRET_KEY"]='1234' 

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)
model11.load("model.tflearn")


@app.route("/")
def home():
    #session['val']=0
    return render_template("index.html")

@app.route("/opofchecking")
def opofchecking():
    userText = request.args.get('city')
    outdatagot = ""
    text=userText
    f = open("dataset//"+userText+".txt", "r")
    datais=''
    for x in f:
        datais=datais+x+"\n"
    return render_template("output.html",data=datais)


@app.route("/speech")
def speech():
    global m
    textis=speechrecognition()
    #if textis==0:
        #m-=1
    return textis


@app.route("/usersession", methods=['POST'])
def usersession():
    userinfo=''
    userText = request.form.get('msg')
    print(userText)
    #session['secret']='hi123'
    #session['val']=0
    #session['val']=0
    print(m)
    return 'ok'


@app.route("/gettingserverhit", methods=['POST'])
def gettingserverhit():
    global userinfo
    
    #session['secret']='hi123'
    #session['val']=0
  
    count = request.form.get('count')
    m=int(count)
    print(m)
    cityis=''
    userText = request.form.get('msg')
    print ("userText",userText)
    
    if userText=='Update Profile' or userText=='update profile':
        return str('Go to Profile Icon (Top-left corner),\nwhere you will be able to edit your Name,Address,PhoneNumber.\n\nI hope It is helpful!!')
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Thank You!!!')
    
   
    if userText=='No, I still need help' or userText=='no I still need help':
        return str('What can I help you with ?\nUpdate Profile\nSet Security Questions\nFind Route\nFind Main Tourist Spots\nFind Parks\nFind Hotels\nFind Resorts\nHotel Bookings\n')
    
    
    if userText=='Set Security Questions':
        return str('Go to Profile Icon (Top-left corner),\nwhere you can find the set security questions button.\nYou can set your question and answers,\n so that you will be able to easily reset \nyour password in case you forgot your password.\n\nI hope It is helpful!!')
    
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Thank You!!!')
    
    if userText=='No, I still need help' or userText=='no I still need help':
        return str('What can I help you with ?\nUpdate Profile\nSet Security Questions\nFind Route\nFind Main Tourist Spots\nFind Parks\nFind Hotels\nFind Resorts\nHotel Bookings\n')
    
    
    if userText=='Find Route' or userText=='find route' or userText=='find path':
        return str('Go to map icon,Enter your source location.\nEnter your Destination location.(Do not Enter anything if Your source location is\n your current location it is by default.)\nI hope It is helpful!!')
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Thank You!!!')
    
    if userText=='No, I still need help' or userText=='no I still need help':
        return str('What can I help you with ?\nUpdate Profile\nSet Security Questions\nFind Route\nFind Main Tourist Spots\nFind Parks\nFind Hotels\nFind Resorts\nHotel Bookings\n')
    
    if userText=='Main Spots' or userText=='main spots':
        return str('Go to Main Spots,\nSearch the spot that you want to visit in the \nsearch bar.You will get all the details about that\n tourist spot such as\nimages,address,description,route.\n\nI hope It is helpful!!')
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Thank You!!!')
    
    if userText=='No, I still need help' or userText=='no i still need help':
        return str('What can I help you with ?\nUpdate Profile\nSet Security Questions\nFind Route\nFind Main Tourist Spots\nFind Parks\nFind Hotels\nFind Resorts\nHotel Bookings\n')
    
    if userText=='Find Parks' or userText=='find parks':
        return str('Go to Parks icon,\nSearch any park that you \nwould like to visit in the search bar.You are able \nto see all details about that park such as\nimages,address,\ndescription,route.\n\nI hope It is helpful!!')
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Thank You!!!')
    
    if userText=='No, I still need help' or userText=='no I still need help':
        return str('What can I help you with ?\nUpdate Profile\nSet Security Questions\nFind Route\nFind Main Tourist Spots\nFind Parks\nFind Hotels\nFind Resorts\nHotel Bookings\n')
    
    
    if userText=='Find Resorts' or userText=='find resorts':
        return str('Go to Resorts icon,\nSearch the resort\nthat you want to stay in \nthe search bar.You will be able to see all\n details about that resort such asimages,address,description,route.\n\nI hope It is helpful!!')
    
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Thank You!!!')
    
    if userText=='No, I still need help' or userText=='no i still need help':
        return str('What can I help you with ?\nUpdate Profile\nSet Security Questions\nFind Route\nFind Main Tourist Spots\nFind Parks\nFind Hotels\nFind Resorts\nHotel Bookings\n')
    
    
    if userText=='Find Hotels' or userText=='find hotels':
        return str('Go to Hotels icon,\nSearch the Hotel that you want to stay in the search bar.\nYou will be able to see all details about that hotel such as\nimages,address,description,route,Check-in time,\nCheck-out time,facilities,rooms,cost\nI hope It is helpful!!')
    
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Thank You!!!')
    
    if userText=='No, I still need help' or userText=='no I still need help':
        return str('What can I help you with ?\nUpdate Profile\nSet Security Questions\nFind Route\nFind Main Tourist Spots\nFind Parks\nFind Hotels\nFind Resorts\nHotel Bookings\n')
    
    if userText=='Hotel Bookings' or userText=='hotel booking' or userText=='hotel bookings' or userText=='Hotel bookings':
        return str('Go to Hotels icon,\nSearch the Hotel that you want to stay in the search bar.\nYou will be able to see all details about that hotel such as\nimages,address,description,route,Check-in time,\nCheck-out time,facilities,rooms,cost\n\nI hope It is helpful!!')
    
    
    if userText=='Yes, Thank you !' or userText=='yes thank you':
        return str('Go to Profile Icon (Top-left corner),\nwhere you will be able to edit your Name,Address,PhoneNumber.\n\nI hope It is helpful!!')
    
    
    if userText=='No, I still need help' or userText=='no I still need help':
        return str('Go to Profile Icon (Top-left corner),\nwhere you will be able to edit your Name,Address,PhoneNumber.\n\nI hope It is helpful!!')
    
    
    
    userinfo=userinfo+" "+userText
    #m=5
    outdatagot=''
   
    
    
    #if m == 19:
    print(userinfo)
    results = model11.predict([bag_of_words(userinfo, words)])
    results_index = numpy.argmax(results)
    print(results_index)
    print(labels)
    tag = labels[results_index]
    print(tag)
    cityis=userinfo.split(" ")
    outdatagot = ""
    text=userText
    #datais="<a href='/opofchecking?city="+cityis[2]+"'>click here for iternary"+"</a>"
  
    for tg in data["intents"]:
         if tg['tag'] == tag:
            responses = tg['responses']
            outdatagot = random.choice(responses)
            m=0
            print(outdatagot)
            return outdatagot

    return str(outdatagot+"\n")


@app.route("/get")
def get_bot_response():
    global m,userinfo
    cityis=''
    userText = request.args.get('msg')
    userinfo=userinfo+" "+userText
    #m=5

    if m==0:
        outdatagot="Hi, I am your travel buddy., <br>What can I help you with ?"
        m+=1
        return str(outdatagot)
    if m==1:
        #cityis=userText
        outdatagot="what is your budget"
        m+=1
        return str(outdatagot)
    if m==2:
        outdatagot="how many number of peoples"
        m+=1
        return str(outdatagot)
    if m==3:
        outdatagot="is this family trip"
        m+=1
        return str(outdatagot)
    if m == 4:
        outdatagot = "what type of hotels do you want <br> AC <br> Luxury <br> Deluxe"
        m += 1
        return str(outdatagot)
    if m == 5:
        outdatagot = "Which type of places you wan't to  travel<br> Fort<br> Historical Places<br> Hill Station <br> Beach >"
        m += 1
        return str(outdatagot)
    if m == 6:
        outdatagot = "How many days you wan't to stay "
        m += 1
        return str(outdatagot)
    if m == 7:
        print(userinfo)
        results = model11.predict([bag_of_words(userinfo, words)])
        results_index = numpy.argmax(results)
        print(results_index)
        print(labels)
        tag = labels[results_index]
        print(tag)
        cityis=userinfo.split(" ")
        outdatagot = ""
        text=userText
        datais="<a href='/opofchecking?city="+cityis[2]+"'>click here for iternary"+"</a>"
        #f = open("dataset//"+cityis[2]+".txt", "r")
        #datais=''
        #for x in f:
            #datais=datais+x+"\n"
        for tg in data["intents"]:
             if tg['tag'] == tag:
                responses = tg['responses']
                outdatagot = random.choice(responses)
                m=0
                print(outdatagot)

    return str(outdatagot+"\n"+datais)

@app.route("/rr", methods=['POST','GET'])
def rr():
    return "suss"

        





# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run('0.0.0.0')
