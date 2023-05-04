
from flask import Flask, render_template,redirect, request
import testing_model as naive_bayes

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    result={"s_code":0,"status":"none"}
    return render_template('index.html',s_code=result['s_code'], status=result['status'],textArea="")

@app.route('/', methods=['POST'])
def get_text():
    text = request.form['text']

    if(request.form.get('action1')=='submit' and len(text)>0):
        result={"s_code":1,"status":naive_bayes.run_model([text])}
 
        return render_template('index.html',s_code=result['s_code'], status=result['status'],textArea=text)
    elif(request.form.get('action2')=='reset'):
        return redirect('/')
    else:
        result={"s_code":-1,"status":"Write something in the textbox, Before submit"}
        return render_template('index.html',s_code=result['s_code'], status=result['status'],textArea=text)
        



if __name__ == '__main__':
    app.run(debug=True)
